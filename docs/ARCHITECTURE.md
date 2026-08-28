# Architecture

## Scope

This document describes the architecture reviewed from `main` at commit `ca543dcf6abd1fc19af4747ac67f9afe259071a6` and the target architecture for the modernization work.

The target design preserves the useful parts of the existing implementation—one long-lived model owner per GPU, bounded backpressure, recursive path preservation, and resumability—while separating orchestration from model frameworks and fixing the completion boundary.

## Current architecture

The current repository is a single Python script with four broad responsibilities:

1. CLI parsing, file discovery, and checkpoint loading.
2. One spawned GPU process per detected NVIDIA GPU.
3. A shared pool of CPU writer processes consuming full NumPy image arrays from a multiprocessing queue.
4. Main-process status collection, checkpoint appends, Rich rendering, and process shutdown.

Each GPU process statically receives a round-robin slice of input paths, loads one Ultralytics YOLO model and one `SimpleLama` instance, decodes each image, detects bounding boxes, creates a filled rectangular mask, optionally dilates it, runs full-frame LaMa when a detection exists, and queues the full result array for a writer.

This architecture is compact and achieved high throughput on the original approximately one-megapixel workload. It also has several correctness and scalability limitations.

## Known weaknesses in the current implementation

### Completion is reported before durable output

The GPU worker sends progress immediately after placing an array in the writer queue. The coordinator can reach the expected total and terminate writer processes while outputs are still queued or being encoded. The checkpoint is therefore not a durable commit log.

### Failure can prevent loop termination

Unreadable images and processing exceptions do not emit a terminal progress result. The main loop still waits for the original input count, so a failed job can leave the application waiting after workers have exited.

### Writer failures are unobservable

`cv2.imwrite()` returns a success flag that is not checked. Broad writer exceptions are swallowed. A failed write can therefore be treated as a successful checkpoint.

### Shutdown is destructive rather than draining

Normal completion and Ctrl+C both lead to process termination rather than an ordered drain of GPU work, writer queues, result acknowledgements, and the checkpoint/ledger.

### Static scheduling can leave GPUs idle

Round-robin path slices assume every image costs approximately the same. Decode time, resolution, watermark hit rate, detected area, and inpainting cost vary substantially.

### Large arrays cross process boundaries

Decoded or inpainted full-frame arrays are serialized through `multiprocessing.Queue`. For high-resolution photographs this creates avoidable memory copies, RAM pressure, and queue latency.

### Full-frame inpainting is used for small masks

The current `simple-lama-inpainting` path converts and processes the whole image even when a watermark occupies a small corner. This increases CPU conversion, host/device transfer, inference, and postprocessing work.

### The mask is a filled bounding box

Valid pixels inside transparent logo regions and between letters are erased. A precise segmentation or known alpha mask would preserve more source content.

### No-detection images are needlessly re-encoded

Images with no detection are decoded, transferred to a writer, and encoded again rather than copied byte-for-byte. This wastes CPU time, loses metadata, and introduces another lossy JPEG generation.

### Checkpoints are not configuration-aware

A path is skipped after changing detector weights, confidence, dilation, inpainting model, ROI policy, or output encoding settings. The old and new outputs are not distinguishable by the checkpoint.

### Image metadata and fidelity are not first-class

The current OpenCV path does not explicitly preserve EXIF, IPTC, XMP, ICC profiles, orientation semantics, alpha channels, or high-bit-depth data.

## Target architecture

The target system is a staged pipeline with small typed messages and framework-independent backend contracts.

```text
Scanner / manifest builder
          |
          v
Durable job ledger + bounded dynamic scheduler
          |
     +----+----+
     |         |
     v         v
GPU worker 0  GPU worker 1 ...
     |         |
Decode prefetch / image normalization
     |
Locator -> mask builder/refiner -> ROI planner
     |
Restoration policy router
     |---- no mark: copy/review
     |---- known alpha: inverse composite
     |---- standard: fast inpainter
     `---- difficult: optional quality fallback
     |
Bounded local encoder/commit workers
          |
Temporary output -> validation -> metadata -> atomic rename
          |
Commit acknowledgement
          |
Ledger transaction + progress/telemetry
```

## Architectural layers

### 1. Domain layer

The domain layer has no dependency on Ultralytics, PyTorch, OpenCV model wrappers, or Diffusers. It defines stable types such as:

- `JobId`
- `ImageJob`
- `ImageInfo`
- `Region`
- `Detection`
- `MaskResult`
- `RoiPlan`
- `RestorationRequest`
- `RestorationResult`
- `OutputCommit`
- `StageTiming`
- `TerminalResult`

Coordinates should have an explicit convention. Pixel-space rectangles use half-open bounds `[x1, x2)` and `[y1, y2)` to avoid off-by-one ambiguity. Masks use a documented polarity and `uint8` representation at boundaries.

### 2. Backend contracts

Backends implement small protocols and return domain types rather than framework objects.

```python
class LocatorBackend(Protocol):
    def load(self, device: DeviceSpec) -> None: ...
    def locate(self, images: Sequence[ImageFrame]) -> Sequence[LocationResult]: ...
    def close(self) -> None: ...

class MaskBackend(Protocol):
    def build_mask(self, image: ImageFrame, locations: LocationResult) -> MaskResult: ...

class InpaintBackend(Protocol):
    def load(self, device: DeviceSpec) -> None: ...
    def inpaint(self, request: RestorationRequest) -> RestorationResult: ...
    def close(self) -> None: ...
```

Capabilities are declared rather than inferred:

- supports batches;
- accepts arbitrary resolution;
- required dimension multiple;
- maximum recommended pixels;
- deterministic seed support;
- prompt requirement;
- supports alpha-aware reconstruction;
- preserves unmasked pixels internally;
- runtime/device support;
- preferred precision;
- license and weight-license identifiers.

A built-in registry resolves backend IDs to factories. External Python entry points may be added later, but the first implementation should avoid a complex plugin framework until built-in contracts are stable.

### 3. Model manifest and configuration

Every loaded artifact receives a manifest containing at least:

```text
backend_id
backend_version
task
artifact_path_or_source
sha256
runtime
code_license
weights_license
input_contract
output_contract
recommended_settings
benchmark_record
```

The processing configuration is normalized and hashed. The fingerprint includes all settings that can materially change output, including:

- locator backend and weights hash;
- confidence/NMS/end-to-end settings;
- mask strategy and dilation radius;
- ROI padding and fallback threshold;
- restoration backend and weights hash;
- seed, steps, prompt, and precision where relevant;
- output codec, quality, bit depth, and metadata policy;
- application version.

The ledger binds each committed output to this fingerprint.

### 4. Scheduler and workers

The coordinator owns the ledger, job scheduling, progress aggregation, and shutdown state. It sends paths and compact metadata—not full decoded images—to GPU workers through a bounded dynamic queue.

Each GPU worker owns all models assigned to that device. It may use a small thread pool for decode prefetch and a small bounded thread pool for encoding/commit work. Passing a NumPy array to a thread is reference-based and avoids multiprocessing serialization.

The local output queue must be bounded by both item count and estimated bytes. A queue of twenty one-megapixel images is not equivalent to a queue of twenty 45-megapixel TIFFs.

OpenCV and other native libraries may create internal thread pools. Worker startup must set and benchmark thread counts to avoid oversubscription across Python processes and encoder threads.

### 5. ROI planner and compositor

The ROI planner converts one or more mask components into context-padded crops:

1. merge overlapping or nearby mask components;
2. expand each component by configurable context;
3. clamp to image bounds;
4. align dimensions to the backend's required multiple;
5. split or fall back when an ROI exceeds backend limits;
6. record transforms so the result can be mapped back exactly.

Only masked pixels, plus an explicitly controlled feathering band where needed, are composited into the source image. Unmasked pixels outside that band must remain unchanged.

Full-frame inference remains a fallback for masks that cover a large fraction of the image or where a backend requires global context.

### 6. Restoration policy router

The router selects the least destructive method that satisfies the request:

1. **Known watermark inverse compositing** when the logo alpha/color/blend assumptions are available and alignment confidence is high.
2. **Classical local inpainting** for tiny marks on simple backgrounds when validated by the benchmark.
3. **Fast learned inpainting** such as LaMa for the normal high-throughput path.
4. **Alternative GAN/transformer backend** for cases where it beats LaMa on the benchmark.
5. **Diffusion fallback** only for difficult cases and only when explicitly enabled.
6. **Review** rather than fabricated confidence when no method meets thresholds.

The policy and fallback path are recorded in output provenance.

### 7. Output commit service

Output completion follows a two-phase pattern:

1. Prepare and validate a temporary file in the destination filesystem.
2. Atomically replace the final path and acknowledge the commit.

The commit service is responsible for codec settings, metadata policy, output validation, directory creation, and cleanup of stale temporary files.

The ledger is updated only after acknowledgement. See [`PROCESSING_SEMANTICS.md`](PROCESSING_SEMANTICS.md).

### 8. Observability

Workers emit structured stage events. The Rich UI consumes aggregated metrics but is not part of the correctness path. Headless JSONL or machine-readable reporting must be possible for benchmarks and long unattended runs.

Logging, telemetry, and progress queues are bounded or coalesced so a slow terminal cannot block image processing.

## Proposed source layout

The single-file CLI should be migrated incrementally into a package while keeping `watermark_remover.py` as a compatibility entry point during the transition.

```text
watermark_remover.py                 # compatibility launcher
pyproject.toml
src/watermark_remover/
  __init__.py
  cli.py
  config.py
  domain.py
  pipeline.py
  scheduler.py
  ledger.py
  telemetry.py
  io/
    discovery.py
    decode.py
    commit.py
    metadata.py
  processing/
    masks.py
    roi.py
    composite.py
    policy.py
  backends/
    protocols.py
    registry.py
    locators/
      ultralytics.py
      template.py
      static_region.py
    masks/
      boxes.py
      segmentation.py
      alpha_template.py
    inpainters/
      lama.py
      opencv.py
      migan.py
      diffusers.py
  workers/
    gpu_worker.py
    local_io.py
tests/
  unit/
  integration/
  failure_injection/
  fixtures/
```

Optional backends may be placed behind dependency extras such as `.[migan]`, `.[diffusion]`, or `.[sam]`.

## Compatibility and migration

The initial modernization must retain the existing CLI flags and output directory structure. New configuration may be added without forcing existing users to adopt a config file.

Checkpoint migration must be explicit. Existing `.processing_log.txt` entries may be imported into the new ledger as legacy successes only after checking that the referenced output exists and is decodable. Because the old log does not record configuration fingerprints, imported entries must be marked `legacy_unverified` rather than silently treated as equivalent to new commits.

The legacy text log can remain as an optional human-readable export, but SQLite should become the authoritative ledger because it provides uniqueness constraints, transactions, typed status, attempt history, and indexed resume queries.

## Rejected shortcuts

- Replacing YOLO11 with YOLO26 without retraining and task-specific validation.
- Replacing LaMa with a diffusion model solely because it is newer.
- Adding more writer processes without measuring storage and codec saturation.
- Marking progress at inference completion rather than output commit.
- Keeping broad `except Exception: continue` blocks in correctness-critical code.
- Running every optional model in the same environment by default.
- Modifying the entire source image with a generative backend when only a small mask requires reconstruction.
