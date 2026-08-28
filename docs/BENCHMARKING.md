# Benchmarking and acceptance criteria

## Goals

The benchmark must answer four separate questions:

1. Does the system locate the watermark reliably?
2. Does the mask remove the watermark without erasing valid photograph content?
3. Does the restoration preserve the photograph and reconstruct plausible pixels?
4. Does the complete pipeline remain fast, bounded, and recoverable under failure?

No single metric answers all four. End-to-end images/second alone can hide false negatives, corrupt outputs, metadata loss, or a queue that still contains unwritten results when the timer stops.

## Benchmark collections

### Synthetic paired set

Create clean source images and apply known watermark assets programmatically. This provides exact clean ground truth, exact alpha masks, layout parameters, and reproducible variants.

The generator should vary:

- source resolution and aspect ratio;
- JPEG quality and repeated recompression;
- watermark location, scale, rotation, and opacity;
- monochrome and multicolor logos;
- drop shadows, outlines, glows, and anti-aliasing;
- simple, textured, high-frequency, architectural, face, skin, sky, foliage, and text backgrounds;
- single versus multiple watermarks;
- watermark area as a fraction of image area;
- complete absence of a watermark for false-positive testing.

Synthetic watermark compositing must be performed in a defined color space and record the exact operation. When testing inverse-alpha recovery, include deviations such as gamma-space blending and unknown opacity to measure robustness.

### Redistributable public set

Maintain a small set of redistributable images with hand-reviewed watermark regions and expected outcomes. This catches unrealistic assumptions in the synthetic generator.

Only assets with clear redistribution terms may be committed.

### Private personal-photo golden set

Use a local manifest that is never committed. Select a stratified set of the user's photographs covering:

- common portrait and landscape export layouts;
- all known old-watermark variants;
- faces and skin under the mark;
- fabric, hair, foliage, water, architecture, and text;
- images where the watermark is partially outside the frame;
- very small and very large images;
- original camera JPEG, edited JPEG, PNG, TIFF, and other relevant formats;
- metadata profiles that must be preserved.

The manifest may store hashes, dimensions, categories, and human review scores without storing image pixels in the repository.

## Detection and mask metrics

For annotated images, report:

- image-level watermark presence precision, recall, and F1;
- false-positive rate on clean images;
- number of missed watermarks per thousand images;
- box average precision where box labels exist;
- mask intersection-over-union;
- mask precision and recall;
- boundary F-score or boundary distance;
- over-mask pixel count: valid source pixels unnecessarily selected;
- under-mask pixel count: watermark pixels left outside the mask;
- confidence calibration and review rate at each threshold.

Metrics must be broken down by watermark opacity, size, location, and background class. An aggregate score can hide failure on faint watermarks.

## Restoration metrics

### Exact preservation outside the mask

Before lossy output encoding, pixels outside the mask and permitted blend band should be exactly equal to the decoded source.

Report:

```text
outside-mask changed pixel count
maximum absolute channel difference
mean absolute channel difference
```

A backend that returns a modified full image must be composited into the source so this invariant holds.

### Paired synthetic metrics

Compare the restored image against the clean source using:

- PSNR;
- SSIM or MS-SSIM;
- LPIPS;
- optional DISTS or another perceptual metric if justified;
- metrics inside the original watermark support;
- metrics inside a halo band around the support;
- seam/edge gradient discontinuity;
- color difference such as Delta E where color management is controlled.

Report metrics separately for transparent pixels, opaque pixels, and the transition band.

### Human review

Generative plausibility is not the same as historical fidelity. Reviewers should score:

- watermark residue;
- visible seam or blur;
- invented content;
- damage to faces, skin, hair, text, architecture, and repeated texture;
- acceptability at 100% view and common delivery size;
- whether manual correction is required.

Use blinded side-by-side comparisons where practical. Preserve the model seed and settings for reproducibility.

## Performance metrics

Report complete-pipeline throughput only after all output commits and ledger transactions have completed.

Required metrics:

- committed images/second;
- committed megapixels/second;
- bytes read and written per second;
- p50, p90, p95, and p99 total latency;
- stage latency for discovery, decode, locate, mask, ROI, restore, composite, encode, metadata, commit, and ledger;
- queue wait time and occupancy;
- GPU utilization and per-GPU committed throughput;
- peak GPU memory by worker;
- peak process and total resident memory;
- startup/model-load time;
- proportion of time spent on watermark-positive and watermark-negative paths.

For GPU timing, use CUDA events or explicit synchronization. For CPU and end-to-end timing, use `time.perf_counter_ns()`.

## Benchmark matrix

At minimum, benchmark these dimensions:

### Resolution buckets

```text
<= 1 MP
1–4 MP
4–12 MP
12–24 MP
> 24 MP
```

### Formats

```text
JPEG
PNG
WebP
TIFF where supported
```

### Watermark area

```text
none
< 0.25%
0.25–1%
1–5%
> 5%
```

### Backend paths

```text
no-detection direct copy
known-alpha inverse recovery
OpenCV classical
LaMa full-frame baseline
LaMa ROI
candidate fast backend
candidate diffusion fallback
```

### Concurrency

```text
one GPU
all GPUs
1/2/4/8 local encoder workers
multiple queue byte limits
small detector batch sizes
```

Do not assume the optimal writer/encoder count equals `os.cpu_count()`.

## Baseline protocol

Before substantial refactoring:

1. freeze a representative benchmark manifest;
2. record environment information;
3. hash current detector and LaMa artifacts;
4. run the current implementation to establish output and timing baselines;
5. record known hangs, failed writes, and shutdown behavior separately rather than excluding them;
6. retain a small set of current output images for visual regression comparison where licensing permits.

Environment information includes Python, PyTorch, Ultralytics, CUDA runtime, NVIDIA driver, OpenCV, Pillow, CPU, GPU, storage path/filesystem, and relevant codec versions.

## Reliability and failure-injection benchmark

The test harness must inject failures at deterministic job numbers and prove that the coordinator terminates cleanly and resumes correctly.

Scenarios include:

- corrupt input;
- detector exception;
- inpainting exception;
- encoder returns `False`;
- disk full or permission failure;
- metadata failure;
- temporary-file validation failure;
- atomic rename failure;
- GPU worker exits unexpectedly;
- commit worker exits unexpectedly;
- full output queue/backpressure;
- interrupt while detecting;
- interrupt while inpainting;
- interrupt while encoding;
- interrupt after rename but before ledger update;
- restart with stale temporary files;
- restart with an output missing despite a legacy text checkpoint;
- duplicate retry of the same failing path.

Required assertions:

- no false success record;
- exactly one terminal result per attempt;
- no indefinite wait after all workers exit;
- no duplicate current failure row;
- committed outputs remain valid;
- unfinished jobs remain eligible on resume;
- successful jobs are not unnecessarily reprocessed under the same fingerprint.

## Model comparison policy

A model comparison report must include:

- exact model and artifact hash;
- code/runtime version;
- license notes;
- configuration and seed;
- quality metrics by subset;
- committed throughput and memory;
- failure/review rate;
- representative successes and failures;
- recommendation: reject, experimental, optional, or default candidate.

The default backend changes only when a documented ADR explains why the benchmark evidence justifies the migration.

## Initial acceptance gates

### Correctness gate

- zero known false checkpoint records in failure-injection tests;
- no indefinite coordinator wait after worker failure;
- graceful first-interrupt drain is proven;
- writer/encoder return values are checked;
- output commits are atomic;
- failure de-duplication is enforced by the ledger;
- existing CLI and relative output paths remain compatible.

### Fidelity gate

- unmasked pixels are unchanged before encoding;
- no material increase in watermark misses on the frozen benchmark;
- ROI output is visually and quantitatively non-inferior to full-frame baseline for accepted cases;
- metadata behavior is explicit and tested.

### Performance gate

No universal percentage is specified before baseline measurement. A change is accepted when it improves a measured bottleneck without violating correctness/fidelity gates. Stage-level evidence must explain the gain.

### Model-default gate

A new default must outperform the incumbent on the intended workload, have acceptable license terms, and not impose an unreasonable installation/runtime burden. A backend may still be valuable as an optional fallback without becoming the default.
