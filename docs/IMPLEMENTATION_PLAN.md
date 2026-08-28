# Modernization implementation plan

## Objective

Modernize `watermark_remover` into a safe, measurable, model-replaceable, high-throughput restoration pipeline while preserving the useful behavior of the current tool:

- recursive discovery and relative output paths;
- multi-GPU execution;
- long-lived models per GPU;
- interruption and resume;
- Rich progress/status reporting;
- current CLI compatibility during migration.

The plan deliberately fixes correctness before optimizing throughput and establishes a benchmark before changing default models.

## Guiding decisions

- The durable completion boundary is the acknowledged atomic output commit, not GPU inference completion.
- SQLite becomes the authoritative ledger; text logs become compatibility exports.
- Model frameworks are isolated behind typed locator, mask, and inpainting contracts.
- The current custom YOLO11 checkpoint and LaMa output remain the initial regression baseline.
- ROI-first processing and exact source compositing are core architecture, not backend-specific optimizations.
- Known-watermark template/alpha reconstruction is a first-class path for personal photos.
- Diffusion backends are optional quality fallbacks rather than the default bulk path.
- Model changes require reproducible watermark-specific benchmarks and documented licenses.

## Stage 0 — Planning and architecture documentation

- [x] Review the current `main` implementation and identify correctness/performance risks.
- [x] Survey current detector, segmentation, and inpainting options as of 2026-08-27.
- [x] Add repository agent instructions.
- [x] Document the current and target architecture.
- [x] Define durable processing/checkpoint semantics.
- [x] Define model-backend selection and licensing policy.
- [x] Define benchmark and failure-injection requirements.
- [x] Record initial architecture decisions under `docs/adr/`.
- [ ] Link the documentation index from the public README.

**Exit condition:** the PR and documents agree on scope, invariants, stages, and model-evaluation policy.

## Stage 1 — Test foundation and immediate correctness fixes

This stage intentionally minimizes architectural change while making the current script safe enough to benchmark.

### Repository/test foundation

- [ ] Add `pyproject.toml` with pinned/minimum supported dependencies and optional groups.
- [ ] Add formatting, linting, type-checking, and pytest configuration.
- [ ] Add small programmatically generated image fixtures.
- [ ] Add fake detector, inpainting, and writer components for deterministic tests.
- [ ] Add a fast local check script and a separate GPU/model integration test marker.

### Input and CLI validation

- [ ] Validate input directory, output root, weights path, confidence range, dilation, and worker counts.
- [ ] Resolve input/output roots and reject recursive output-inside-input configurations unless explicitly safe.
- [ ] Replace filename `endswith()` extension matching with suffix-based matching.
- [ ] Define symlink behavior and prevent path traversal outside output root.
- [ ] Remove unused imports only where unrelated behavior/comments remain untouched.

### Terminal outcomes and liveness

- [ ] Emit exactly one terminal result for each attempted input.
- [ ] Represent decode, detection, mask, inpainting, and output failures as typed results.
- [ ] Detect dead GPU/writer workers and resolve their in-flight jobs.
- [ ] Make coordinator termination depend on worker/queue state plus terminal results, not only a numeric progress count.
- [ ] Add tests proving a corrupt image or model exception cannot cause an infinite wait.

### Writer correctness

- [ ] Check `cv2.imwrite()`/encoder success explicitly.
- [ ] Stop swallowing writer exceptions.
- [ ] Write to a destination-local temporary file, validate it, and atomically replace the final output.
- [ ] Send success acknowledgement only after atomic commit.
- [ ] Count progress/checkpoint success from acknowledgements rather than GPU queue submission.
- [ ] Reconcile stale temporary files on startup.

### Pause/resume and failure de-duplication

- [ ] Preserve current first-Ctrl+C pause behavior at the CLI level while changing internals to drain safely.
- [ ] Stop new submissions on first interrupt, finish bounded in-flight work, drain writers, flush results, and exit.
- [ ] Use second Ctrl+C for forceful termination.
- [ ] Ensure normal completion also drains queues instead of terminating healthy writers.
- [ ] De-duplicate `processing_failures.txt` entries without altering resume semantics.
- [ ] Add failure attempt counts and latest-error reporting.
- [ ] Validate legacy checkpoint outputs before skipping them.

**Exit condition:** all failure-injection tests pass, no false success is possible, and current CLI/output behavior remains compatible.

## Stage 2 — Package extraction and stable backend contracts

Migrate incrementally from the single script into `src/watermark_remover/` while retaining `watermark_remover.py` as a compatibility launcher.

- [ ] Define domain dataclasses/enums for jobs, detections, masks, ROIs, timings, commits, and terminal results.
- [ ] Define `LocatorBackend`, `MaskBackend`, and `InpaintBackend` protocols.
- [ ] Define backend capability metadata and typed configuration.
- [ ] Add a built-in backend registry.
- [ ] Implement a model/artifact manifest with SHA-256 and license/provenance fields.
- [ ] Normalize processing configuration and compute a stable configuration fingerprint.
- [ ] Extract discovery/path handling.
- [ ] Extract telemetry/Rich rendering from correctness logic.
- [ ] Wrap the current custom Ultralytics checkpoint in an `ultralytics` locator adapter.
- [ ] Implement box-mask behavior as a separate mask adapter to reproduce current output.
- [ ] Implement an internal LaMa adapter that reproduces the existing baseline.
- [ ] Add contract tests that do not require large weights.
- [ ] Add compatibility tests comparing the legacy path and adapters on small fixtures.

**Exit condition:** orchestration imports only backend protocols/domain types; Ultralytics and LaMa objects do not leak into coordinator logic.

## Stage 3 — Durable SQLite ledger and provenance

- [ ] Add SQLite schema and migrations for jobs, attempts, outputs, run configuration, and model provenance.
- [ ] Make the coordinator the single ledger writer or use one explicit serialized ledger service.
- [ ] Add primary-key/unique constraints that naturally de-duplicate current failures.
- [ ] Record input identity, attempts, typed errors, output identity, timings, and configuration fingerprint.
- [ ] Import existing `.processing_log.txt` as `legacy_unverified` only after output validation.
- [ ] Export human-readable success/failure text reports for compatibility.
- [ ] Reject, namespace, or explicitly reprocess configuration mismatches.
- [ ] Add reconciliation for the crash window between output rename and ledger update.
- [ ] Add migration/recovery tests.

**Exit condition:** SQLite is the authoritative resume source and every committed output is bound to model/configuration provenance.

## Stage 4 — Measurement and reproducible baseline

- [ ] Add structured per-stage timing events.
- [ ] Use CUDA events/synchronization for detector and inpainting GPU timing.
- [ ] Record decode, locate, mask, ROI, restore, composite, queue wait, encode, metadata, commit, and ledger timing.
- [ ] Record image dimensions, format, watermark hit, detected/masked area, and selected backend path.
- [ ] Record queue occupancy, blocked time, RAM, VRAM, and per-GPU committed throughput.
- [ ] Add machine-readable benchmark output plus a human summary.
- [ ] Build synthetic paired benchmark generation.
- [ ] Create a frozen local benchmark manifest.
- [ ] Benchmark current full-frame YOLO11 + LaMa behavior before performance refactors.
- [ ] Document baseline results, environment, and known failure behavior.

**Exit condition:** bottlenecks are demonstrated with data and every later optimization can be compared to a frozen baseline.

## Stage 5 — I/O and scheduling efficiency

### Dynamic scheduling

- [ ] Replace static round-robin image slices with a bounded dynamic path/job queue.
- [ ] Track in-flight job ownership per worker.
- [ ] Support heterogeneous GPU capability records without assuming identical speed.
- [ ] Add fairness and shutdown tests.

### Decode prefetch

- [ ] Add a small bounded decode-prefetch pool per GPU worker.
- [ ] Benchmark OpenCV/Pillow/libvips-compatible decode paths where relevant.
- [ ] Control native library thread counts to prevent oversubscription.
- [ ] Bound decoded-image memory by bytes, not only item count.

### Local encoding and commit

- [ ] Replace full-frame NumPy transfers through `multiprocessing.Queue` with bounded local thread-based encode/commit workers or another measured zero/low-copy design.
- [ ] Benchmark 1/2/4/8 local encoder workers rather than defaulting to all logical CPUs.
- [ ] Add direct byte-copy jobs for no-watermark images.
- [ ] Preserve timestamps and configured metadata on direct copies.
- [ ] Add output codec/quality controls and lossless cleaned-master mode.

### Detector batching

- [ ] Add configurable small detector microbatches.
- [ ] Bucket by dimensions where needed to limit padding waste.
- [ ] Measure latency, throughput, and VRAM before selecting defaults.

**Exit condition:** large image arrays no longer cross a shared process queue by default, load balances dynamically, and throughput gains are documented.

## Stage 6 — ROI, masks, and source-preserving compositing

- [ ] Define dilation as a radius in pixels and use a `2r + 1` kernel where appropriate.
- [ ] Convert detector outputs to one canonical mask convention.
- [ ] Merge nearby mask components using configurable geometry.
- [ ] Add context-padded ROI planning with backend alignment multiples.
- [ ] Add configurable full-frame fallback based on ROI/image area and backend limits.
- [ ] Run LaMa only on planned crops for the normal watermark path.
- [ ] Composite only masked pixels plus an explicit feather band into the original decoded image.
- [ ] Assert exact outside-mask preservation before encoding.
- [ ] Add edge/seam tests and multi-component mask tests.
- [ ] Benchmark ROI versus full-frame LaMa by resolution and mask area.

**Exit condition:** ROI LaMa is non-inferior on the accepted quality set, preserves unmasked pixels, and materially reduces processing cost for small watermarks.

## Stage 7 — Detector and mask modernization experiments

### YOLO experiments

- [ ] Inventory the training data and exact current YOLO11 checkpoint provenance.
- [ ] Build reproducible train/validation/test splits with clean negative images.
- [ ] Preserve a YOLO11 baseline training/evaluation recipe.
- [ ] Benchmark smaller YOLO11 detector scales.
- [ ] Train/evaluate YOLO26 detection variants.
- [ ] Evaluate YOLO26-P2 for small watermarks.
- [ ] Build segmentation labels from known alpha masks and manual corrections.
- [ ] Train/evaluate YOLO26 segmentation variants.
- [ ] Compare end-to-end and one-to-many heads where applicable.
- [ ] Evaluate export runtimes only after PyTorch correctness parity.

### Optional refinement/research

- [ ] Evaluate SAM 3 as an annotation tool and box-to-mask refiner on a bounded sample.
- [ ] Evaluate visual-prompt localization using the old logo as an exemplar.
- [ ] Reject these paths if their quality gain does not justify runtime/maintenance cost.

**Exit condition:** retain YOLO11 or adopt a new detector/mask default based on watermark-specific recall, false positives, mask quality, speed, and an ADR.

## Stage 8 — Inpainting backend experiments

- [ ] Add an OpenCV Telea/Navier–Stokes backend as a fast CPU baseline.
- [ ] Complete and optimize the internal LaMa backend.
- [ ] Evaluate MI-GAN/ONNX after verifying weight-license terms; do not redistribute unclear artifacts.
- [ ] Evaluate a structural model such as MAT/ZITS++ only if a maintainable artifact/runtime is available.
- [ ] Add one optional diffusion-quality backend, initially PowerPaint v2.1 or BrushNetX, behind optional dependencies.
- [ ] Add a generic Diffusers adapter only with an allowlisted model profile system.
- [ ] Treat FLUX.1 Fill [dev] as an explicit non-commercial research backend, never an automatic default/download.
- [ ] Track dedicated visible-watermark restoration research and verify code/weights before integration.
- [ ] Record seeds/prompts/steps and enforce mask-only compositing for generative backends.
- [ ] Produce a model comparison report and ADR before any default change.

**Exit condition:** the project has at least one stable fast default and an optional measured quality fallback, with clear artifact/license provenance.

## Stage 9 — Personal-photo known-watermark workflow

This stage is designed for the user's own old-business photographs.

- [ ] Search for unwatermarked masters before reconstructing pixels.
- [ ] Inventory formats, resolutions, metadata, orientations, and old-watermark variants without modifying files.
- [ ] Recover the original watermark PNG/PSD/export preset if available.
- [ ] Implement template/layout profiles for portrait, landscape, and known variants.
- [ ] Estimate alignment, opacity, and blend behavior with confidence scores.
- [ ] Implement stable inverse-alpha recovery for partially transparent pixels.
- [ ] Route opaque/unstable residual pixels to ROI inpainting.
- [ ] Add YOLO segmentation fallback for unexpected layouts.
- [ ] Add `--expect-watermark` and review manifests for no-detection/low-confidence images.
- [ ] Preserve EXIF/IPTC/XMP/ICC and original timestamps according to explicit policy.
- [ ] Write cleaned archival intermediates losslessly.
- [ ] Never modify originals in place.
- [ ] Generate contact sheets or a local review report for flagged cases without committing private images.
- [ ] Design the future `replace-watermark` path to decode once, remove the old mark, add the new mark, and encode once.

**Exit condition:** a dry-run inventory and reviewed pilot batch pass before bulk processing begins.

## Stage 10 — Packaging, CI, and release documentation

- [ ] Provide reproducible installation instructions and a lock/constraints strategy.
- [ ] Separate default, GPU, detector-training, MI-GAN, SAM, and diffusion dependencies.
- [ ] Add CPU-only CI for core logic and fake-backend integration tests.
- [ ] Add optional local GPU check scripts.
- [ ] Add CLI reference and migration guide from the single-script version.
- [ ] Document artifact downloads, hashes, and licenses.
- [ ] Update README architecture/performance claims using measured results.
- [ ] Document limitations, expected review cases, and responsible-use requirements.
- [ ] Run the complete test/benchmark acceptance suite.

**Exit condition:** installation, migration, operation, recovery, and model selection are documented and the PR checklist reflects implemented reality.

## Pull-request management

This draft PR is the tracking PR for the modernization branch. The checklist will be updated as stages are implemented. Commits should remain stage-focused and independently testable.

If implementation size makes review unsafe, the work may be split into dependent PRs only after preserving this document as the umbrella plan and receiving user approval. The default is to continue on this branch/PR.

## Overall definition of done

- No known false-success checkpoint path remains.
- Pause/resume works under normal interruption and injected failures.
- Outputs are atomically committed and configuration-aware.
- Unmasked pixels are preserved before encoding.
- Model backends are replaceable without orchestration changes.
- The current model path remains available as a measured baseline.
- Model/default changes are supported by reproducible watermark-specific evidence.
- Personal-photo processing has an inventory, pilot, review, metadata, and archival strategy.
- Code, tests, docs, ADRs, and PR checklist agree.
