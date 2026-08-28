# Agent instructions

These instructions apply to all automated agents and contributors working in this repository.

## Project purpose

`watermark_remover` is a local, high-throughput image restoration pipeline for images the operator has the right to modify. It detects visible watermarks, constructs removal masks, reconstructs masked pixels, preserves the source directory structure, and supports interruption/resume for very large collections.

The project prioritizes, in order:

1. **Data safety and truthful completion semantics.** Never claim an image is complete before a valid output has been durably committed.
2. **Faithful restoration.** Preserve all unmasked source pixels exactly whenever the selected backend permits it.
3. **Recoverability.** Ctrl+C, process failure, or restart must not corrupt outputs or produce false checkpoint records.
4. **Measurable throughput.** Optimize only after collecting stage-level timings and quality metrics.
5. **Replaceable models.** Detector, mask-generation, and inpainting implementations must not be hard-coded into orchestration logic.

## Required GitHub workflow

- Never commit directly to `main`.
- Use one development branch and one pull request for the assigned task unless the user explicitly approves a split.
- Keep the pull-request checklist current as work is completed.
- Use the GitHub connector for GitHub reads and writes. Do not substitute the `gh` CLI.
- Do not merge the pull request without explicit user direction.
- Before editing an existing file, read the current branch version and preserve unrelated code and comments.

## Required reading

Before changing code, read:

- [`ARCHITECTURE.md`](ARCHITECTURE.md)
- [`PROCESSING_SEMANTICS.md`](PROCESSING_SEMANTICS.md)
- [`MODEL_BACKENDS.md`](MODEL_BACKENDS.md)
- [`BENCHMARKING.md`](BENCHMARKING.md)
- [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)
- applicable records under [`adr/`](adr/)

## Non-negotiable processing invariants

Every discovered input image must produce exactly one terminal result for an attempt. Terminal results include committed success, a typed failure, or cancellation before processing began.

An image is **not** successfully processed when inference finishes, when an output is placed in a queue, or when encoding starts. Success exists only after all of the following occur:

1. the output is encoded successfully;
2. the temporary output is validated;
3. required metadata is transferred according to policy;
4. the temporary file is atomically moved to its final path;
5. the commit is acknowledged to the ledger owner;
6. the ledger transaction records the committed result.

Additional invariants:

- Never add a success checkpoint for a failed, missing, partial, or merely queued output.
- Never silently swallow decode, inference, encoding, metadata, filesystem, or worker errors.
- Failure records must be de-duplicated by stable job identity while retaining attempt count and the latest error.
- A dead worker must not leave the coordinator waiting forever.
- The first Ctrl+C requests graceful draining; forceful termination is a last resort.
- Output paths must remain within the configured output root.
- Input and output roots must be validated to prevent recursive self-processing.
- Reusing an output directory with a materially different model/configuration fingerprint must be rejected or explicitly handled.
- No-watermark detection is distinct from successful watermark removal. In `expect-watermark` mode it must enter review/failure handling rather than being silently accepted.

## Model and dependency policy

- Treat model weights as versioned artifacts with a cryptographic hash, task, runtime, source, and license metadata.
- Verify both code license and weight/license terms. Do not assume a repository license automatically covers separately hosted weights.
- Do not make a new backend the default based only on benchmark claims from its authors.
- Keep the currently working YOLO11 detector and LaMa path available as a regression baseline until a replacement passes the project benchmark.
- Diffusion-based inpainting must remain opt-in unless it demonstrates acceptable fidelity, deterministic controls, throughput, and licensing for the intended use.
- A backend must implement the documented protocol rather than leaking framework-specific objects into orchestration code.
- Optional heavy backends belong in optional dependency groups; the default installation must not require every model stack.

## Performance work

- Add instrumentation before optimization.
- Report throughput in both images/second and megapixels/second.
- Measure decode, detection, mask construction, inpaint preprocessing, inference, postprocessing, queue wait, encode, metadata, and commit time separately.
- Use CUDA events or explicit synchronization for GPU timing; wall-clock timing around asynchronous launches is insufficient.
- Record p50, p90, p95, and p99 latency, queue occupancy, peak RAM, and peak VRAM.
- Never weaken output correctness, metadata preservation, or shutdown semantics for a speculative speed improvement.
- Retain a benchmark result before and after each material performance change.

## Image fidelity and privacy

- Do not commit private photographs, private metadata, model weights, or generated debug artifacts.
- Tests must use synthetic fixtures, redistributable public fixtures, or programmatically generated images.
- Personal-photo evaluation sets remain outside the repository and should be referenced through local manifests only.
- Outside-mask pixels must remain byte-identical after compositing where the codec/output format allows exact comparison.
- Lossy re-encoding must be explicit. Avoid a second JPEG generation when later adding a replacement watermark.

## Testing requirements

Changes to orchestration or checkpointing require failure-injection tests covering at least:

- unreadable input;
- detector exception;
- inpainting exception;
- encoder returning failure;
- metadata-copy failure;
- atomic-rename failure;
- worker exit during a job;
- first and second Ctrl+C behavior;
- resume after interruption;
- duplicate failure reporting;
- changed configuration fingerprint;
- output queue backpressure.

Model adapters require contract tests with fake or tiny deterministic models. Tests that require large downloaded weights or a GPU must be separately marked and must not be the only validation of core logic.

## Documentation requirements

Update the relevant design document and architecture decision record whenever implementation changes a durable decision. Include the reason, alternatives considered, consequences, migration behavior, and test evidence.

## Definition of done

A task is complete only when:

- implementation and tests are committed to the task branch;
- relevant local checks pass;
- the PR checklist and documentation match actual behavior;
- no known correctness regression is hidden behind a performance claim;
- failures and unresolved risks are stated explicitly in the pull request.
