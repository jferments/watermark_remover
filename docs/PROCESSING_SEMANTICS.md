# Processing semantics

## Purpose

This document defines when work begins, when it ends, what may be checkpointed, how failures are represented, and how pause/resume behaves. These rules are correctness requirements, not implementation suggestions.

## Stable job identity

A job is identified by a normalized input-relative path plus an input identity record. At minimum, the identity record contains file size and nanosecond modification time. A content hash may be added for stronger guarantees or generated lazily when required.

The output root is not part of the relative-path identity, but the ledger belongs to one output root and one dataset/configuration context.

## Attempt state machine

```text
DISCOVERED
    |
    v
QUEUED
    |
    v
RUNNING
    |-- DECODE_FAILED ---------------------> FAILED
    |-- LOCATE_FAILED ---------------------> FAILED
    |-- NO_WATERMARK + expect-watermark --> REVIEW_REQUIRED
    |-- MASK_FAILED -----------------------> FAILED
    |-- RESTORE_FAILED --------------------> FAILED
    |-- ENCODE_FAILED ---------------------> FAILED
    |-- METADATA_FAILED -------------------> FAILED
    |-- COMMIT_FAILED ---------------------> FAILED
    |
    v
OUTPUT_COMMITTED
    |
    v
LEDGER_COMMITTED --------------------------> SUCCEEDED
```

`CANCELLED` is terminal for an attempt that was intentionally stopped before durable success. A later resume may create a new attempt for the same job.

Every started attempt must eventually receive exactly one terminal status. The coordinator must synthesize a typed worker-loss failure for in-flight jobs if a worker exits without reporting them.

## Durable success boundary

The following events are not success:

- detector inference completed;
- inpainting inference completed;
- a NumPy array was produced;
- a result was placed in a queue;
- a temporary file was opened;
- an encoder call returned without raising;
- a progress counter was incremented.

A job is successful only after:

1. the encoder explicitly reports success;
2. the temporary file exists, is non-empty, and passes configured decode/format validation;
3. required metadata operations succeed or a documented best-effort policy records a warning;
4. the temporary file is atomically renamed/replaced to the final path;
5. the commit worker sends an acknowledgement containing the final path and output identity;
6. the ledger transaction stores the success and configuration fingerprint.

Progress shown as “completed” must use this durable count. The UI may separately show “inference finished” or “awaiting write” counts.

## Atomic output protocol

Temporary files must be created in the destination directory or same filesystem so `os.replace()` is atomic.

A typical sequence is:

```text
.final-name.jpg.wmr-tmp-<job-id>-<attempt-id>
    -> encode
    -> fsync file when durable mode requires it
    -> validate
    -> transfer metadata
    -> os.replace(temp, final)
    -> optionally fsync parent directory
    -> acknowledge
```

On startup, stale temporary files are reconciled. A temporary file is never considered a completed output merely because it exists.

The default overwrite policy must be explicit. Existing final outputs should not be destroyed until the replacement temporary file has passed validation.

## Ledger

SQLite is the planned authoritative checkpoint because it supports transactions and uniqueness constraints. A representative schema is:

```sql
CREATE TABLE jobs (
    relative_path TEXT PRIMARY KEY,
    input_size INTEGER NOT NULL,
    input_mtime_ns INTEGER NOT NULL,
    input_hash TEXT,
    current_status TEXT NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    last_error_stage TEXT,
    last_error_type TEXT,
    last_error_message TEXT,
    output_relative_path TEXT,
    output_size INTEGER,
    output_hash TEXT,
    config_fingerprint TEXT,
    model_provenance_json TEXT,
    timing_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    committed_at TEXT
);

CREATE TABLE attempts (
    attempt_id TEXT PRIMARY KEY,
    relative_path TEXT NOT NULL,
    status TEXT NOT NULL,
    worker_id TEXT,
    error_stage TEXT,
    error_type TEXT,
    error_message TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    FOREIGN KEY(relative_path) REFERENCES jobs(relative_path)
);
```

The exact schema may evolve, but these semantics remain:

- one current job row per relative path;
- multiple attempt records when retries occur;
- failure de-duplication comes from the primary key rather than text-file scanning;
- attempt count and latest error remain visible;
- success records include output identity and configuration fingerprint;
- a transaction is used for terminal updates.

Human-readable `.processing_log.txt` and `processing_failures.txt` files may be exported for compatibility, but they are not the source of truth.

## Resume rules

On resume, a prior success is skipped only when all required checks pass:

- the final output exists;
- it passes configured lightweight validation;
- the input identity still matches or the selected policy explicitly permits changes;
- the configuration fingerprint matches;
- the ledger status is successful.

If the existing legacy text log is imported, each referenced output must be checked. Imported rows are marked `legacy_unverified` because the old format lacks model/configuration provenance.

Failures are retried according to policy. A default bounded retry count should apply only to failures likely to be transient. Invalid/corrupt input and deterministic model-contract failures should not loop indefinitely.

## No-watermark outcomes

A locator returning no watermark has three possible policy outcomes:

- `COPY_UNCHANGED`: copy source bytes and metadata when a mixed dataset is expected.
- `REVIEW_REQUIRED`: use when `--expect-watermark` is enabled or confidence is otherwise suspicious.
- `SKIP_WITHOUT_OUTPUT`: available only through explicit policy; it must not be represented as removal success.

Direct copy should use `shutil.copy2()` or a metadata-aware equivalent and still pass through the atomic commit protocol.

## Failure taxonomy

Failures must include stage, type, message, source path, worker identity, and attempt identity. Recommended stages are:

```text
discovery
validation
decode
locate
mask
roi
restore
composite
encode
metadata
commit
ledger
worker
shutdown
```

Error messages shown to the user may be concise, but the structured record should retain enough context to diagnose the failure. Secrets, private metadata, and raw image contents must not be logged.

## Coordinator liveness

The coordinator may not wait only on a numeric completed count. It must also monitor:

- worker process liveness;
- in-flight job ownership;
- result queue activity;
- queue closure/sentinels;
- shutdown state.

When all workers have exited, the coordinator must resolve every remaining in-flight job as success already acknowledged, typed failure, or cancellation. It must never sleep forever waiting for a message that can no longer arrive.

## Graceful pause and shutdown

### First Ctrl+C

The first interrupt requests a graceful pause:

1. set a shared stop-submission event;
2. stop discovering/submitting new work;
3. allow workers to finish their current image or bounded microbatch;
4. join GPU workers;
5. signal local or external commit workers after producers stop;
6. drain all output work and commit acknowledgements;
7. flush ledger transactions and reports;
8. exit with a paused summary.

Jobs that were discovered but never started remain pending. Jobs started but intentionally abandoned are marked cancelled and are eligible on resume.

### Second Ctrl+C

A second interrupt requests forceful shutdown. The program should make a best effort to record in-flight ownership and remove known temporary files before terminating child processes. It must clearly report that forced shutdown may require reconciliation at next startup.

### Normal completion

Normal completion follows the same producer-stop, queue-drain, acknowledgement, and ledger-flush sequence. It must not use unconditional `terminate()` on healthy workers.

## Queue semantics and backpressure

Every queue is bounded. Capacity is configured using both item count and approximate bytes where image arrays are involved.

Workers may block on backpressure; they may not discard outputs or progress events silently. Low-value telemetry may be coalesced, but terminal results and commit acknowledgements are lossless.

Sentinels are sent only after all producers for a queue have stopped. One sentinel is required per independent consumer unless queue implementation semantics document otherwise.

## Output fidelity and metadata

Output policy must specify:

- codec and quality/compression;
- bit depth and color mode;
- orientation handling;
- ICC profile behavior;
- EXIF/IPTC/XMP/GPS behavior;
- timestamps and filesystem metadata;
- alpha-channel support;
- lossless cleaned-master mode.

Unmasked pixels should be sourced from the original decoded image rather than blindly accepting a generative model's full-frame output. For lossy output, outside-mask pixel identity is evaluated before encoding.

## Configuration mismatch

When a ledger contains successes created by another configuration fingerprint, the application must choose one explicit behavior:

- reject and explain the mismatch;
- create/use a separate run namespace;
- reprocess the mismatched outputs;
- allow a user-confirmed override recorded in provenance.

Silently skipping mismatched outputs is forbidden.

## Required failure-injection tests

The implementation must prove these semantics with controlled failures at every stage, abrupt worker exit, queue backpressure, duplicate retries, and interrupt/resume scenarios. See [`BENCHMARKING.md`](BENCHMARKING.md).
