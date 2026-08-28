# ADR 0002: Define success at durable output commit

- **Status:** Accepted
- **Date:** 2026-08-27

## Context

The current implementation increments progress after a GPU worker places an array in a writer queue. The coordinator may then terminate writers before all files and checkpoint messages are complete. Writer return values are not checked, writer exceptions are swallowed, and failed images may never increment the expected total.

This creates three unacceptable outcomes:

- an image can be reported as processed before it exists safely on disk;
- a write failure can enter the checkpoint as success;
- a decode/model failure can leave the coordinator waiting forever.

## Decision

The only successful completion boundary is an acknowledged atomic output commit followed by a ledger transaction.

The commit path is:

1. encode to a destination-local temporary file;
2. verify explicit encoder success;
3. validate the temporary output;
4. apply required metadata policy;
5. atomically replace the final path;
6. acknowledge the final output identity;
7. transactionally record success in the authoritative ledger.

Every attempted job must produce exactly one terminal result. Worker liveness and in-flight ownership are monitored so missing results become typed worker-loss failures rather than an infinite wait.

Normal completion and the first Ctrl+C drain producers, commit workers, acknowledgements, and ledger updates. Forceful termination is reserved for a second interrupt or unresponsive worker.

## Consequences

### Positive

- Progress and resume data reflect real outputs.
- Interrupted runs can be reconciled deterministically.
- Writer and filesystem failures become visible.
- Failure records can be de-duplicated transactionally.
- Coordinator liveness no longer depends only on a target count.

### Negative

- Reported progress may lag GPU inference because it waits for encoding/commit.
- The coordinator must track in-flight ownership and queue shutdown ordering.
- A crash between atomic rename and ledger update requires startup reconciliation.
- Optional `fsync` durability may reduce throughput and should be a documented mode.

## Alternatives considered

### Continue logging when work enters the writer queue

Rejected because queue insertion is not a durable side effect.

### Log after `cv2.imwrite()` without a temporary file

Rejected because a crash or failed overwrite can leave a partial final file and destroy the previously valid output.

### Let each worker append directly to a text log

Rejected because concurrent appends do not provide sufficient transactional state, typed attempts, configuration provenance, or reliable de-duplication.
