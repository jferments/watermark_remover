# Watermark Remover documentation

This directory is the design and maintenance source of truth for the modernization of `watermark_remover`.

## Documents

- [`AGENTS.md`](AGENTS.md) — mandatory instructions for coding agents and contributors.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — current architecture, known weaknesses, and target modular pipeline.
- [`PROCESSING_SEMANTICS.md`](PROCESSING_SEMANTICS.md) — job states, durable output commits, checkpointing, retries, and shutdown behavior.
- [`MODEL_BACKENDS.md`](MODEL_BACKENDS.md) — detector, mask, and inpainting backend survey and selection policy.
- [`BENCHMARKING.md`](BENCHMARKING.md) — correctness, quality, reliability, and throughput evaluation.
- [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) — staged implementation checklist for the modernization PR.
- [`adr/`](adr/) — architecture decision records that explain the durable decisions behind the design.

## Authority and change policy

The executable code and tests determine actual behavior. These documents describe intended behavior and must be updated in the same pull request whenever a change modifies:

- a processing invariant;
- a model or runtime dependency;
- output/checkpoint semantics;
- metadata handling;
- performance architecture;
- CLI compatibility;
- benchmark or acceptance criteria.

Model quality and performance claims must be backed by a reproducible benchmark. A newer model name or publication date is not, by itself, evidence that it is better for watermark removal.
