# ADR 0001: Separate locator, mask, and restoration backends

- **Status:** Accepted
- **Date:** 2026-08-27

## Context

The current GPU worker directly constructs Ultralytics YOLO and `SimpleLama` objects and embeds detection, mask construction, and restoration in one loop. This makes a model change an orchestration change and prevents clean comparison of detector, mask, and inpainting alternatives.

A watermark-removal pipeline has at least three distinct model/algorithm boundaries:

1. locate a watermark or expected region;
2. produce a pixel mask;
3. reconstruct the selected pixels.

For repeated known watermarks, one or more boundaries may be deterministic rather than learned.

## Decision

Define framework-independent protocols for locator, mask, and inpainting backends. Orchestration consumes only project domain types. Backend factories are selected through a built-in registry and validated against declared capabilities.

Model artifacts are described by manifests containing hashes, runtime/task information, and license/provenance fields. Optional heavyweight backends use optional dependency groups.

The current YOLO11 detector, box-mask behavior, and LaMa model are implemented first as compatibility adapters and remain the regression baseline.

## Consequences

### Positive

- Models can be benchmarked and swapped independently.
- Known-template, classical, segmentation, and diffusion paths fit the same pipeline.
- Framework-specific dependencies do not leak into coordinator tests.
- Fake deterministic backends can test failure and shutdown semantics without large downloads or GPUs.
- Model provenance and license policy become enforceable.

### Negative

- More types and modules than the current single script.
- Adapter contracts must be designed carefully around batching, resolution, prompts, and device ownership.
- Some model-specific features may require capability extensions.

## Alternatives considered

### Keep direct imports and add CLI conditionals

Rejected because each model would add branching throughout workers and make combinations difficult to test.

### Adopt IOPaint as the application framework

Rejected as the core dependency because IOPaint was archived in 2025, carries a broad UI/application scope, and would not solve this project's durable checkpoint and high-throughput worker requirements. Its model-adapter ideas remain useful reference material.

### Use arbitrary Diffusers model identifiers as the plugin system

Rejected because locator/non-diffusion backends do not fit, and arbitrary model IDs bypass tested profiles, memory limits, artifact hashes, and license review.
