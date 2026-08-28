# ADR 0003: Use ROI-first, source-preserving restoration

- **Status:** Accepted
- **Date:** 2026-08-27

## Context

The current LaMa path processes the full decoded image and replaces the full result even when a small corner watermark is the only modified region. This increases preprocessing, host/device transfer, GPU inference, postprocessing, multiprocessing transfer, and encoding pressure. It also gives a generative model the opportunity to alter pixels that did not require reconstruction.

For a 6000×4000 image, a 1200×600 context crop contains approximately one thirty-third as many pixels as the full frame. Actual speedup is model- and overhead-dependent, but the asymmetry is too large to ignore.

## Decision

The pipeline plans context-padded regions of interest around connected mask components and invokes the restoration backend only on those regions when the backend and mask geometry permit it.

ROI planning must:

- merge overlapping/nearby components;
- add configurable context;
- clamp to image bounds;
- align dimensions to backend requirements;
- preserve exact coordinate transforms;
- fall back to full-frame processing when the mask/ROI is too large or global context is required.

Restored output is composited into the original decoded source using only the mask plus an explicit, bounded feather band. Pixels outside that support remain sourced from the original image.

Known-alpha inverse reconstruction occurs before generative inpainting where available, reducing the residual mask further.

## Consequences

### Positive

- Potentially large reductions in compute and memory for small watermarks.
- Unmasked photograph content is protected from model drift.
- Different backends can operate at their preferred fixed or bounded resolution.
- Multiple separated watermark regions can be processed independently or bucketed.
- The same design supports Diffusers-style mask cropping and MI-GAN-style crop pipelines.

### Negative

- Insufficient context can create implausible fills or seams.
- Multiple crops require careful blending and overlap handling.
- Global structures may require full-frame fallback.
- ROI resizing can lose detail unless transforms and output scale are chosen carefully.
- Quality must be benchmarked by mask size/background class rather than assumed.

## Alternatives considered

### Always use full-frame inference

Rejected as the default because it wastes work for small corner watermarks and weakens source-pixel preservation. It remains a fallback.

### Resize every full image to a fixed model resolution

Rejected because it discards useful local detail and still exposes unrelated pixels to the model.

### Accept the entire model-returned frame

Rejected because generative backends may alter unmasked content. The original source is authoritative outside the controlled composite support.
