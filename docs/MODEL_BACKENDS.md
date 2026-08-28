# Model backends

_Last reviewed: 2026-08-27._

## Purpose

This document separates three different problems that are currently coupled together:

1. **Locate** a watermark or expected watermark region.
2. **Build/refine a mask** that identifies pixels requiring reconstruction.
3. **Restore** the masked pixels while preserving the rest of the image.

A newer model in one stage does not automatically improve the complete pipeline. For the current workload, mask precision and the number of pixels sent to the restoration model may matter more than changing the detector generation.

## Selection criteria

Every candidate backend is evaluated on:

- watermark detection recall and false-positive rate;
- mask precision, recall, and boundary quality;
- reconstruction quality inside the mask and transition band;
- preservation of unmasked pixels;
- deterministic/reproducible behavior;
- throughput and latency across resolution buckets;
- peak VRAM and RAM;
- model initialization cost;
- current runtime compatibility;
- maintenance burden;
- code license and weight/license clarity;
- ability to run locally without sending photographs to a service.

The project keeps a working baseline until another backend wins on the repository benchmark. Publication metrics on Places2, COCO, or a generic object-removal dataset do not substitute for watermark-specific evaluation.

## Current baseline

The current script uses:

- a custom YOLO11x watermark detector checkpoint;
- filled bounding-box masks plus dilation;
- the `simple-lama-inpainting` wrapper around a TorchScript Big-LaMa model.

The wrapper is small and convenient, but it hides preprocessing and forces a full-image/PIL-oriented call path. The modernization should retain LaMa as a baseline while moving it behind an internal adapter so ROI handling, tensor conversion, compositing, artifact hashing, and error reporting are controlled by this project.

Sources:

- [Ultralytics YOLO documentation](https://docs.ultralytics.com/)
- [LaMa official repository](https://github.com/advimman/lama)
- [`simple-lama-inpainting`](https://github.com/enesmsahin/simple-lama-inpainting)

## Locator and mask candidates

### Known-template / alpha-mask locator — highest priority for personal photos

When the old business watermark asset, anchor, opacity, scale rules, or export preset can be recovered, a deterministic template backend is preferable to a generic detector.

Potential methods include:

- normalized anchor/scale rules;
- multi-scale alpha-template matching;
- edge or phase-correlation registration;
- feature matching for rotated/scaled variants;
- visual-exemplar matching;
- a fixed set of layout profiles for portrait and landscape exports.

The backend returns an alignment transform, confidence, exact alpha mask, and estimated blend parameters. This can improve speed and quality simultaneously and may enable inverse alpha compositing before inpainting.

### YOLO11 baseline

The existing custom checkpoint remains the regression baseline. Its value comes from task-specific training, not the version number. The modernization must first wrap it behind `LocatorBackend` and reproduce current detections before considering retraining.

### YOLO26 detection and segmentation — recommended experiment

Ultralytics identifies YOLO26 as its January 2026 model family. It supports detection and instance segmentation, includes an end-to-end head that avoids NMS, and provides P2/P6 architecture configurations for unusually small objects or large inputs. P2/P6 configurations require task-specific training; released generic weights are not watermark models.

A YOLO26 checkpoint cannot replace the current custom YOLO11 checkpoint without retraining. The most valuable experiment is likely **YOLO26 segmentation**, because a precise watermark mask could preserve pixels between letters and transparent logo regions. A YOLO26 detector using the same box labels may improve speed or recall, but it would retain the rectangular-mask limitation.

Recommended experiment matrix:

- YOLO11x detect baseline;
- smaller YOLO11 detect scales to test whether `x` is unnecessary;
- YOLO26 detect at multiple scales;
- YOLO26-P2 detect for small corner marks;
- YOLO26 segmentation using generated/curated alpha masks;
- end-to-end versus one-to-many head where supported;
- PyTorch versus exported ONNX/TensorRT only after correctness parity.

Source: [Ultralytics YOLO26 documentation](https://docs.ultralytics.com/models/yolo26/)

### YOLOE-26 / visual-prompt localization — optional research

YOLOE-26 supports text, visual, and prompt-free open-vocabulary detection/segmentation. A visual prompt from the old logo may be useful when there are multiple logo variants and insufficient labels for immediate retraining.

It should be treated as a research backend rather than a default. Open-vocabulary capability does not guarantee sensitivity to faint, transparent watermarks, and a specialized supervised model is likely to remain faster and easier to calibrate.

### SAM 3 / SAM 3.1 — optional mask refiner

Meta's SAM 3 can segment from text, geometry, and image exemplars. It may be useful as:

- a box-to-mask refiner after a fast detector;
- an annotation tool for creating YOLO segmentation labels;
- a visual-exemplar locator for repeated logos;
- a review-time interactive correction tool.

It is an 848M-parameter foundation model with a substantially heavier runtime than the current detector. It should not sit in the default path unless it materially improves mask quality enough to justify its cost. Its separate model/license terms must be reviewed before redistribution.

Source: [Meta SAM 3 repository](https://github.com/facebookresearch/sam3)

## Restoration candidates

### Inverse alpha compositing — preferred when the watermark is known

For a normal alpha blend,

```text
observed = alpha * watermark + (1 - alpha) * background
```

so the background can be estimated by:

```text
background = (observed - alpha * watermark) / (1 - alpha)
```

This is more faithful than generative inpainting when alignment, watermark color, alpha, and blend behavior are known and `alpha` is not close to one. It becomes numerically unstable for opaque pixels, unknown blend modes, JPEG artifacts, or inaccurate alignment.

The preferred personal-photo backend is therefore hybrid:

1. estimate alignment and effective alpha;
2. reverse pixels where the solution is stable;
3. build a residual/opaque mask;
4. use learned inpainting only on the unresolved pixels.

### OpenCV Telea and Navier–Stokes — fast classical baseline

Classical inpainting is extremely fast and can outperform learned models for tiny marks on smooth or locally predictable backgrounds. It is likely to fail on faces, textural detail, long edges, and large logos.

It should be implemented as a CPU backend and included in the benchmark. A routing policy may select it only for small masks that satisfy simple geometric/background criteria.

### LaMa — recommended default baseline after modernization

LaMa remains a strong default for high-throughput object removal because it is a single-pass learned model, handles irregular masks, and was designed to generalize to resolutions above its training resolution. It is much less computationally expensive than iterative diffusion.

The proposed change is not to abandon LaMa immediately. It is to replace the thin third-party wrapper with a controlled adapter that supports:

- direct NumPy/tensor input;
- ROI crops;
- batched or bucketed experiments where feasible;
- model artifact hashing;
- explicit device and precision handling;
- mask-only compositing;
- typed errors and timings;
- tests independent of the wrapper package.

Source: [LaMa official repository](https://github.com/advimman/lama)

### MI-GAN — promising fast candidate, pending weight-license clarification

MI-GAN was designed as a much smaller and faster inpainting model and provides an ONNX-oriented pipeline that crops around masks, resizes to 512×512, inpaints, resizes back, and blends into the source. This makes it architecturally attractive for high-throughput ROI processing.

However:

- the official implementation notes fixed-resolution operations in the provided PyTorch path;
- high-resolution output depends on crop/resize behavior;
- its quality on thin semi-transparent watermarks must be measured;
- the repository code is MIT, but an August 2026 issue asks the authors to clarify whether the separately hosted weights inherit restrictions from teacher-model dependencies.

Until weight terms are clarified, MI-GAN should be an experimental optional backend rather than the default or a redistributed artifact.

Sources:

- [MI-GAN official repository](https://github.com/Picsart-AI-Research/MI-GAN)
- [Open weight-license clarification issue](https://github.com/Picsart-AI-Research/MI-GAN/issues/25)

### MAT and ZITS++ — structural candidates

MAT and ZITS/ZITS++ focus on large holes and structural consistency. They may outperform LaMa in scenes containing long lines, architecture, or repeated structure, but their older research environments and multi-component pipelines increase integration and maintenance cost.

They belong in a benchmark watchlist rather than the first implementation wave. A backend should be added only if a maintained inference artifact can be pinned and it wins on a meaningful subset of the watermark benchmark.

Sources:

- [MAT official repository](https://github.com/fenglinglwb/MAT)
- [ZITS++ repository](https://github.com/ewrfcas/ZITS-PlusPlus)

### PowerPaint v2.1 and BrushNet/BrushNetX — diffusion quality fallback

PowerPaint v2.1 and BrushNet-family models are stronger, prompt-aware diffusion inpainting systems intended for high-quality object removal and editing. They may produce more plausible content for difficult, large, semantically meaningful holes than a GAN-style model.

They are not suitable as the default high-throughput path without evidence because they:

- run multiple denoising steps;
- use substantially more VRAM and initialization time;
- can hallucinate semantically plausible but historically false detail;
- may alter lighting, texture, faces, or nearby content;
- require more complex dependency stacks and deterministic controls.

They should be optional `quality` backends used on a review queue or explicit fallback threshold. The source image must be composited outside the mask so diffusion cannot rewrite the full frame.

Sources:

- [PowerPaint official repository](https://github.com/open-mmlab/PowerPaint)
- [BrushNet official repository](https://github.com/TencentARC/BrushNet)

### FLUX.1 Fill [dev] — high-quality experimental fallback with licensing constraints

FLUX.1 Fill [dev] is an open-weight inpainting/outpainting model with a large model footprint and a non-commercial model license. Its reference stack is far heavier than required for normal watermark removal.

It must not become the default, be downloaded automatically, or be presented as unrestricted. It may be supported later behind an explicit optional backend that requires the operator to supply the model and acknowledge its license. It is mainly useful as a quality/research comparison for difficult images.

Source: [Black Forest Labs FLUX Fill documentation](https://github.com/black-forest-labs/flux/blob/main/docs/fill.md)

### Generic Diffusers backend

Hugging Face Diffusers provides `AutoPipelineForInpainting`, optimization hooks, and a `padding_mask_crop` mechanism that crops and enlarges a small masked region before compositing it back. This validates the general ROI-first design but does not remove the need for a project-specific backend contract and model/license manifest.

A generic Diffusers adapter may reduce the cost of adding compatible optional models, but each model still requires an explicit tested profile. Arbitrary model identifiers should not bypass artifact, license, memory, and output-fidelity checks.

Source: [Diffusers inpainting guide](https://huggingface.co/docs/diffusers/en/using-diffusers/inpaint)

### Dedicated visible-watermark restoration research

A 2025 paper, “Bridging Knowledge Gap Between Image Inpainting and Large-Area Visible Watermark Removal,” proposes adapting a pretrained inpainting backbone with residual background information and coarse masks. This direction is highly relevant because semi-transparent watermarks retain partial background evidence that generic inpainting discards.

The project should track this and similar watermark-specific models, but no backend should be promised until usable code/weights, license terms, reproducibility, and performance have been verified.

Source: [arXiv:2504.04687](https://arxiv.org/abs/2504.04687)

## Recommended backend strategy

### Generic mixed datasets

```text
YOLO11 baseline or validated YOLO26 locator
    -> segmentation mask when available
    -> ROI planner
    -> LaMa default
    -> optional quality fallback
    -> review for low-confidence or failed cases
```

### Repeated personal-business watermark

```text
known-template/alpha locator
    -> exact alpha mask and alignment
    -> inverse alpha reconstruction where stable
    -> residual mask
    -> ROI LaMa or validated fast alternative
    -> YOLO segmentation fallback for unknown layouts
    -> optional diffusion only for manually selected failures
```

This cascade is likely to improve fidelity and throughput more than using one newer generative model for every image.

## Backend configuration and registry

Backends are selected by stable IDs, not imported directly in orchestration code:

```text
locator: ultralytics:yolo11-watermark-v1
mask: segmentation-or-box-v1
inpainter: lama:big-lama-v1
fallback_inpainter: powerpaint:v2.1   # optional
```

A registry returns a factory and capability record. The runtime validates the pipeline combination before workers start. Examples:

- a box-only locator requires a mask backend that accepts boxes;
- a prompt-required diffusion backend requires an explicit prompt profile;
- a backend with a 512×512 contract requires ROI/resize planning;
- a non-commercial artifact cannot be silently selected for a commercial profile;
- a CPU-only backend is not assigned to a CUDA worker by accident.

## Model acceptance gate

A candidate may become a built-in optional backend after:

- code and weights are reproducibly obtainable;
- artifact hashes and licenses are documented;
- a contract adapter and unit tests exist;
- out-of-mask preservation is verified;
- the benchmark report records quality, speed, RAM, and VRAM;
- failure behavior is typed and recoverable.

A candidate may become the **default** only after it beats the current default on the intended workload without unacceptable regressions and the decision is recorded in an ADR.
