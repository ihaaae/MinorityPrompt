# AGENTS.md — MinorityPrompt

This repository contains the CVPR 2025 MinorityPrompt implementation for
minority-focused text-to-image generation via prompt optimization. In the parent
UnsafeDistribution repo, this checkout is used as a vendored/submodule source for
generation code; keep changes focused and avoid broad refactors.

## Repository map

- `latent_diffusion.py`: Stable Diffusion 1.5/2.0 solver and prompt-optimization logic.
- `latent_sdxl.py`: SDXL and SDXL-Lightning solver and prompt-optimization logic.
- `examples/`: CLI entry points for single-prompt and MS-COCO generation.
- `examples/assets/`: Prompt assets used by the examples.
- `scripts/`: Example shell wrappers for running the CLIs.
- `utils/`: Image, logging, callback, metric, and classifier helpers.
- `environment.yaml`: Original conda environment specification.

## Running

- Upstream setup assumes a conda environment named `mprompt` created from
  `environment.yaml`.
- SDXL-Lightning runs expect checkpoints such as
  `ckpt/sdxl_lightning_4step_unet.safetensors` under `ckpt/`.
- Example entry points are run as modules from the repo root, for example:
  `python -m examples.text_to_img ...` and `python -m examples.text_to_mscoco ...`.
- Generation is GPU-heavy. Do not run generation jobs, long MS-COCO jobs, or scripts
  that download model weights unless the user explicitly asks.

## Coding conventions

- Prefer the existing simple Python style: module-level functions, `argparse` CLIs,
  `Path` for paths in entry points, and dictionaries for prompt-optimization kwargs.
- Keep CLI flags compatible with the existing scripts and examples; avoid renaming
  arguments or changing defaults unless that is the requested behavior.
- Preserve output path conventions under `examples/workdir/...` because downstream
  scripts may rely on them.
- Avoid introducing new dependencies unless they are already present in
  `environment.yaml` or the user explicitly requests them.
- Do not edit generated outputs, checkpoints, caches, or model artifacts.

## Verification

- For syntax-only changes, prefer `python -m py_compile <touched files>`.
- For CLI argument changes, use `python -m examples.<name> --help` if verification is
  needed; this should not require model downloads or GPU execution.
- Avoid running the shell scripts in `scripts/` as verification unless the user asks,
  because they start image generation.
