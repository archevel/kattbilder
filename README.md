# Kattbilder

Cat images generated entirely in the browser with a diffusion model running on WebGPU
through onnxruntime-web. No backend: the page is static files plus model weights.

## Run

    node serve.mjs        # then open http://localhost:8080

Any static server works, but it must stream large files and serve over http(s), not
`file://` (WebGPU and the Cache API require it).

## Pipelines

The page picks a pipeline from the WebGPU adapter it gets:

| pipeline | model | needs | download | output | speed seen |
|---|---|---|---|---|---|
| `sdturbo` (fp16) | SD-Turbo, fixed prompt, 1 step | WebGPU + `shader-f16` | 1.75 GB | 512×512, photorealistic | ~6 s/cat on Intel Arc iGPU (Meteor Lake) |
| `ddpm` | google/ddpm-ema-cat-256, unconditional, 40 DDIM steps | WebGPU only | 455 MB | 256×256, real-photo look, more artifacts | ~8.4 s/cat on RTX 4070 laptop |

Override with URL params: `?model=sdturbo|ddpm`, `?steps=25` (ddpm), `?eta=1` (ddpm, stochastic DDIM),
`?batch=10` (cats kept ready ahead of the viewer).

The DDPM model is trained on cat photos only, so "only cats" is guaranteed by the model.
The SD-Turbo pipeline enforces cats via a hard-coded prompt whose CLIP embedding is
precomputed (`public/prompt_embeds.bin`), so the browser never loads the text encoder.

## Why two pipelines: shader-f16

Every public ONNX export of SD-Turbo is fp16, and onnxruntime-web refuses fp16 graphs
unless the adapter exposes `shader-f16`. Chrome on Linux does not expose it for NVIDIA
(the driver supports it; Chrome just doesn't enable it there yet), but does for Intel Arc.
An fp32 SD-Turbo conversion works in principle (`?model=sdturbo&precision=fp32`,
files from `tools/fp16_to_fp32.py` + `tools/split_external_data.py`), but it peaks above
7.5 GB of GPU memory and fails on an 8 GB card. Hence the small fp32 DDPM model.

## Chrome on Linux

Chrome may silently use SwiftShader (CPU) for WebGPU. Check the page's Log section: it prints
the adapter. To get hardware Vulkan, launch Chrome with:

    google-chrome-stable --enable-features=Vulkan --use-angle=vulkan --ignore-gpu-blocklist --enable-unsafe-webgpu

On a hybrid laptop Chrome picks the discrete GPU. To force the Intel iGPU (which has
shader-f16 and therefore runs SD-Turbo), point the Vulkan loader at its ICD:

    VK_ICD_FILENAMES=/run/opengl-driver/share/vulkan/icd.d/intel_icd.x86_64.json google-chrome-stable --enable-features=Vulkan --use-angle=vulkan --ignore-gpu-blocklist

(`--render-node-override` does not affect WebGPU's adapter choice.)

## Deploy

Pushing to `main` deploys `public/` to GitHub Pages via `.github/workflows/pages.yml`
(https://archevel.github.io/kattbilder/). Model weights are not in the repo.

## Models

`public/models/` is git-ignored. Regenerate:

    # DDPM cat UNet -> public/models/ddpm-cat/unet.onnx
    nix-shell -p python3Packages.torch python3Packages.diffusers python3Packages.onnx python3Packages.onnxruntime \
      --run "python3 tools/export_ddpm_cat.py public/models/ddpm-cat"

    # prompt embedding for SD-Turbo (only if you change the prompt)
    cd tools && npm install && node embed-prompt.mjs "a photo of a black cat on a sofa"

The DDPM model is served from https://huggingface.co/archevel/kattomat (`ddpm-cat/unet.onnx`),
so `public/` can be deployed as plain static files (GitHub Pages works). To re-upload after
re-exporting:

    nix-shell -p python3Packages.huggingface-hub --run "hf auth login"
    nix-shell -p python3Packages.huggingface-hub --run \
      "hf upload archevel/kattomat public/models/ddpm-cat ddpm-cat --repo-type model"
