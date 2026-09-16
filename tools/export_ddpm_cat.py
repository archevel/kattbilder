"""Export google/ddpm-ema-cat-256 (unconditional DDPM UNet, cats only) to fp32 ONNX.

usage: nix-shell -p python3Packages.torch python3Packages.diffusers python3Packages.onnx \
         --run "python3 tools/export_ddpm_cat.py public/models/ddpm-cat"
Output: <out_dir>/unet.onnx (+ config.json with the scheduler betas info)
Inputs:  sample  float32 [1,3,256,256],  timestep int64 [1]
Output:  eps     float32 [1,3,256,256]   (predicted noise)
"""
import json, os, sys
import torch
from diffusers import UNet2DModel

repo = os.environ.get("REPO", "google/ddpm-ema-cat-256")
out_dir = sys.argv[1] if len(sys.argv) > 1 else "public/models/ddpm-cat"
os.makedirs(out_dir, exist_ok=True)

unet = UNet2DModel.from_pretrained(repo).eval()
print("params:", sum(p.numel() for p in unet.parameters()) / 1e6, "M")


class Wrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, sample, timestep):
        return self.m(sample, timestep).sample


sample = torch.randn(1, 3, 256, 256)
timestep = torch.tensor([999], dtype=torch.int64)
with torch.no_grad():
    ref = Wrapper(unet)(sample, timestep)
    torch.onnx.export(
        Wrapper(unet), (sample, timestep), os.path.join(out_dir, "unet.onnx"),
        input_names=["sample", "timestep"], output_names=["eps"],
        opset_version=17, dynamo=False, do_constant_folding=True,
    )

# verify with onnxruntime if available
try:
    import onnxruntime as ort, numpy as np
    s = ort.InferenceSession(os.path.join(out_dir, "unet.onnx"), providers=["CPUExecutionProvider"])
    out = s.run(None, {"sample": sample.numpy(), "timestep": timestep.numpy()})[0]
    print("max abs diff vs torch:", float(np.abs(out - ref.numpy()).max()))
except ImportError:
    print("onnxruntime not available; skipped numeric check")

json.dump({
    "repo": repo, "num_train_timesteps": 1000, "beta_schedule": "linear",
    "beta_start": 1e-4, "beta_end": 0.02, "clip_sample": True, "sample_size": 256,
}, open(os.path.join(out_dir, "config.json"), "w"), indent=2)
print("saved to", out_dir, os.path.getsize(os.path.join(out_dir, "unet.onnx")) / 1e6, "MB")
