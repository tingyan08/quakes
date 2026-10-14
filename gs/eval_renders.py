"""Score a trained 3DGS model on the held-out views at a chosen resolution.

The trainer scores each model at its own training resolution, so a model
trained at 1/2 resolution and one trained at 1/4 are scored on different
images (finer images are harder to match). This renders the same held-out
views (every 8th image) at --eval_factor for any checkpoint, with the
trainer's metrics: torchmetrics PSNR, SSIM, and LPIPS (AlexNet, normalize=True).

Usage (gs environment's python):
  python gs/eval_renders.py --result_dir runs/whitney/gs/3dgs-f2 --eval_factor 4
Writes RESULT_DIR/eval_renders_f<factor>.json
"""
import argparse
import json
import os
import re
import sys

os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("CUDA_HOME", os.environ.get("RCAC_CUDA_ROOT", "/usr/local/cuda"))
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")

import numpy as np
import torch
from gsplat.rendering import rasterization
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # project root
sys.path.insert(0, f"{ROOT}/gs/third_party/gsplat/examples")
from datasets.colmap import Dataset, Parser  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--ckpt", default=None, help="default: the latest checkpoint in result_dir/ckpts")
    ap.add_argument("--data_dir", default=f"{ROOT}/runs/whitney/gs/dataset")
    ap.add_argument("--eval_factor", type=int, default=4)
    args = ap.parse_args()
    dev = "cuda"

    ckpt = args.ckpt or sorted(
        (os.path.join(args.result_dir, "ckpts", f) for f in os.listdir(os.path.join(args.result_dir, "ckpts"))),
        key=lambda p: int(re.search(r"ckpt_(\d+)", p).group(1)))[-1]
    sp = torch.load(ckpt, map_location=dev, weights_only=True)["splats"]
    means, quats = sp["means"], sp["quats"]
    scales, opac = torch.exp(sp["scales"]), torch.sigmoid(sp["opacities"])
    colors = torch.cat([sp["sh0"], sp["shN"]], 1)
    sh_degree = int(np.sqrt(colors.shape[1]) - 1)

    # Same normalization and split as training (the normalization depends only on
    # the cameras, not on the image resolution)
    parser = Parser(args.data_dir, factor=args.eval_factor, normalize=True, test_every=8)
    valset = Dataset(parser, split="val")

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(dev)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(dev)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(dev)
    m = {"psnr": [], "ssim": [], "lpips": []}
    for i in range(len(valset)):
        d = valset[i]
        pixels = (torch.from_numpy(np.asarray(d["image"])).float().to(dev) / 255.0)[None]
        h, w = pixels.shape[1:3]
        c2w = torch.as_tensor(d["camtoworld"], dtype=torch.float32, device=dev)[None]
        K = torch.as_tensor(d["K"], dtype=torch.float32, device=dev)[None]
        with torch.no_grad():
            out, _, _ = rasterization(means, quats, scales, opac, colors, torch.linalg.inv(c2w), K,
                                      w, h, sh_degree=sh_degree, packed=False)
            out = torch.clamp(out, 0.0, 1.0).permute(0, 3, 1, 2)
            gt = pixels.permute(0, 3, 1, 2)
            m["psnr"].append(psnr(out, gt).item())
            m["ssim"].append(ssim(out, gt).item())
            m["lpips"].append(lpips(out, gt).item())

    res = {k: float(np.mean(v)) for k, v in m.items()}
    res.update(ckpt=ckpt, eval_factor=args.eval_factor, eval_size=[int(w), int(h)],
               n_views=len(valset), num_gs=int(len(means)))
    print(json.dumps(res, indent=1))
    with open(os.path.join(args.result_dir, f"eval_renders_f{args.eval_factor}.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
