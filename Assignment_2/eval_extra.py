##########################################################################
#   Extra test-set evaluations of the 100% models (no retraining),       #
#   to check two claims of the SegFormer paper on our task:              #
#   1. test resolution different from training (no positional encoding)  #
#   2. robustness to image corruptions (noise, blur, JPEG)               #
#   Example: python eval_extra.py --arch segformer_b2 --run-dir <dir>    #
#   Output: <run-dir>/extra_<arch>.json                                  #
##########################################################################

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

import config
from data import PetSegmentation, get_rows, get_split
from metrics import SegMetrics
from models import build_model
from train import pick_device

TEST_SIZES = (256, 320, 384, 512, 640)              # Training size is 384
CORRUPTIONS = {                                      # 3 severity levels each, applied before normalization
    "gaussian_noise": [v2.GaussianNoise(sigma = s, clip = True) for s in (0.04, 0.08, 0.16)],
    "gaussian_blur": [v2.GaussianBlur(kernel_size = 2 * round(3 * s) + 1, sigma = s) for s in (1.0, 2.0, 4.0)],
    "jpeg": [v2.JPEG(quality = q) for q in (30, 15, 5)],
}


def make_transform(size, corruption = None):
    # Same as data.eval_transform but with a chosen size; JPEG works on uint8, noise/blur on float in [0, 1]
    steps = [v2.Resize((size, size), antialias = True)]
    if isinstance(corruption, v2.JPEG):
        steps.append(corruption)
    steps.append(v2.ToDtype(torch.float32, scale = True))
    if corruption is not None and not isinstance(corruption, v2.JPEG):
        steps.append(corruption)
    steps.append(v2.Normalize(config.MEAN, config.STD))
    return v2.Compose(steps)


@torch.no_grad()
def score(model, rows, transform, device, args):
    loader = DataLoader(PetSegmentation(rows, transform), batch_size = args.batch_size, num_workers = args.workers,
                        pin_memory = device.type == "cuda")
    metrics = SegMetrics()
    for images, masks in loader:
        with torch.autocast(device.type, dtype = torch.float16, enabled = device.type == "cuda"):
            logits = model(images.to(device, non_blocking = True))
        metrics.update(logits.argmax(1), masks.to(device))
    result = metrics.compute()
    return {"miou": result["miou"], "pet_iou": result["pet_iou"], "pixel_acc": result["pixel_acc"]}


def main():
    parser = argparse.ArgumentParser(description = "Đánh giá thêm mô hình 100% dữ liệu trên tập test")
    parser.add_argument("--arch", required = True, choices = list(config.ARCHS))
    parser.add_argument("--run-dir", default = str(config.OUTPUT_DIR / "runs"))
    parser.add_argument("--batch-size", type = int, default = 16)
    parser.add_argument("--workers", type = int, default = 2)
    parser.add_argument("--subset", type = int, default = 0, help = "Chỉ dùng N ảnh test (chạy thử)")
    parser.add_argument("--device", default = None)
    args = parser.parse_args()

    run_dir, device = Path(args.run_dir), pick_device(args.device)
    model = build_model(args.arch, pretrained = False)
    model.load_state_dict(torch.load(run_dir / f"{args.arch}_f100_best.pth", map_location = "cpu"))
    model.to(device).eval()
    all_rows = get_rows()
    rows = [all_rows[n] for n in get_split()[2]]
    if args.subset:
        rows = rows[::max(1, len(rows) // args.subset)][:args.subset]

    out = {"arch": args.arch, "test_images": len(rows), "train_size": config.IMG_SIZE, "resolution": {}, "corruption": {}}
    start = time.perf_counter()
    for size in TEST_SIZES:
        out["resolution"][size] = score(model, rows, make_transform(size), device, args)
        print(f"[{args.arch}] {size}x{size}: mIoU {out['resolution'][size]['miou']:.4f}", flush = True)
    for name, levels in CORRUPTIONS.items():
        out["corruption"][name] = [score(model, rows, make_transform(config.IMG_SIZE, c), device, args) for c in levels]
        print(f"[{args.arch}] {name}: mIoU " + ", ".join(f"{r['miou']:.4f}" for r in out["corruption"][name]), flush = True)
    out["seconds"] = time.perf_counter() - start
    (run_dir / f"extra_{args.arch}.json").write_text(json.dumps(out, indent = 1))


if __name__ == "__main__":
    main()
