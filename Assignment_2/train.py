##########################################################################
#   Train + evaluate one A2 experiment (hand-written loop, based on A1)  #
#   Example: python train.py --exp segformer_b2_f100                     #
#   A checkpoint is saved after every epoch; rerunning the same command  #
#   resumes from it (useful when a Colab session disconnects).           #
##########################################################################

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import config
from data import get_datasets
from metrics import SegMetrics
from models import build_model, count_parameters, param_groups


def pick_device(name = None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    # Wait for queued GPU work so wall-clock timings are correct
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def make_optimizer(model, epochs, steps_per_epoch):
    # AdamW with two learning rates (pretrained encoder vs new decoder), 1 epoch linear warmup then "poly" decay
    hp = config.TRAIN_HPARAMS
    encoder, decoder = param_groups(model)
    optimizer = torch.optim.AdamW([{"params": encoder, "lr": hp["lr_encoder"]},
                                   {"params": decoder, "lr": hp["lr_decoder"]}], weight_decay = hp["weight_decay"])
    total, warmup = epochs * steps_per_epoch, steps_per_epoch

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, 1 - (step - warmup) / max(1, total - warmup)) ** hp["poly_power"]

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device):
    model.train()
    total_loss, seen = 0.0, 0
    for images, masks in loader:
        images = images.to(device, non_blocking = True)
        masks = masks.to(device, non_blocking = True)
        optimizer.zero_grad(set_to_none = True)                                       # 1. Clear old gradients
        with torch.autocast(device.type, dtype = torch.float16, enabled = scaler.is_enabled()):
            logits = model(images)                                                    # 2. Forward pass (fp16 on CUDA)
            loss = criterion(logits.float(), masks)                                   # 3. Pixel-wise CE, border ignored
        scaler.scale(loss).backward()                                                 # 4. Backward pass (scaled for fp16)
        scaler.step(optimizer)                                                        # 5. Update the weights
        scaler.update()
        scheduler.step()
        total_loss += loss.item() * images.size(0)
        seen += images.size(0)
    return total_loss / seen


def image_stats(preds, masks):
    # Per-image numbers used to pick typical failures in analyze.py:
    # pet IoU (37 breeds merged) and the breed predicted for the image (majority vote over predicted pet pixels)
    rows = []
    for pred, mask in zip(preds, masks):
        valid = mask != config.IGNORE_INDEX
        gt_pet, pr_pet = (mask > 0) & valid, (pred > 0) & valid
        union = (gt_pet | pr_pet).sum().item()
        breeds = pred[pr_pet]
        pred_breed = int(breeds.bincount(minlength = config.NUM_CLASSES).argmax()) if breeds.numel() else 0
        true_breed = int(mask[gt_pet].max()) if gt_pet.any() else 0
        rows.append({"pet_iou": (gt_pet & pr_pet).sum().item() / union if union else 1.0,
                     "true_breed": true_breed, "pred_breed": pred_breed})
    return rows


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp, per_image = False):
    model.eval()
    metrics, total_loss, seen, images_out = SegMetrics(), 0.0, 0, []
    for images, masks in loader:
        images, masks = images.to(device, non_blocking = True), masks.to(device, non_blocking = True)
        with torch.autocast(device.type, dtype = torch.float16, enabled = use_amp):
            logits = model(images)
        total_loss += criterion(logits.float(), masks).item() * images.size(0)
        seen += images.size(0)
        preds = logits.argmax(1)
        metrics.update(preds, masks)
        if per_image:
            images_out.extend(image_stats(preds.cpu(), masks.cpu()))
    result = metrics.compute()
    result["loss"] = total_loss / seen
    result["confusion"] = metrics.confusion.tolist()
    if per_image:
        result["breed_acc_image"] = sum(r["pred_breed"] == r["true_breed"] for r in images_out) / len(images_out)
    return result, images_out


def build_loaders(args, exp, device):
    train_set, val_set, test_set = get_datasets(exp["train_fraction"])
    if args.subset:
        # Only for quick pipeline checks, never for reported results
        train_set, val_set, test_set = (Subset(s, range(min(args.subset, len(s)))) for s in (train_set, val_set, test_set))
    kwargs = {"batch_size": args.batch_size, "num_workers": args.workers, "pin_memory": device.type == "cuda",
              "persistent_workers": args.workers > 0}
    return (DataLoader(train_set, shuffle = True, drop_last = True, **kwargs),
            DataLoader(val_set, shuffle = False, **kwargs),
            DataLoader(test_set, shuffle = False, **kwargs))


def save_checkpoint(path, model, optimizer, scheduler, scaler, log, best_miou, best_state):
    # Everything needed to continue training exactly where it stopped
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "scaler": scaler.state_dict(), "log": log, "best_miou": best_miou, "best_state": best_state,
             "rng_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["rng_cuda"] = torch.cuda.get_rng_state_all()
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)                                   # Atomic rename: a disconnect never leaves a half-written checkpoint


def load_checkpoint(path, model, optimizer, scheduler, scaler):
    state = torch.load(path, map_location = "cpu", weights_only = False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    torch.set_rng_state(state["rng_cpu"])
    if "rng_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["rng_cuda"])
    return state["log"], state["best_miou"], state["best_state"]


@torch.no_grad()
def measure_inference(model, loader, device, use_amp, batches = 20):
    # Inference throughput (images/second) at batch size 16, data loading excluded
    model.eval()
    images = next(iter(loader))[0].to(device)
    for _ in range(3):                                  # Warm-up (kernel selection, caches)
        with torch.autocast(device.type, dtype = torch.float16, enabled = use_amp):
            model(images)
    sync(device)
    start = time.perf_counter()
    for _ in range(batches):
        with torch.autocast(device.type, dtype = torch.float16, enabled = use_amp):
            model(images)
    sync(device)
    return batches * images.size(0) / (time.perf_counter() - start)


def short(result):
    # Epoch log entry without the large confusion matrix / per-class list
    return {k: v for k, v in result.items() if k not in ("confusion", "per_class_iou")}


def run(args):
    torch.manual_seed(config.SEED)
    exp = next(e for e in config.EXPERIMENTS if e["name"] == args.exp)
    device = pick_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    run_dir = Path(args.run_dir) if args.run_dir else config.OUTPUT_DIR / ("smoke" if args.subset else "runs")
    run_dir.mkdir(parents = True, exist_ok = True)
    if (run_dir / f"{exp['name']}.json").exists() and not (args.force or args.subset):
        print(f"[{exp['name']}] đã xong, bỏ qua (dùng --force để chạy lại)")
        return

    model = build_model(exp["arch"], pretrained = True).to(device)
    hp = config.TRAIN_HPARAMS
    epochs = args.epochs or hp["epochs"]
    train_loader, val_loader, test_loader = build_loaders(args, exp, device)
    criterion = nn.CrossEntropyLoss(ignore_index = config.IGNORE_INDEX)
    optimizer, scheduler = make_optimizer(model, epochs, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled = use_amp)

    ckpt_path = run_dir / f"{exp['name']}_last.pth"
    if ckpt_path.exists():
        log, best_miou, best_state = load_checkpoint(ckpt_path, model, optimizer, scheduler, scaler)
        print(f"[{exp['name']}] chạy tiếp sau epoch {len(log['history'])}/{epochs}")
    else:
        log = {"experiment": exp, "device": str(device), "amp": use_amp, "epochs": epochs,
               "batch_size": args.batch_size, "img_size": config.IMG_SIZE, "hparams": hp,
               "train_images": len(train_loader.dataset), "params": count_parameters(model), "history": []}
        best_miou, best_state = -1.0, None
    if device.type == "cuda":
        log["gpu"] = torch.cuda.get_device_name()
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(len(log["history"]), epochs):
        sync(device)
        start = time.perf_counter()
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device)
        sync(device)
        train_sec = time.perf_counter() - start
        val, _ = evaluate(model, val_loader, criterion, device, use_amp)
        log["history"].append({"epoch": epoch + 1, "train_loss": tr_loss, "lr_encoder": optimizer.param_groups[0]["lr"],
                               **{f"val_{k}": v for k, v in short(val).items()},
                               "train_sec": train_sec, "epoch_sec": time.perf_counter() - start})
        print(f"[{exp['name']}] epoch {epoch + 1}/{epochs} | train loss {tr_loss:.4f} | val loss {val['loss']:.4f} "
              f"| val mIoU {val['miou']:.4f} | pet IoU {val['pet_iou']:.4f} | {train_sec:.0f}s", flush = True)
        if val["miou"] > best_miou:
            best_miou, best_state = val["miou"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if device.type == "cuda":
            log["peak_gpu_mem_mb"] = max(log.get("peak_gpu_mem_mb", 0), torch.cuda.max_memory_allocated() / 2**20)
        save_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, log, best_miou, best_state)

    # Sum of per-epoch times, so the cost stays correct when the run was resumed across sessions
    log["train_time_sec"] = sum(h["train_sec"] for h in log["history"])
    log["total_time_sec"] = sum(h["epoch_sec"] for h in log["history"])

    # Validation and test (once) with the checkpoint that has the best validation mIoU
    model.load_state_dict(best_state)
    val, _ = evaluate(model, val_loader, criterion, device, use_amp)
    test, test_images = evaluate(model, test_loader, criterion, device, use_amp, per_image = True)
    dataset = test_loader.dataset
    rows = [dataset.dataset.rows[i] for i in dataset.indices] if isinstance(dataset, Subset) else dataset.rows
    for stats, row in zip(test_images, rows):
        stats["name"] = row["name"]
    log.update({"best_epoch": 1 + max(range(epochs), key = lambda i: log["history"][i]["val_miou"]),
                "val": val, "test": test, "inference_img_per_sec": measure_inference(model, test_loader, device, use_amp)})
    save_outputs(run_dir, exp, log, best_state, test_images, keep_weights = not args.subset)
    for leftover in (ckpt_path, ckpt_path.with_suffix(".tmp")):
        leftover.unlink(missing_ok = True)              # The run is finished; the best weights are kept in *_best.pth


def save_outputs(run_dir, exp, log, best_state, test_images, keep_weights):
    (run_dir / f"{exp['name']}_test_images.json").write_text(json.dumps(test_images))
    if keep_weights:
        torch.save(best_state, run_dir / f"{exp['name']}_best.pth")
    # The log is written last: its presence marks the run as finished
    (run_dir / f"{exp['name']}.json").write_text(json.dumps(log, indent = 1))
    t = log["test"]
    print(f"[{exp['name']}] TEST mIoU {t['miou']:.4f} | mDice {t['mdice']:.4f} | pixel acc {t['pixel_acc']:.4f} "
          f"| pet IoU {t['pet_iou']:.4f} | giống đúng (theo ảnh) {t['breed_acc_image']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description = "Huấn luyện một cấu hình A2")
    parser.add_argument("--exp", required = True, choices = [e["name"] for e in config.EXPERIMENTS])
    parser.add_argument("--epochs", type = int, default = None, help = "Mặc định lấy theo config.TRAIN_HPARAMS")
    parser.add_argument("--batch-size", type = int, default = config.TRAIN_HPARAMS["batch_size"])
    parser.add_argument("--workers", type = int, default = 2)
    parser.add_argument("--subset", type = int, default = 0, help = "Chạy thử với N ảnh mỗi tập (0 = dùng toàn bộ)")
    parser.add_argument("--run-dir", default = None, help = "Thư mục lưu log/checkpoint (mặc định output/runs)")
    parser.add_argument("--force", action = "store_true", help = "Huấn luyện lại dù đã có log hoàn tất")
    parser.add_argument("--device", default = None, help = "Ép thiết bị (cuda/mps/cpu); mặc định tự chọn")
    parser.add_argument("--no-amp", action = "store_true", help = "Tắt mixed precision trên CUDA")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
