############################################################################
#   Train + evaluate one A1 experiment configuration (hand-written loop)   #
#   Example: python train.py --exp deit_tiny_partial                       #
#   A checkpoint is saved after every epoch; rerunning the same command    #
#   resumes from it (useful when a Colab session disconnects).             #
############################################################################

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset

import config
from data import get_datasets, to_model_input
from models import build_model, count_parameters, set_train_mode


def pick_device():
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


def make_optimizer(model, lr, epochs, steps_per_epoch):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr = lr, weight_decay = config.TRAIN_HPARAMS["weight_decay"])
    total, warmup = epochs * steps_per_epoch, steps_per_epoch

    def lr_factor(step):
        # 1 epoch of linear warmup, then cosine decay
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device):
    set_train_mode(model)
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images = to_model_input(images.to(device, non_blocking = True))
        labels = labels.to(device, non_blocking = True)
        optimizer.zero_grad(set_to_none = True)                                       # 1. Clear old gradients
        with torch.autocast(device.type, dtype = torch.float16, enabled = scaler.is_enabled()):
            outputs = model(images)                                                   # 2. Forward pass (fp16 on CUDA)
            loss = criterion(outputs, labels)                                         # 3. Compute the loss
        scaler.scale(loss).backward()                                                 # 4. Backward pass (scaled to avoid fp16 underflow)
        scaler.step(optimizer)                                                        # 5. Update the weights
        scaler.update()
        scheduler.step()
        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        seen += labels.size(0)
    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss, preds, targets = 0.0, [], []
    for images, labels in loader:
        images, labels = to_model_input(images.to(device, non_blocking = True)), labels.to(device, non_blocking = True)
        with torch.autocast(device.type, dtype = torch.float16, enabled = use_amp):
            outputs = model(images)
        total_loss += criterion(outputs.float(), labels).item() * labels.size(0)
        preds.append(outputs.argmax(1).cpu())
        targets.append(labels.cpu())
    preds, targets = torch.cat(preds), torch.cat(targets)
    acc = (preds == targets).float().mean().item()
    macro_f1 = f1_score(targets.numpy(), preds.numpy(), average = "macro")
    return total_loss / len(targets), acc, macro_f1, preds.tolist()


def build_loaders(args, device):
    train_set, val_set, test_set = get_datasets()
    if args.subset:
        # Only for quick pipeline checks, never for reported results
        train_set, val_set, test_set = (Subset(s, range(min(args.subset, len(s)))) for s in (train_set, val_set, test_set))
    kwargs = {"batch_size": args.batch_size, "num_workers": args.workers, "pin_memory": device.type == "cuda",
              "persistent_workers": args.workers > 0}
    return (DataLoader(train_set, shuffle = True, drop_last = True, **kwargs),
            DataLoader(val_set, shuffle = False, **kwargs),
            DataLoader(test_set, shuffle = False, **kwargs))


def save_checkpoint(path, model, optimizer, scheduler, scaler, log, best_acc, best_state):
    # Everything needed to continue training exactly where it stopped
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "scaler": scaler.state_dict(), "log": log, "best_acc": best_acc, "best_state": best_state,
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
    return state["log"], state["best_acc"], state["best_state"]


@torch.no_grad()
def measure_inference(model, loader, device, use_amp):
    # Test-set throughput (images/second) of the final model, preprocessing on the GPU included
    model.eval()
    seen, start = 0, None
    for i, (images, _) in enumerate(loader):
        if i == 1:                                      # Skip the first batch (warm-up / kernel selection)
            sync(device)
            start, seen = time.perf_counter(), 0
        with torch.autocast(device.type, dtype = torch.float16, enabled = use_amp):
            model(to_model_input(images.to(device, non_blocking = True)))
        seen += images.size(0)
    sync(device)
    return seen / (time.perf_counter() - start) if start else float("nan")


def run(args):
    torch.manual_seed(config.SEED)
    exp = next(e for e in config.EXPERIMENTS if e["name"] == args.exp)
    device = pick_device()
    use_amp = device.type == "cuda" and not args.no_amp
    run_dir = Path(args.run_dir) if args.run_dir else config.OUTPUT_DIR / ("smoke" if args.subset else "runs")
    run_dir.mkdir(parents = True, exist_ok = True)
    if (run_dir / f"{exp['name']}.json").exists() and not (args.force or args.subset):
        print(f"[{exp['name']}] already finished, skipping (use --force to retrain)")
        return

    model = build_model(exp["arch"], exp["pretrained"], exp["freeze"]).to(device)
    total_params, trainable_params = count_parameters(model)

    hp = config.TRAIN_HPARAMS
    kind = "pretrained" if exp["pretrained"] else "scratch"
    epochs = args.epochs or hp[f"epochs_{kind}"]
    train_loader, val_loader, test_loader = build_loaders(args, device)
    criterion = nn.CrossEntropyLoss(label_smoothing = hp["label_smoothing"])
    optimizer, scheduler = make_optimizer(model, hp[f"lr_{kind}"], epochs, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled = use_amp)

    ckpt_path = run_dir / f"{exp['name']}_last.pth"
    if ckpt_path.exists():
        log, best_acc, best_state = load_checkpoint(ckpt_path, model, optimizer, scheduler, scaler)
        print(f"[{exp['name']}] resuming after epoch {len(log['history'])}/{epochs}")
    else:
        log = {"experiment": exp, "device": str(device), "amp": use_amp, "epochs": epochs,
               "batch_size": args.batch_size, "lr": hp[f"lr_{kind}"],
               "params_total": total_params, "params_trainable": trainable_params, "history": []}
        best_acc, best_state = -1.0, None
    if device.type == "cuda":
        log["gpu"] = torch.cuda.get_device_name()
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(len(log["history"]), epochs):
        sync(device)
        start = time.perf_counter()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device)
        sync(device)
        train_sec = time.perf_counter() - start
        va_loss, va_acc, va_f1, _ = evaluate(model, val_loader, criterion, device, use_amp)
        log["history"].append({"epoch": epoch + 1, "train_loss": tr_loss, "train_acc": tr_acc,
                               "val_loss": va_loss, "val_acc": va_acc, "val_macro_f1": va_f1,
                               "train_sec": train_sec, "epoch_sec": time.perf_counter() - start})
        print(f"[{exp['name']}] epoch {epoch + 1}/{epochs} | train acc {tr_acc:.4f} | val acc {va_acc:.4f} "
              f"| val F1 {va_f1:.4f} | {train_sec:.0f}s")
        if va_acc > best_acc:
            best_acc, best_state = va_acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if device.type == "cuda":
            log["peak_gpu_mem_mb"] = max(log.get("peak_gpu_mem_mb", 0), torch.cuda.max_memory_allocated() / 2**20)
        save_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, log, best_acc, best_state)

    # Sum of per-epoch times, so the cost stays correct when the run was resumed across sessions
    log["train_time_sec"] = sum(h["train_sec"] for h in log["history"])
    log["total_time_sec"] = sum(h["epoch_sec"] for h in log["history"])

    # Evaluate on test with the checkpoint that has the best validation accuracy
    model.load_state_dict(best_state)
    te_loss, te_acc, te_f1, te_preds = evaluate(model, test_loader, criterion, device, use_amp)
    log.update({"best_val_acc": best_acc, "best_epoch": 1 + max(range(epochs), key = lambda i: log["history"][i]["val_acc"]),
                "test_loss": te_loss, "test_acc": te_acc, "test_macro_f1": te_f1,
                "inference_img_per_sec": measure_inference(model, test_loader, device, use_amp)})
    save_outputs(run_dir, exp, log, best_state, te_preds, keep_weights = not args.subset)
    for leftover in (ckpt_path, ckpt_path.with_suffix(".tmp")):
        leftover.unlink(missing_ok = True)              # The run is finished; the best weights are kept in *_best.pth


def save_outputs(run_dir, exp, log, best_state, test_preds, keep_weights):
    (run_dir / f"{exp['name']}_test_preds.json").write_text(json.dumps(test_preds))
    if keep_weights:
        torch.save(best_state, run_dir / f"{exp['name']}_best.pth")
    # The log is written last: its presence marks the run as finished
    (run_dir / f"{exp['name']}.json").write_text(json.dumps(log, indent = 2))
    print(f"Test acc {log['test_acc']:.4f} | macro-F1 {log['test_macro_f1']:.4f} | log: {run_dir / (exp['name'] + '.json')}")


def parse_args():
    parser = argparse.ArgumentParser(description = "Huấn luyện một cấu hình A1")
    parser.add_argument("--exp", required = True, choices = [e["name"] for e in config.EXPERIMENTS])
    parser.add_argument("--epochs", type = int, default = None, help = "Mặc định lấy theo config.TRAIN_HPARAMS")
    parser.add_argument("--batch-size", type = int, default = config.TRAIN_HPARAMS["batch_size"])
    parser.add_argument("--workers", type = int, default = 2)
    parser.add_argument("--subset", type = int, default = 0, help = "Chạy thử với N ảnh mỗi tập (0 = dùng toàn bộ)")
    parser.add_argument("--run-dir", default = None, help = "Thư mục lưu log/checkpoint (mặc định output/runs)")
    parser.add_argument("--force", action = "store_true", help = "Huấn luyện lại dù đã có log hoàn tất")
    parser.add_argument("--no-amp", action = "store_true", help = "Tắt mixed precision trên CUDA")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
