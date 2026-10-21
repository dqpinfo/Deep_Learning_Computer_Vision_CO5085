############################################################################
#   Train + evaluate one A1 experiment configuration (hand-written loop)   #
#   Example: python train.py --exp deit_tiny_partial                       #
############################################################################

import argparse
import json
import math
import time

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset

import config
from data import get_datasets
from models import build_model, count_parameters, set_train_mode


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    set_train_mode(model)
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()                       # 1. Clear old gradients
        outputs = model(images)                     # 2. Forward pass
        loss = criterion(outputs, labels)           # 3. Compute the loss
        loss.backward()                             # 4. Backward pass
        optimizer.step()                            # 5. Update the weights
        scheduler.step()
        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        seen += labels.size(0)
    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, preds, targets = 0.0, [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item() * labels.size(0)
        preds.append(outputs.argmax(1).cpu())
        targets.append(labels.cpu())
    preds, targets = torch.cat(preds), torch.cat(targets)
    acc = (preds == targets).float().mean().item()
    macro_f1 = f1_score(targets.numpy(), preds.numpy(), average = "macro")
    return total_loss / len(targets), acc, macro_f1, preds.tolist()


def build_loaders(args):
    train_set, val_set, test_set = get_datasets()
    if args.subset:
        # Only for quick pipeline checks, never for reported results
        train_set, val_set, test_set = (Subset(s, range(min(args.subset, len(s)))) for s in (train_set, val_set, test_set))
    kwargs = {"batch_size": args.batch_size, "num_workers": args.workers}
    return (DataLoader(train_set, shuffle = True, drop_last = True, **kwargs),
            DataLoader(val_set, shuffle = False, **kwargs),
            DataLoader(test_set, shuffle = False, **kwargs))


def run(args):
    torch.manual_seed(config.SEED)
    exp = next(e for e in config.EXPERIMENTS if e["name"] == args.exp)
    device = pick_device()
    model = build_model(exp["arch"], exp["pretrained"], exp["freeze"]).to(device)
    total_params, trainable_params = count_parameters(model)

    hp = config.TRAIN_HPARAMS
    kind = "pretrained" if exp["pretrained"] else "scratch"
    epochs = args.epochs or hp[f"epochs_{kind}"]
    train_loader, val_loader, test_loader = build_loaders(args)
    criterion = nn.CrossEntropyLoss(label_smoothing = hp["label_smoothing"])
    optimizer, scheduler = make_optimizer(model, hp[f"lr_{kind}"], epochs, len(train_loader))

    log = {"experiment": exp, "device": str(device), "epochs": epochs,
           "params_total": total_params, "params_trainable": trainable_params, "history": []}
    best_acc, best_state, start = -1.0, None, time.perf_counter()
    for epoch in range(epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        va_loss, va_acc, va_f1, _ = evaluate(model, val_loader, criterion, device)
        log["history"].append({"epoch": epoch + 1, "train_loss": tr_loss, "train_acc": tr_acc,
                               "val_loss": va_loss, "val_acc": va_acc, "val_macro_f1": va_f1})
        print(f"[{exp['name']}] epoch {epoch + 1}/{epochs} | train acc {tr_acc:.4f} | val acc {va_acc:.4f} | val F1 {va_f1:.4f}")
        if va_acc > best_acc:
            best_acc, best_state = va_acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    log["train_time_sec"] = time.perf_counter() - start
    if device.type == "cuda":
        log["peak_gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20

    # Evaluate on test with the checkpoint that has the best validation accuracy
    model.load_state_dict(best_state)
    te_loss, te_acc, te_f1, te_preds = evaluate(model, test_loader, criterion, device)
    log.update({"best_val_acc": best_acc, "test_loss": te_loss, "test_acc": te_acc, "test_macro_f1": te_f1})
    save_outputs(args, exp, log, best_state, te_preds)


def save_outputs(args, exp, log, best_state, test_preds):
    run_dir = config.OUTPUT_DIR / ("smoke" if args.subset else "runs")
    run_dir.mkdir(parents = True, exist_ok = True)
    (run_dir / f"{exp['name']}.json").write_text(json.dumps(log, indent = 2))
    (run_dir / f"{exp['name']}_test_preds.json").write_text(json.dumps(test_preds))
    if not args.subset:
        torch.save(best_state, run_dir / f"{exp['name']}_best.pth")
    print(f"Test acc {log['test_acc']:.4f} | macro-F1 {log['test_macro_f1']:.4f} | log: {run_dir / (exp['name'] + '.json')}")


def parse_args():
    parser = argparse.ArgumentParser(description = "Huấn luyện một cấu hình A1")
    parser.add_argument("--exp", required = True, choices = [e["name"] for e in config.EXPERIMENTS])
    parser.add_argument("--epochs", type = int, default = None, help = "Mặc định lấy theo config.TRAIN_HPARAMS")
    parser.add_argument("--batch-size", type = int, default = config.TRAIN_HPARAMS["batch_size"])
    parser.add_argument("--workers", type = int, default = 2)
    parser.add_argument("--subset", type = int, default = 0, help = "Chạy thử với N ảnh mỗi tập (0 = dùng toàn bộ)")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
