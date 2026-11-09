###################################################################
#   Collect the numbers used in the E1-E3 report for Exercise 3   #
#   (uses the committed weights in output/, does not retrain)     #
#   Output: output/report_metrics.json, output/rows_seen.png      #
###################################################################

import importlib.util
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from dataset import get_data_loaders
from models import ImageRNNClassifier
from train import train_model

OUTPUT_DIR = Path("./output")
E1_DIR = Path("../Exercise_1")
RNN_TYPES = ("LSTM", "GRU")
ROWS_SEEN = (4, 7, 10, 14, 18, 21, 24, 28)
BACKGROUND = -1.0                                   # Value of a blank pixel after Normalize(0.5, 0.5)


def load_e1_models():
    # Import Exercise_1/models.py under another name (both exercises have a models.py) and load the E1 baselines
    spec = importlib.util.spec_from_file_location("e1_models", E1_DIR / "models.py")
    e1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e1)
    baselines = {}
    for name, cls in (("MLP", e1.MLPClassifier), ("CNN", e1.CNNClassifier)):
        model = cls()
        model.load_state_dict(torch.load(E1_DIR / "output" / f"{name.lower()}_weights.pth"))
        baselines[name] = model.eval()
    return baselines


def load_rnn(rnn_type):
    model = ImageRNNClassifier(rnn_type = rnn_type, input_size = 28, hidden_size = 128, num_classes = 10)
    model.load_state_dict(torch.load(OUTPUT_DIR / f"{rnn_type.lower()}_weights.pth"))
    return model.eval()


def predict(model, loader, transform = None):
    # Predictions on the whole loader (optionally after transforming the images) plus wall-clock time
    preds, labels = [], []
    start = time.perf_counter()
    with torch.no_grad():
        for images, targets in loader:
            if transform is not None:
                images = transform(images)
            preds.append(model(images).argmax(1))
            labels.append(targets)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy(), time.perf_counter() - start


def top_confusions(cm, k = 3):
    # Most frequent (true, predicted) pairs off the diagonal
    off = cm.copy()
    np.fill_diagonal(off, 0)
    order = np.argsort(off, axis = None)[::-1][:k]
    return [{"true": int(i // 10), "pred": int(i % 10), "count": int(off.flat[i])} for i in order]


def mask_after(k):
    # Keep the top k rows, blank out the rest: every model still gets a full 28x28 image
    def transform(images):
        images = images.clone()
        images[:, :, k:, :] = BACKGROUND
        return images
    return transform


def truncate_to(k):
    # Keep only the top k rows: the RNN reads a sequence of k time steps and stops there
    return lambda images: images[:, :, :k, :]


def accuracy(model, loader, transform):
    preds, labels, _ = predict(model, loader, transform)
    return float((preds == labels).mean() * 100)


def rows_seen_study(rnns, baselines, test_loader):
    # How much of the digit (counted in rows from the top) each model needs, and whether the RNNs
    # still remember the top rows after reading 28 - k blank rows (masked) instead of stopping (truncated)
    study = {"rows": list(ROWS_SEEN), "masked": {}, "truncated": {}}
    for name, model in {**baselines, **rnns}.items():
        study["masked"][name] = [accuracy(model, test_loader, mask_after(k)) for k in ROWS_SEEN]
    for name, model in rnns.items():
        study["truncated"][name] = [accuracy(model, test_loader, truncate_to(k)) for k in ROWS_SEEN]
    return study


def plot_rows_seen(study):
    colors = {"MLP": "#a8a29e", "CNN": "#57534e", "LSTM": "#c2410c", "GRU": "#2563eb"}
    _, ax = plt.subplots(figsize = (9, 4.5))
    for name, values in study["masked"].items():
        ax.plot(study["rows"], values, marker = "o", color = colors[name], linewidth = 2, label = f"{name} (che phần dưới)")
    for name, values in study["truncated"].items():
        ax.plot(study["rows"], values, marker = "s", color = colors[name], linewidth = 1.5, linestyle = "--",
                label = f"{name} (dừng chuỗi sau k hàng)")
    ax.set_xlabel("Số hàng đầu tiên mô hình được thấy (k)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Accuracy khi chỉ thấy k hàng đầu của ảnh (10.000 ảnh test)")
    ax.set_xticks(study["rows"])
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle = "--", alpha = 0.5)
    ax.legend(fontsize = 8, ncols = 2, loc = "lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rows_seen.png", dpi = 150)
    plt.close()


def epoch_time(rnn_type):
    # Time one training epoch of a fresh model with the same loop as train.py (the result is discarded)
    torch.manual_seed(0)
    start = time.perf_counter()
    train_model(rnn_type = rnn_type, batch_size = 64, epochs = 1, lr = 0.001)
    return time.perf_counter() - start


def main():
    train_loader, test_loader = get_data_loaders(batch_size = 64)
    history = json.loads((OUTPUT_DIR / "training_history.json").read_text())
    rnns = {name: load_rnn(name) for name in RNN_TYPES}
    baselines = load_e1_models()
    results = {"device": "cpu", "train_size": len(train_loader.dataset), "test_size": len(test_loader.dataset),
               "sequence_mode": "row", "models": {}}

    for name, model in rnns.items():
        preds, labels, infer_sec = predict(model, test_loader)
        cm = confusion_matrix(labels, preds)
        results["models"][name] = {
            "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "final_train_acc": history[name]["train_acc"][-1],
            "test_acc": float((preds == labels).mean() * 100),
            "test_macro_f1": float(f1_score(labels, preds, average = "macro") * 100),
            "test_errors": int((preds != labels).sum()),
            "per_class_acc": (cm.diagonal() / cm.sum(1) * 100).round(2).tolist(),
            "top_confusions": top_confusions(cm),
            "inference_sec_test": infer_sec,
        }
    # Inference time of the E1 baselines on the same machine, for the cost comparison
    results["baseline_inference_sec_test"] = {name: predict(m, test_loader)[2] for name, m in baselines.items()}

    results["rows_seen"] = rows_seen_study(rnns, baselines, test_loader)
    plot_rows_seen(results["rows_seen"])

    for name in RNN_TYPES:
        results["models"][name]["train_epoch_sec"] = epoch_time(name)
        print(name, {k: v for k, v in results["models"][name].items() if k != "per_class_acc"})

    (OUTPUT_DIR / "report_metrics.json").write_text(json.dumps(results, indent = 2))


if __name__ == "__main__":
    main()
