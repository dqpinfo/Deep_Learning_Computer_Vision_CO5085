###################################################################
#   Collect the numbers used in the E1-E3 report for Exercise 1   #
#   (uses the committed weights in output/, does not retrain)     #
#   Output: output/report_metrics.json                            #
###################################################################

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from dataset import get_data_loaders
from models import CNNClassifier, MLPClassifier, SoftmaxClassifier
from train import train_model

OUTPUT_DIR = Path("./output")
MODELS = {"Softmax": SoftmaxClassifier, "MLP": MLPClassifier, "CNN": CNNClassifier}


def predict(model, loader):
    # Predictions on the whole loader plus wall-clock inference time
    model.eval()
    preds, labels = [], []
    start = time.perf_counter()
    with torch.no_grad():
        for images, targets in loader:
            preds.append(model(images).argmax(1))
            labels.append(targets)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy(), time.perf_counter() - start


def top_confusions(cm, k = 3):
    # Most frequent (true, predicted) pairs off the diagonal
    off = cm.copy()
    np.fill_diagonal(off, 0)
    order = np.argsort(off, axis = None)[::-1][:k]
    return [{"true": int(i // 10), "pred": int(i % 10), "count": int(off.flat[i])} for i in order]


def epoch_time(model_cls, train_loader):
    # Time one training epoch of a fresh model with the same loop as train.py (the result is discarded)
    torch.manual_seed(0)
    start = time.perf_counter()
    train_model(model_cls(), train_loader, epochs = 1, lr = 0.001)
    return time.perf_counter() - start


def main():
    train_loader, test_loader = get_data_loaders(batch_size = 64)
    history = json.loads((OUTPUT_DIR / "training_history.json").read_text())
    results = {"device": "cpu", "train_size": len(train_loader.dataset), "test_size": len(test_loader.dataset), "models": {}}

    for name, model_cls in MODELS.items():
        model = model_cls()
        model.load_state_dict(torch.load(OUTPUT_DIR / f"{name.lower()}_weights.pth"))
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
            "train_epoch_sec": epoch_time(model_cls, train_loader),
        }
        print(name, {k: v for k, v in results["models"][name].items() if k != "per_class_acc"})

    (OUTPUT_DIR / "report_metrics.json").write_text(json.dumps(results, indent = 2))


if __name__ == "__main__":
    main()
