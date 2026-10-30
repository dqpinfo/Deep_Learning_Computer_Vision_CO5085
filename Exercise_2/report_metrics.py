###################################################################
#   Collect the numbers used in the E1-E3 report for Exercise 2   #
#   (uses the committed weights in output/, does not retrain)     #
#   Output: output/report_metrics.json                            #
###################################################################

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from dataset import get_data_loaders
from models import ImageTransformerClassifier, MyMSA, PatchTokenizer, PyTorchMSA, RowTokenizer
from train import train_model

OUTPUT_DIR = Path("./output")
EMBED_DIM, NUM_HEADS = 64, 4
TOKENIZERS = {
    "PatchTokenizer": lambda: PatchTokenizer(img_size = 28, patch_size = 7, in_channels = 1, embed_dim = EMBED_DIM),
    "RowTokenizer": lambda: RowTokenizer(img_size = 28, in_channels = 1, embed_dim = EMBED_DIM),
}
ATTENTIONS = {"MyMSA": MyMSA, "PyTorchMSA": PyTorchMSA}


def build(tokenizer, attention):
    return ImageTransformerClassifier(tokenizer = TOKENIZERS[tokenizer](), attention = ATTENTIONS[attention](EMBED_DIM, NUM_HEADS),
                                      embed_dim = EMBED_DIM, num_classes = 10)


def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    start = time.perf_counter()
    with torch.no_grad():
        for images, targets in loader:
            preds.append(model(images).argmax(1))
            labels.append(targets)
    preds, labels = torch.cat(preds), torch.cat(labels)
    return {"acc": float((preds == labels).float().mean() * 100),
            "macro_f1": float(f1_score(labels, preds, average = "macro") * 100),
            "inference_sec": time.perf_counter() - start}


def msa_equivalence_check():
    # Copy the weights of nn.MultiheadAttention into MyMSA and compare outputs on random tokens.
    # A near-zero difference shows the hand-written MSA computes exactly the same function.
    torch.manual_seed(0)
    mine, ref = MyMSA(EMBED_DIM, NUM_HEADS), nn.MultiheadAttention(EMBED_DIM, NUM_HEADS, batch_first = True)
    w_q, w_k, w_v = ref.in_proj_weight.chunk(3)
    b_q, b_k, b_v = ref.in_proj_bias.chunk(3)
    with torch.no_grad():
        for layer, w, b in ((mine.q_linear, w_q, b_q), (mine.k_linear, w_k, b_k), (mine.v_linear, w_v, b_v),
                            (mine.out_linear, ref.out_proj.weight, ref.out_proj.bias)):
            layer.weight.copy_(w)
            layer.bias.copy_(b)
        x = torch.randn(8, 17, EMBED_DIM)
        diff = (mine(x) - ref(x, x, x)[0]).abs().max().item()
    return {"max_abs_diff": diff, "input_shape": [8, 17, EMBED_DIM]}


def epoch_time(tokenizer, attention, train_loader):
    # Time one training epoch of a fresh model with the same loop as train.py (the result is discarded)
    torch.manual_seed(0)
    start = time.perf_counter()
    train_model(build(tokenizer, attention), train_loader, epochs = 1, lr = 0.001)
    return time.perf_counter() - start


def main():
    train_loader, test_1k = get_data_loaders(batch_size = 64)                 # Protocol used by train.py / evaluate.py
    _, test_full = get_data_loaders(batch_size = 64, subset_test = None)      # Full 10,000-image MNIST test set
    history = json.loads((OUTPUT_DIR / "training_history.json").read_text())
    results = {"device": "cpu", "train_size": len(train_loader.dataset), "msa_check": msa_equivalence_check(), "models": {}}

    for tokenizer in TOKENIZERS:
        for attention in ATTENTIONS:
            name = f"{tokenizer}_{attention}"
            model = build(tokenizer, attention)
            model.load_state_dict(torch.load(OUTPUT_DIR / f"{name.lower()}_weights.pth"))
            results["models"][name] = {
                "tokens": model.tokenizer.n_patches + 1,
                "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "final_train_acc": history[name]["train_acc"][-1],
                "test_1k": evaluate(model, test_1k),
                "test_10k": evaluate(model, test_full),
                "train_epoch_sec": epoch_time(tokenizer, attention, train_loader),
            }
            print(name, results["models"][name])

    print("MSA check", results["msa_check"])
    (OUTPUT_DIR / "report_metrics.json").write_text(json.dumps(results, indent = 2))


if __name__ == "__main__":
    main()
