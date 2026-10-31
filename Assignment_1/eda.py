###############################################
#   Exploratory data analysis for CIFAR-100   #
#   Results are saved to output/eda/          #
###############################################

import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from data import get_split, load_cifar100, to_model_input, train_transform

EDA_DIR = config.OUTPUT_DIR / "eda"


def class_counts(train_set, test_set, train_idx, val_idx):
    targets = np.asarray(train_set.targets)
    return {
        "train": Counter(targets[train_idx].tolist()),
        "val": Counter(targets[val_idx].tolist()),
        "test": Counter(test_set.targets),
    }


def plot_class_counts(counts):
    _, ax = plt.subplots(figsize = (14, 4))
    classes = np.arange(config.NUM_CLASSES)
    bottom = np.zeros(config.NUM_CLASSES)
    for split, color in (("train", "#c2410c"), ("val", "#57534e"), ("test", "#a8a29e")):
        values = np.array([counts[split][c] for c in classes])
        ax.bar(classes, values, bottom = bottom, color = color, label = split, width = 0.85)
        bottom += values
    ax.set_xlabel("Chỉ số lớp (0-99)")
    ax.set_ylabel("Số ảnh")
    ax.set_title("Số ảnh mỗi lớp theo split (train / val / test)")
    ax.set_ylim(0, 700)
    ax.legend(loc = "upper center", ncols = 3, frameon = False)
    ax.set_xlim(-1, config.NUM_CLASSES)
    plt.tight_layout()
    plt.savefig(EDA_DIR / "class_counts.png", dpi = 150)
    plt.close()


def plot_samples(train_set):
    # One sample image per class: 10x10 grid
    fig, axes = plt.subplots(10, 10, figsize = (13, 14))
    targets = np.asarray(train_set.targets)
    for c, ax in enumerate(axes.flat):
        ax.imshow(train_set.data[np.flatnonzero(targets == c)[0]])
        ax.set_title(train_set.classes[c], fontsize = 7)
        ax.axis("off")
    fig.suptitle("CIFAR-100: một ảnh mẫu cho mỗi lớp (32x32)", fontsize = 13)
    plt.tight_layout()
    plt.savefig(EDA_DIR / "samples_per_class.png", dpi = 150)
    plt.close()


def plot_augmentation(train_set):
    # One original image and 5 augmented versions (upscaled to 224x224, then de-normalized for display)
    from PIL import Image
    image = Image.fromarray(train_set.data[0])
    tf = train_transform()
    mean, std = np.array(config.MEAN), np.array(config.STD)
    fig, axes = plt.subplots(1, 6, figsize = (15, 3))
    axes[0].imshow(image)
    axes[0].set_title("Gốc 32x32")
    for ax in axes[1:]:
        aug = to_model_input(tf(image).unsqueeze(0))[0].permute(1, 2, 0).numpy() * std + mean
        ax.imshow(np.clip(aug, 0, 1))
        ax.set_title("Sau augmentation")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "augmentation_examples.png", dpi = 150)
    plt.close()


def pixel_stats(train_set, train_idx):
    pixels = train_set.data[train_idx].reshape(-1, 3) / 255.0
    return {"mean": pixels.mean(0).round(4).tolist(), "std": pixels.std(0).round(4).tolist()}


def coarse_groups(train_set):
    # 20 superclasses, each made of 5 fine classes
    groups = {name: set() for name in train_set.coarse_classes}
    for fine, coarse in zip(train_set.targets, train_set.coarse_targets):
        groups[train_set.coarse_classes[coarse]].add(train_set.classes[fine])
    return {name: sorted(fines) for name, fines in groups.items()}


def main():
    EDA_DIR.mkdir(parents = True, exist_ok = True)
    train_set, test_set = load_cifar100(train = True), load_cifar100(train = False)
    train_idx, val_idx = get_split()
    counts = class_counts(train_set, test_set, train_idx, val_idx)

    plot_class_counts(counts)
    plot_samples(train_set)
    plot_augmentation(train_set)

    summary = {
        "image_shape": list(train_set.data.shape[1:]),
        "num_classes": config.NUM_CLASSES,
        "split_sizes": {k: sum(v.values()) for k, v in counts.items()},
        "images_per_class": {k: sorted(set(v.values())) for k, v in counts.items()},
        "pixel_stats_train": pixel_stats(train_set, train_idx),
        "coarse_groups": coarse_groups(train_set),
    }
    (EDA_DIR / "summary.json").write_text(json.dumps(summary, indent = 2, ensure_ascii = False))
    print(json.dumps({k: v for k, v in summary.items() if k != "coarse_groups"}, indent = 2))


if __name__ == "__main__":
    main()
