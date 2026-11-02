#####################################################
#   Exploratory data analysis for Oxford-IIIT Pet   #
#   Results are saved to output/eda/                #
#####################################################

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import config
from data import (breed_names, eval_transform, get_rows, get_split, load_image, load_trimap,
                  train_transform, trimap_to_mask)

EDA_DIR = config.OUTPUT_DIR / "eda"
SPLIT_COLORS = {"train": "#c2410c", "val": "#57534e", "test": "#a8a29e"}


def scan_images(rows):
    # One pass over every image: size, colour mode and the pet / background / border pixel fractions of its trimap
    stats = {}
    for name in rows:
        with Image.open(config.PET_DIR / "images" / f"{name}.jpg") as img:
            size, mode = img.size, img.mode
        trimap = load_trimap(name)
        total = trimap.size
        stats[name] = {
            "width": size[0], "height": size[1], "mode": mode, "trimap_shape_ok": trimap.shape == (size[1], size[0]),
            "trimap_values": sorted(np.unique(trimap).tolist()),
            "pet": float((trimap == config.TRIMAP_PET).sum() / total),
            "background": float((trimap == config.TRIMAP_BACKGROUND).sum() / total),
            "border": float((trimap == config.TRIMAP_BORDER).sum() / total),
        }
    return stats


def plot_breed_counts(rows, splits, names):
    _, ax = plt.subplots(figsize = (14, 4.8))
    x = np.arange(config.NUM_BREEDS)
    bottom = np.zeros(config.NUM_BREEDS)
    for split, members in splits.items():
        values = np.array([sum(rows[n]["breed"] == b + 1 for n in members) for b in x])
        ax.bar(x, values, bottom = bottom, color = SPLIT_COLORS[split], label = f"{split} ({len(members)})", width = 0.85)
        bottom += values
    species = [next(r["species"] for r in rows.values() if r["breed"] == b + 1) for b in x]
    ax.set_xticks(x, names, rotation = 75, ha = "right", fontsize = 8)
    for tick, sp in zip(ax.get_xticklabels(), species):
        tick.set_fontweight("bold" if sp == 1 else "normal")
    ax.set_ylabel("Số ảnh")
    ax.set_title("Số ảnh mỗi giống theo split (tên in đậm = giống mèo)")
    ax.legend(loc = "upper center", ncols = 3, frameon = False, bbox_to_anchor = (0.5, 1.0))
    ax.set_ylim(0, 240)
    ax.set_xlim(-1, config.NUM_BREEDS)
    plt.tight_layout()
    plt.savefig(EDA_DIR / "breed_counts.png", dpi = 150)
    plt.close()


def plot_image_sizes(stats):
    widths = np.array([s["width"] for s in stats.values()])
    heights = np.array([s["height"] for s in stats.values()])
    fig, axes = plt.subplots(1, 2, figsize = (12, 4))
    axes[0].scatter(widths, heights, s = 4, alpha = 0.3, color = "#c2410c")
    axes[0].set_xlabel("Chiều rộng (px)")
    axes[0].set_ylabel("Chiều cao (px)")
    axes[0].set_title("Kích thước ảnh gốc")
    axes[1].hist(widths / heights, bins = 60, color = "#57534e")
    axes[1].axvline(1.0, color = "#c2410c", linestyle = "--", linewidth = 1)
    axes[1].set_xlabel("Tỉ lệ rộng / cao")
    axes[1].set_ylabel("Số ảnh")
    axes[1].set_title("Tỉ lệ khung hình")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "image_sizes.png", dpi = 150)
    plt.close()


def plot_pixel_fractions(stats):
    fig, axes = plt.subplots(1, 3, figsize = (14, 3.8), sharey = True)
    for ax, key, title, color in zip(axes, ("pet", "background", "border"),
                                     ("Thú cưng (nhãn = giống)", "Nền (lớp 0)", "Viền (bỏ qua khi tính loss/metric)"),
                                     ("#c2410c", "#57534e", "#a8a29e")):
        values = np.array([s[key] for s in stats.values()]) * 100
        ax.hist(values, bins = 50, range = (0, 100), color = color)
        ax.axvline(np.median(values), color = "black", linestyle = "--", linewidth = 1)
        ax.set_title(f"{title}\ntrung vị {np.median(values):.1f}%", fontsize = 10)
        ax.set_xlabel("% điểm ảnh trong ảnh")
    axes[0].set_ylabel("Số ảnh")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "pixel_fractions.png", dpi = 150)
    plt.close()


def class_pixel_share(stats, rows, members):
    # Share of all labelled (non-border) pixels per class over a split: background vs each breed
    share = np.zeros(config.NUM_CLASSES)
    for name in members:
        s = stats[name]
        pixels = s["width"] * s["height"]
        share[0] += s["background"] * pixels
        share[rows[name]["breed"]] += s["pet"] * pixels
    return share / share.sum()


def plot_class_share(share, names):
    order = np.argsort(share[1:])[::-1] + 1
    _, ax = plt.subplots(figsize = (14, 4.4))
    ax.bar(0, share[0] * 100, color = "#57534e", width = 0.85)
    ax.bar(np.arange(1, config.NUM_CLASSES), share[order] * 100, color = "#c2410c", width = 0.85)
    ax.set_xticks(np.arange(config.NUM_CLASSES), ["background"] + [names[i - 1] for i in order],
                  rotation = 75, ha = "right", fontsize = 8)
    ax.set_yscale("log")
    ax.set_ylabel("% điểm ảnh có nhãn (thang log)")
    ax.set_title("Tỉ lệ điểm ảnh mỗi lớp trên tập train: nền chiếm áp đảo, mỗi giống chỉ khoảng 1%")
    ax.set_xlim(-1, config.NUM_CLASSES)
    plt.tight_layout()
    plt.savefig(EDA_DIR / "class_pixel_share.png", dpi = 150)
    plt.close()


def colorize(mask):
    # Background black, pet pixels coloured by breed (tab20b + tab20c), ignored border white
    palette = np.vstack([[0, 0, 0], (np.vstack([plt.cm.tab20b.colors, plt.cm.tab20c.colors])[:config.NUM_BREEDS] * 255)])
    rgb = palette[np.clip(mask, 0, config.NUM_BREEDS)].astype(np.uint8)
    rgb[mask == config.IGNORE_INDEX] = 255
    return rgb


def plot_samples(rows, members, names):
    # 6 cats + 6 dogs from train: image on top, 38-class mask below
    rng = np.random.default_rng(config.SEED)
    picks = []
    for species in (1, 2):
        pool = [n for n in members if rows[n]["species"] == species]
        picks += [pool[i] for i in rng.choice(len(pool), 6, replace = False)]
    fig, axes = plt.subplots(4, 6, figsize = (15, 10.5))
    for k, name in enumerate(picks):
        r, c = 2 * (k // 6), k % 6
        breed = rows[name]["breed"]
        axes[r, c].imshow(load_image(name))
        axes[r, c].set_title(f"{names[breed - 1]} (lớp {breed})", fontsize = 9)
        axes[r + 1, c].imshow(colorize(trimap_to_mask(load_trimap(name), breed)))
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle("Ảnh mẫu và mặt nạ 38 lớp (đen = nền, màu = giống, trắng = viền bị bỏ qua)", fontsize = 13)
    plt.tight_layout()
    plt.savefig(EDA_DIR / "samples_with_masks.png", dpi = 120)
    plt.close()


def plot_augmentation(rows, members):
    from data import PetSegmentation
    name = sorted(members)[0]
    torch.manual_seed(config.SEED)
    aug_set = PetSegmentation([rows[name]], train_transform())
    eval_set = PetSegmentation([rows[name]], eval_transform())
    mean, std = np.array(config.MEAN), np.array(config.STD)
    pairs = [eval_set[0]] + [aug_set[0] for _ in range(4)]
    fig, axes = plt.subplots(2, 5, figsize = (15, 6.4))
    for c, (image, mask) in enumerate(pairs):
        axes[0, c].imshow(np.clip(image.permute(1, 2, 0).numpy() * std + mean, 0, 1))
        axes[1, c].imshow(colorize(mask.numpy()))
        axes[0, c].set_title("Val/test: resize 384x384" if c == 0 else f"Train: augmentation #{c}", fontsize = 10)
    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "augmentation_examples.png", dpi = 120)
    plt.close()


def pixel_stats(members, sample = 500):
    # Per-channel mean/std over a fixed random sample of train images (at full resolution)
    rng = np.random.default_rng(config.SEED)
    members = sorted(members)
    total, sq, count = np.zeros(3), np.zeros(3), 0
    for i in rng.choice(len(members), min(sample, len(members)), replace = False):
        x = np.asarray(load_image(members[i]), dtype = np.float64).reshape(-1, 3) / 255.0
        total += x.sum(0)
        sq += (x ** 2).sum(0)
        count += len(x)
    mean = total / count
    return {"mean": mean.round(4).tolist(), "std": np.sqrt(sq / count - mean ** 2).round(4).tolist(), "images": int(sample)}


def summarize(values):
    values = np.asarray(values, dtype = float)
    return {"min": float(values.min()), "median": float(np.median(values)), "mean": float(values.mean().round(4)),
            "max": float(values.max())}


def main():
    EDA_DIR.mkdir(parents = True, exist_ok = True)
    rows = get_rows()
    train, val, test = get_split()
    splits = {"train": train, "val": val, "test": test}
    names = breed_names()
    stats = scan_images(rows)

    plot_breed_counts(rows, splits, names)
    plot_image_sizes(stats)
    plot_pixel_fractions(stats)
    share = class_pixel_share(stats, rows, train)
    plot_class_share(share, names)
    plot_samples(rows, train, names)
    plot_augmentation(rows, train)

    per_breed = {split: [sum(rows[n]["breed"] == b for n in members) for b in range(1, config.NUM_BREEDS + 1)]
                 for split, members in splits.items()}
    summary = {
        "num_images": len(rows),
        "num_classes": config.NUM_CLASSES,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "species_counts": {split: {"cat": sum(rows[n]["species"] == 1 for n in m), "dog": sum(rows[n]["species"] == 2 for n in m)}
                           for split, m in splits.items()},
        "images_per_breed": {split: {"min": min(c), "max": max(c)} for split, c in per_breed.items()},
        "width": summarize([s["width"] for s in stats.values()]),
        "height": summarize([s["height"] for s in stats.values()]),
        "image_modes": {m: sum(s["mode"] == m for s in stats.values()) for m in sorted({s["mode"] for s in stats.values()})},
        "trimap_size_mismatch": sum(not s["trimap_shape_ok"] for s in stats.values()),
        "trimap_values": sorted({v for s in stats.values() for v in s["trimap_values"]}),
        "excluded_empty_trimap": json.loads(config.SPLIT_FILE.read_text())["excluded_empty_trimap"],
        "pixel_fraction_pet": summarize([s["pet"] for s in stats.values()]),
        "pixel_fraction_background": summarize([s["background"] for s in stats.values()]),
        "pixel_fraction_border": summarize([s["border"] for s in stats.values()]),
        "train_pixel_share": {"background": float(share[0].round(4)), "breed_min": float(share[1:].min().round(4)),
                              "breed_max": float(share[1:].max().round(4))},
        "pixel_stats_train": pixel_stats(train),
        "breeds": {"cat": [names[b - 1] for b in range(1, config.NUM_BREEDS + 1)
                           if next(r["species"] for r in rows.values() if r["breed"] == b) == 1],
                   "dog": [names[b - 1] for b in range(1, config.NUM_BREEDS + 1)
                           if next(r["species"] for r in rows.values() if r["breed"] == b) == 2]},
    }
    (EDA_DIR / "summary.json").write_text(json.dumps(summary, indent = 2, ensure_ascii = False))
    print(json.dumps({k: v for k, v in summary.items() if k != "breeds"}, indent = 2, ensure_ascii = False))


if __name__ == "__main__":
    main()
