#############################################################################
#   A2.2 analysis: turn the run logs into tables, figures and numbers       #
#   for the report and the A2 page. Reads <run-dir>/*.json (+ the 100%      #
#   weights for the qualitative figure), writes figures to output/a2_2/,    #
#   LaTeX tables to report/generated/ and all numbers to summary.json.      #
#   Example: python analyze.py --run-dir output/runs                        #
#############################################################################

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

import config
from data import breed_names, eval_transform, get_rows, get_split, load_image, load_trimap, trimap_to_mask
from metrics import species_confusion

ARCH_LABEL = {"unet_r50": "U-Net + ResNet-50", "segformer_b2": "SegFormer-B2"}
FRACTIONS = sorted(config.TRAIN_FRACTIONS)

# Chart styling (same system as A1): one fixed color per model, recessive axes and grid
ARCH_COLOR = {"unet_r50": "#2a78d6", "segformer_b2": "#eb6834"}
ARCH_MARKER = {"unet_r50": "o", "segformer_b2": "s"}
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ_BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# Pixel categories of the qualitative error maps
ERROR_TYPES = [("Nền đúng", "#2b2b2b"), ("Đúng giống", "#1baf7a"), ("Sai giống, đúng loài", "#eda100"),
               ("Sai loài (chó <-> mèo)", "#d03b3b"), ("Bỏ sót thú (đoán nền)", "#2a78d6"),
               ("Nền bị đoán là thú", "#b18cf0"), ("Viền (bỏ qua)", "#ffffff")]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2, "font.size": 9, "legend.frameon": False,
    "lines.linewidth": 2,
})


# ----------------------------------------------------------------------------- loading
def load_runs(run_dir):
    runs = {}
    for exp in config.EXPERIMENTS:
        path = run_dir / f"{exp['name']}.json"
        if path.exists():
            log = json.loads(path.read_text())
            log["images"] = json.loads((run_dir / f"{exp['name']}_test_images.json").read_text())
            runs[exp["name"]] = log
    return runs


def species_of_breed():
    rows = get_rows().values()
    return {r["breed"]: r["species"] for r in rows}


# ----------------------------------------------------------------------------- number formatting (Vietnamese style)
def pct(x, digits = 1):
    return f"{x * 100:.{digits}f}".replace(".", ",")


def num(x, digits = 1):
    return f"{x:.{digits}f}".replace(".", ",")


def thousands(n):
    return f"{n:,}".replace(",", ".")


# ----------------------------------------------------------------------------- tables and numbers
def run_summary(log):
    t, v = log["test"], log["val"]
    return {"name": log["experiment"]["name"], "arch": log["experiment"]["arch"],
            "fraction": log["experiment"]["train_fraction"], "train_images": log["train_images"],
            "params": log["params"]["total"], "epochs": log["epochs"], "best_epoch": log["best_epoch"],
            "val_miou": v["miou"], "test_miou": t["miou"], "test_mdice": t["mdice"], "test_pixel_acc": t["pixel_acc"],
            "test_pet_iou": t["pet_iou"], "test_breed_acc_image": t["breed_acc_image"],
            "train_min": log["train_time_sec"] / 60, "sec_per_epoch": log["train_time_sec"] / len(log["history"]),
            "peak_mem_gb": log.get("peak_gpu_mem_mb", 0) / 1024, "img_per_sec": log["inference_img_per_sec"],
            "gpu": log.get("gpu", log["device"])}


def write_tables(rows, extras, tex_dir):
    tex_dir.mkdir(parents = True, exist_ok = True)
    main = [r for r in rows if r["fraction"] == 1.0]
    lines = []
    for r in main:
        lines.append(f"{ARCH_LABEL[r['arch']]} & {num(r['params'] / 1e6)}M & {pct(r['val_miou'])} & {pct(r['test_miou'])} & "
                     f"{pct(r['test_mdice'])} & {pct(r['test_pixel_acc'])} & {pct(r['test_pet_iou'])} & "
                     f"{pct(r['test_breed_acc_image'])} \\\\")
    (tex_dir / "main_table.tex").write_text("\n".join(lines) + "\n")

    lines = []
    for r in main:
        lines.append(f"{ARCH_LABEL[r['arch']]} & {r['best_epoch']}/{r['epochs']} & {num(r['sec_per_epoch'], 0)} & "
                     f"{num(r['train_min'], 0)} & {num(r['peak_mem_gb'])} & {num(r['img_per_sec'], 0)} \\\\")
    (tex_dir / "cost_table.tex").write_text("\n".join(lines) + "\n")

    lines = []
    for frac in sorted(config.TRAIN_FRACTIONS, reverse = True):
        cells = []
        for arch in config.ARCHS:
            r = next((x for x in rows if x["arch"] == arch and x["fraction"] == frac), None)
            cells += [pct(r["test_miou"]), pct(r["test_pet_iou"]), pct(r["test_breed_acc_image"])] if r else ["--"] * 3
        n = next((x["train_images"] for x in rows if x["fraction"] == frac), None)
        lines.append(f"{int(frac * 100)}\\% ({thousands(n) if n else '--'}) & " + " & ".join(cells) + " \\\\")
    (tex_dir / "fraction_table.tex").write_text("\n".join(lines) + "\n")

    if extras:
        lines = []
        for arch, ex in extras.items():
            res = [pct(ex["resolution"][str(s)]["miou"]) for s in sorted(map(int, ex["resolution"]))]
            lines.append(f"{ARCH_LABEL[arch]} & " + " & ".join(res) + " \\\\")
        (tex_dir / "resolution_table.tex").write_text("\n".join(lines) + "\n")
        lines = []
        for arch, ex in extras.items():
            clean = ex["resolution"][str(config.IMG_SIZE)]["miou"]
            cells = [pct(clean)] + [pct(lvl["miou"]) for name in ex["corruption"] for lvl in ex["corruption"][name]]
            lines.append(f"{ARCH_LABEL[arch]} & " + " & ".join(cells) + " \\\\")
        (tex_dir / "corruption_table.tex").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- figures
def plot_learning_curves(runs, out_dir):
    fig, axes = plt.subplots(1, 2, figsize = (10, 3.6))
    for arch in config.ARCHS:
        log = runs.get(f"{arch}_f100")
        if not log:
            continue
        ep = [h["epoch"] for h in log["history"]]
        kw = {"color": ARCH_COLOR[arch], "label": ARCH_LABEL[arch]}
        axes[0].plot(ep, [h["train_loss"] for h in log["history"]], **kw)
        axes[0].plot(ep, [h["val_loss"] for h in log["history"]], linestyle = "--", color = ARCH_COLOR[arch])
        axes[1].plot(ep, [h["val_miou"] * 100 for h in log["history"]], **kw)
        axes[1].axvline(log["best_epoch"], color = ARCH_COLOR[arch], linewidth = 0.8, linestyle = ":")
    axes[0].set(title = "Loss (liền: train, đứt: validation)", xlabel = "Epoch", ylabel = "Cross-entropy")
    axes[1].set(title = "mIoU validation (chấm: epoch tốt nhất)", xlabel = "Epoch", ylabel = "mIoU (%)")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curves.png", dpi = 200)
    plt.close(fig)


def plot_data_efficiency(rows, out_dir):
    metrics = [("test_miou", "mIoU 38 lớp"), ("test_pet_iou", "IoU thú / nền"), ("test_breed_acc_image", "Đúng giống (theo ảnh)")]
    fig, axes = plt.subplots(1, 3, figsize = (12, 3.6))
    for ax, (key, title) in zip(axes, metrics):
        for arch in config.ARCHS:
            pts = sorted((r["train_images"], r[key] * 100) for r in rows if r["arch"] == arch)
            if not pts:
                continue
            x, y = zip(*pts)
            ax.plot(x, y, marker = ARCH_MARKER[arch], markersize = 6, color = ARCH_COLOR[arch], label = ARCH_LABEL[arch])
            ax.annotate(num(y[-1]), (x[-1], y[-1]), textcoords = "offset points", xytext = (6, -3), color = INK_2, fontsize = 8)
        ax.set_xscale("log")
        xs = sorted({r["train_images"] for r in rows})
        ax.set_xticks(xs, [thousands(n) for n in xs])
        ax.minorticks_off()
        ax.set(title = title, xlabel = "Số ảnh train (thang log)", ylabel = "%")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "data_efficiency.png", dpi = 200)
    plt.close(fig)


def plot_per_breed(runs, names, species, out_dir):
    a, b = runs.get("unet_r50_f100"), runs.get("segformer_b2_f100")
    if not (a and b):
        return None
    iou_a = np.array(a["test"]["per_class_iou"][1:]) * 100
    iou_b = np.array(b["test"]["per_class_iou"][1:]) * 100
    order = np.argsort(iou_b - iou_a)
    fig, ax = plt.subplots(figsize = (7, 9))
    y = np.arange(len(order))
    ax.hlines(y, iou_a[order], iou_b[order], color = AXIS, linewidth = 1.5)
    ax.scatter(iou_a[order], y, color = ARCH_COLOR["unet_r50"], s = 28, zorder = 3, label = ARCH_LABEL["unet_r50"])
    ax.scatter(iou_b[order], y, color = ARCH_COLOR["segformer_b2"], s = 28, marker = "s", zorder = 3,
               label = ARCH_LABEL["segformer_b2"])
    ax.set_yticks(y, [f"{names[i].replace('_', ' ')} ({'mèo' if species[i + 1] == 1 else 'chó'})" for i in order], fontsize = 7)
    ax.set(xlabel = "IoU trên tập test (%)", title = "IoU từng giống (xếp theo chênh lệch SegFormer - U-Net)")
    ax.grid(axis = "y", visible = False)
    ax.legend(loc = "lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "per_breed_iou.png", dpi = 200)
    plt.close(fig)
    diff = iou_b - iou_a
    return {"segformer_better": int((diff > 0).sum()), "unet_better": int((diff < 0).sum()),
            "largest_gain_segformer": [(names[i], float(diff[i])) for i in np.argsort(diff)[::-1][:3]],
            "largest_gain_unet": [(names[i], float(-diff[i])) for i in np.argsort(diff)[:3]],
            "worst_unet": [(names[i], float(iou_a[i])) for i in np.argsort(iou_a)[:3]],
            "worst_segformer": [(names[i], float(iou_b[i])) for i in np.argsort(iou_b)[:3]]}


def pet_pixel_breakdown(cm, species):
    # Share of ground-truth pet pixels by outcome: correct breed / wrong breed same species / wrong species / background
    cm = np.asarray(cm, dtype = np.float64)
    out = {"correct": 0.0, "wrong_breed_same_species": 0.0, "wrong_species": 0.0, "background": 0.0}
    for t in range(1, config.NUM_CLASSES):
        for p in range(config.NUM_CLASSES):
            key = ("background" if p == 0 else "correct" if p == t else
                   "wrong_breed_same_species" if species[p] == species[t] else "wrong_species")
            out[key] += cm[t, p]
    total = sum(out.values())
    return {k: v / total for k, v in out.items()}


def plot_species_confusion(runs, species, out_dir):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ_BLUE)
    present = [a for a in config.ARCHS if f"{a}_f100" in runs]
    if not present:
        return {}
    fig, axes = plt.subplots(1, len(present), figsize = (4.2 * len(present), 3.6), squeeze = False)
    labels = ["Nền", "Mèo", "Chó"]
    out = {}
    for ax, arch in zip(axes[0], present):
        cm = runs[f"{arch}_f100"]["test"]["confusion"]
        sc = species_confusion(cm, species)
        norm = sc / np.maximum(sc.sum(1, keepdims = True), 1)
        ax.imshow(norm, cmap = cmap, vmin = 0, vmax = 1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, pct(norm[i, j]), ha = "center", va = "center", fontsize = 9,
                        color = "white" if norm[i, j] > 0.5 else INK)
        ax.set_xticks(range(3), labels)
        ax.set_yticks(range(3), labels)
        ax.set(xlabel = "Dự đoán", ylabel = "Nhãn thật", title = ARCH_LABEL[arch])
        ax.grid(False)
        out[arch] = {"species_confusion_row_norm": norm.tolist(), "pet_pixels": pet_pixel_breakdown(cm, species)}
    fig.suptitle("Ma trận nhầm lẫn theo điểm ảnh, gộp nền / mèo / chó (chuẩn hoá theo hàng)", fontsize = 10)
    fig.tight_layout()
    fig.savefig(out_dir / "species_confusion.png", dpi = 200)
    plt.close(fig)
    return out


def plot_extras(extras, out_dir):
    if not extras:
        return
    fig, axes = plt.subplots(1, 4, figsize = (14, 3.4))
    for arch, ex in extras.items():
        kw = {"color": ARCH_COLOR[arch], "marker": ARCH_MARKER[arch], "markersize": 6, "label": ARCH_LABEL[arch]}
        sizes = sorted(map(int, ex["resolution"]))
        axes[0].plot(sizes, [ex["resolution"][str(s)]["miou"] * 100 for s in sizes], **kw)
        clean = ex["resolution"][str(config.IMG_SIZE)]["miou"] * 100
        for ax, name in zip(axes[1:], ex["corruption"]):
            ax.plot([0, 1, 2, 3], [clean] + [lvl["miou"] * 100 for lvl in ex["corruption"][name]], **kw)
    axes[0].axvline(config.IMG_SIZE, color = MUTED, linewidth = 0.8, linestyle = ":")
    axes[0].set(title = "Đổi độ phân giải khi test", xlabel = "Cạnh ảnh test (px), train ở 384", ylabel = "mIoU (%)")
    axes[0].set_xticks(sizes)
    titles = {"gaussian_noise": "Nhiễu Gauss", "gaussian_blur": "Làm mờ Gauss", "jpeg": "Nén JPEG"}
    for ax, name in zip(axes[1:], next(iter(extras.values()))["corruption"]):
        ax.set(title = titles.get(name, name), xlabel = "Mức độ (0 = ảnh gốc)", ylabel = "mIoU (%)")
        ax.set_xticks([0, 1, 2, 3])
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "robustness.png", dpi = 200)
    plt.close(fig)


def plot_corruption_examples(name, out_dir):
    # What the corruption levels of eval_extra.py look like on one test image (no model involved)
    from eval_extra import CORRUPTIONS
    from torchvision.transforms import v2
    image = v2.functional.pil_to_tensor(load_image(name))
    image = v2.functional.resize(image, [config.IMG_SIZE, config.IMG_SIZE], antialias = True)
    titles = {"gaussian_noise": "Nhiễu Gauss", "gaussian_blur": "Làm mờ Gauss", "jpeg": "Nén JPEG"}
    fig, axes = plt.subplots(len(CORRUPTIONS), 4, figsize = (9, 2.4 * len(CORRUPTIONS)))
    for row, (cname, levels) in zip(axes, CORRUPTIONS.items()):
        row[0].imshow(image.permute(1, 2, 0))
        row[0].set_title("Ảnh gốc" if cname == "gaussian_noise" else "", fontsize = 8)
        row[0].set_ylabel(titles[cname])
        for k, (ax, c) in enumerate(zip(row[1:], levels), start = 1):
            x = c(image) if isinstance(c, v2.JPEG) else (c(image.float() / 255) * 255).clamp(0, 255).byte()
            ax.imshow(x.permute(1, 2, 0))
            ax.set_title(f"Mức {k}", fontsize = 8)
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_dir / "corruption_examples.png", dpi = 150)
    plt.close(fig)


# ----------------------------------------------------------------------------- qualitative examples (needs weights)
def pick_examples(runs, k = 2):
    # Test images where SegFormer is clearly better, where U-Net is clearly better, and where both fail
    a = {r["name"]: r for r in runs["unet_r50_f100"]["images"]}
    b = {r["name"]: r for r in runs["segformer_b2_f100"]["images"]}
    names = sorted(a)
    score = {n: (a[n]["pet_iou"] + (a[n]["pred_breed"] == a[n]["true_breed"]),
                 b[n]["pet_iou"] + (b[n]["pred_breed"] == b[n]["true_breed"])) for n in names}
    groups = [("SegFormer tốt hơn", lambda n: score[n][0] - score[n][1]),
              ("U-Net tốt hơn", lambda n: score[n][1] - score[n][0]),
              ("Cả hai sai", lambda n: a[n]["pet_iou"] + b[n]["pet_iou"])]
    picked, used = [], set()
    for group, key in groups:
        chosen = [n for n in sorted(names, key = key) if n not in used][:k]     # An image appears in one group only
        used.update(chosen)
        picked += [(group, n) for n in chosen]
    return picked


def error_map(pred, mask, species):
    out = np.zeros(mask.shape, dtype = np.uint8)                       # 0 = correct background
    pet, bg, border = (mask > 0) & (mask != config.IGNORE_INDEX), mask == 0, mask == config.IGNORE_INDEX
    sp = np.vectorize(lambda c: species.get(int(c), 0))
    out[pet & (pred == mask)] = 1
    wrong = pet & (pred > 0) & (pred != mask)
    same = sp(pred) == sp(np.where(pet, mask, 0))
    out[wrong & same] = 2
    out[wrong & ~same] = 3
    out[pet & (pred == 0)] = 4
    out[bg & (pred > 0)] = 5
    out[border] = 6
    return out


def plot_examples(runs, run_dir, names, species, out_dir, device):
    from models import build_model
    weights = {arch: run_dir / f"{arch}_f100_best.pth" for arch in config.ARCHS}
    if not all(w.exists() for w in weights.values()) or not all(f"{a}_f100" in runs for a in config.ARCHS):
        print("Bỏ qua hình ví dụ: thiếu trọng số *_f100_best.pth")
        return []
    rows = get_rows()
    examples = pick_examples(runs)
    preds = {}
    for arch, path in weights.items():
        model = build_model(arch, pretrained = False)
        model.load_state_dict(torch.load(path, map_location = "cpu"))
        model.to(device).eval()
        with torch.no_grad():
            for _, n in examples:
                image, _ = eval_transform()(*sample(n, rows))
                preds[(arch, n)] = model(image[None].to(device)).argmax(1)[0].cpu().numpy()
        del model
    cmap = ListedColormap([c for _, c in ERROR_TYPES])
    fig, axes = plt.subplots(len(examples), 4, figsize = (10, 2.6 * len(examples)))
    out = []
    for row, (group, n) in zip(axes, examples):
        image, mask = eval_transform()(*sample(n, rows))
        mask = mask.numpy()
        shown = (image * torch.tensor(config.STD)[:, None, None] + torch.tensor(config.MEAN)[:, None, None]).clamp(0, 1)
        row[0].imshow(shown.permute(1, 2, 0))
        row[0].set_title(f"{group}: {n}", fontsize = 8)
        row[1].imshow(np.where(mask == config.IGNORE_INDEX, 6, np.where(mask > 0, 1, 0)), cmap = cmap, vmin = 0, vmax = 6,
                      interpolation = "nearest")
        row[1].set_title(f"Nhãn: {names[rows[n]['breed'] - 1]}", fontsize = 8)
        item = {"group": group, "name": n}
        for ax, arch in zip(row[2:], config.ARCHS):
            p = preds[(arch, n)]
            ax.imshow(error_map(p, mask, species), cmap = cmap, vmin = 0, vmax = 6, interpolation = "nearest")
            pet = p[p > 0]
            guess = names[np.bincount(pet, minlength = config.NUM_CLASSES).argmax() - 1] if pet.size else "(không thấy thú)"
            ax.set_title(f"{ARCH_LABEL[arch]}: {guess}", fontsize = 8)
            item[arch] = guess
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        out.append(item)
    fig.legend(handles = [Patch(color = c, label = l) for l, c in ERROR_TYPES[:6]], loc = "lower center", ncols = 3,
               fontsize = 8, bbox_to_anchor = (0.5, 0))
    fig.tight_layout(rect = (0, 0.03, 1, 1))
    fig.savefig(out_dir / "examples.png", dpi = 130)
    plt.close(fig)
    return out


def sample(name, rows):
    from torchvision import tv_tensors
    from torchvision.transforms import v2
    image = tv_tensors.Image(v2.functional.pil_to_tensor(load_image(name)))
    mask = tv_tensors.Mask(torch.from_numpy(trimap_to_mask(load_trimap(name), rows[name]["breed"])))
    return image, mask


# ----------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description = "Phân tích kết quả A2.2")
    parser.add_argument("--run-dir", default = str(config.OUTPUT_DIR / "runs"))
    parser.add_argument("--out-dir", default = str(config.OUTPUT_DIR / "a2_2"))
    parser.add_argument("--tex-dir", default = str(config.ROOT / "report" / "generated"))
    parser.add_argument("--device", default = "cpu")
    args = parser.parse_args()
    run_dir, out_dir, tex_dir = Path(args.run_dir), Path(args.out_dir), Path(args.tex_dir)
    out_dir.mkdir(parents = True, exist_ok = True)

    runs = load_runs(run_dir)
    print(f"Đọc {len(runs)} lần chạy: {', '.join(runs)}")
    rows = [run_summary(log) for log in runs.values()]
    extras = {a: json.loads((run_dir / f"extra_{a}.json").read_text()) for a in config.ARCHS
              if (run_dir / f"extra_{a}.json").exists()}
    names, species = breed_names(), species_of_breed()

    write_tables(rows, extras, tex_dir)
    plot_learning_curves(runs, out_dir)
    plot_data_efficiency(rows, out_dir)
    summary = {"runs": rows, "extras": extras,
               "per_breed": plot_per_breed(runs, names, species, out_dir),
               "species": plot_species_confusion(runs, species, out_dir)}
    plot_extras(extras, out_dir)
    plot_corruption_examples(get_split()[2][0], out_dir)
    summary["examples"] = plot_examples(runs, run_dir, names, species, out_dir, torch.device(args.device))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent = 1, ensure_ascii = False))

    for r in sorted(rows, key = lambda r: (r["arch"], -r["fraction"])):
        print(f"{r['name']:<20} test mIoU {pct(r['test_miou'])} | pet IoU {pct(r['test_pet_iou'])} | "
              f"giống/ảnh {pct(r['test_breed_acc_image'])} | {num(r['train_min'])} phút")


if __name__ == "__main__":
    main()
