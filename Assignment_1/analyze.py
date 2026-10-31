#############################################################################
#   A1.2 analysis: turn the 8 run logs into tables, figures and numbers     #
#   for the report. Reads output/runs/*.json, writes figures to             #
#   output/a1_2/ and LaTeX tables/macros to report/generated/.              #
#   Example: python analyze.py --run-dir output/runs                        #
#############################################################################

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

import config
from data import load_cifar100

ARCH_LABEL = {"resnet18": "ResNet-18", "deit_tiny": "DeiT-Tiny"}
ARCH_MACRO = {"resnet18": "ResNet", "deit_tiny": "DeiT"}
MODES = ["scratch", "head", "partial", "full"]
MODE_LABEL = {"scratch": "Học từ đầu", "head": "Chỉ train head", "partial": "Đóng băng một phần", "full": "Fine-tune toàn bộ"}

# Chart styling: one fixed color per training mode (validated categorical order), recessive axes and grid
MODE_COLOR = {"scratch": "#2a78d6", "head": "#eb6834", "partial": "#1baf7a", "full": "#eda100"}
ARCH_MARKER = {"resnet18": "o", "deit_tiny": "s"}
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
NEUTRAL_BAR = "#52514e"
SEQ_BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

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
            log["preds"] = np.array(json.loads((run_dir / f"{exp['name']}_test_preds.json").read_text()))
            runs[exp["name"]] = log
    return runs


def mode_of(exp):
    return "scratch" if not exp["pretrained"] else ("full" if exp["freeze"] == "none" else exp["freeze"])


def best_per_arch(runs):
    # Chosen by validation accuracy, as in the protocol (the test set is never used for selection)
    best = {}
    for name, log in runs.items():
        arch = log["experiment"]["arch"]
        if arch not in best or log["best_val_acc"] > runs[best[arch]]["best_val_acc"]:
            best[arch] = name
    return best


# ----------------------------------------------------------------------------- number formatting (Vietnamese style)
def pct(x, digits = 2):
    return f"{x * 100:.{digits}f}".replace(".", ",")


def num(x, digits = 1):
    return f"{x:.{digits}f}".replace(".", ",")


def thousands(n):
    return f"{n:,}".replace(",", ".")


def tex_name(name):
    return r"\code{" + name.replace("_", r"\_") + "}"


# ----------------------------------------------------------------------------- tables
def summary_rows(runs):
    rows = []
    for exp in config.EXPERIMENTS:
        log = runs.get(exp["name"])
        if log is None:
            rows.append({"name": exp["name"], "arch": exp["arch"], "mode": mode_of(exp), "missing": True})
            continue
        epochs = len(log["history"])
        rows.append({
            "name": exp["name"], "arch": exp["arch"], "mode": mode_of(exp), "missing": False,
            "trainable": log["params_trainable"], "trainable_ratio": log["params_trainable"] / log["params_total"],
            "epochs": epochs, "best_epoch": log["best_epoch"], "val_acc": log["best_val_acc"],
            "test_acc": log["test_acc"], "test_f1": log["test_macro_f1"],
            "final_train_acc": log["history"][-1]["train_acc"],
            "train_min": log["train_time_sec"] / 60, "sec_per_epoch": log["train_time_sec"] / epochs,
            "peak_gb": log.get("peak_gpu_mem_mb", float("nan")) / 1024, "img_per_sec": log["inference_img_per_sec"],
            "gpu": log.get("gpu", log["device"]),
        })
    return rows


def write_tables(rows, tex_dir, out_dir):
    q, c = [], []
    for i, r in enumerate(rows):
        sep = r"\midrule" + "\n" if i == 4 else ""
        if r["missing"]:
            q.append(sep + tex_name(r["name"]) + r" & \multicolumn{6}{c}{\todo{chưa chạy}} \\")
            c.append(sep + tex_name(r["name"]) + r" & \multicolumn{5}{c}{\todo{chưa chạy}} \\")
            continue
        q.append(sep + f"{tex_name(r['name'])} & {MODE_LABEL[r['mode']]} & {r['best_epoch']}/{r['epochs']} & "
                 f"{pct(r['final_train_acc'])} & {pct(r['val_acc'])} & {pct(r['test_acc'])} & {pct(r['test_f1'])} \\\\")
        c.append(sep + f"{tex_name(r['name'])} & {thousands(r['trainable'])} ({pct(r['trainable_ratio'], 1)}\\%) & "
                 f"{num(r['train_min'])} & {num(r['sec_per_epoch'], 0)} & {num(r['peak_gb'], 2)} & {thousands(round(r['img_per_sec']))} \\\\")
    (tex_dir / "table_quality.tex").write_text("\n".join(q) + "\n")
    (tex_dir / "table_cost.tex").write_text("\n".join(c) + "\n")

    # Same numbers as Markdown/JSON for the README and landing page
    md = ["| Cấu hình | Chế độ | Trainable | Val acc | Test acc | Macro-F1 | Train (phút) | GPU (GB) |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if not r["missing"]:
            md.append(f"| `{r['name']}` | {MODE_LABEL[r['mode']]} | {thousands(r['trainable'])} | {pct(r['val_acc'])} | "
                      f"{pct(r['test_acc'])} | {pct(r['test_f1'])} | {num(r['train_min'])} | {num(r['peak_gb'], 2)} |")
    (out_dir / "results_table.md").write_text("\n".join(md) + "\n")
    (out_dir / "results_table.json").write_text(json.dumps(rows, indent = 2, ensure_ascii = False))


def write_macros(rows, extra, tex_dir):
    # One macro per number used in the report text, e.g. \ResNetFullTest, so the prose always matches the logs
    lines = []
    for r in rows:
        if r["missing"]:
            continue
        key = ARCH_MACRO[r["arch"]] + r["mode"].capitalize()
        for suffix, value in (("Test", pct(r["test_acc"])), ("Fone", pct(r["test_f1"])), ("Val", pct(r["val_acc"])),
                              ("TrainAcc", pct(r["final_train_acc"])), ("Min", num(r["train_min"])),
                              ("SecEpoch", num(r["sec_per_epoch"], 0)), ("Mem", num(r["peak_gb"], 2)),
                              ("BestEpoch", str(r["best_epoch"]))):
            lines.append(f"\\newcommand{{\\{key}{suffix}}}{{{value}}}")
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in extra.items()]
    (tex_dir / "macros.tex").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- figures
def spread(values, min_gap):
    # Move apart only the labels that would overlap: overlapping neighbours form a group that is laid out
    # min_gap apart and centered on the group's points; groups merge until no two labels are closer than min_gap
    order = np.argsort(values)
    groups = [[i] for i in order]

    def layout(group):
        center = np.mean([values[i] for i in group])
        return center + (np.arange(len(group)) - (len(group) - 1) / 2) * min_gap

    merged = True
    while merged:
        merged = False
        for k in range(len(groups) - 1):
            if layout(groups[k + 1])[0] - layout(groups[k])[-1] < min_gap:
                groups[k:k + 2] = [groups[k] + groups[k + 1]]
                merged = True
                break
    out = np.zeros(len(values))
    for group in groups:
        out[group] = layout(group)
    return out


def plot_accuracy_by_mode(rows, out_dir):
    # Grouped bars: architectures on the x-axis, one bar per training mode, value printed above each bar
    fig, ax = plt.subplots(figsize = (7.2, 3.4))
    width, x = 0.19, np.arange(2)
    for j, mode in enumerate(MODES):
        vals = [next((r["test_acc"] * 100 for r in rows if r["arch"] == a and r["mode"] == mode and not r["missing"]), np.nan)
                for a in ARCH_LABEL]
        pos = x + (j - 1.5) * (width + 0.01)
        ax.bar(pos, vals, width, color = MODE_COLOR[mode], label = MODE_LABEL[mode], edgecolor = SURFACE, linewidth = 1)
        for p, v in zip(pos, vals):
            if not np.isnan(v):
                ax.text(p, v + 1, f"{v:.1f}".replace(".", ","), ha = "center", va = "bottom", fontsize = 8, color = INK)
    ax.set_xticks(x, list(ARCH_LABEL.values()))
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis = "x", visible = False)
    ax.legend(ncol = 4, loc = "upper center", bbox_to_anchor = (0.5, 1.13), fontsize = 8)
    fig.tight_layout()
    fig.savefig(out_dir / "test_acc_by_mode.png", dpi = 200)
    plt.close(fig)


def plot_learning_curves(runs, out_dir):
    # Small multiples: columns = architecture, rows = validation / train accuracy; color = training mode
    fig, axes = plt.subplots(2, 2, figsize = (7.6, 5.4), sharey = "row")
    for col, arch in enumerate(ARCH_LABEL):
        for row, key in enumerate(("val_acc", "train_acc")):
            ax, ends = axes[row, col], []
            for mode in MODES:
                log = runs.get(f"{arch}_{mode}")
                if log is None:
                    continue
                ep = [h["epoch"] for h in log["history"]]
                vals = [h[key] * 100 for h in log["history"]]
                ax.plot(ep, vals, color = MODE_COLOR[mode], label = MODE_LABEL[mode])
                ends.append((ep[-1], vals[-1]))
            # End-of-line labels, spread vertically when two curves finish close together
            for (x_end, y_end), y_lab in zip(ends, spread([e[1] for e in ends], 4.5)):
                ax.annotate(f"{y_end:.0f}", (x_end, y_end), xytext = (x_end + 0.6, y_lab), textcoords = "data",
                            va = "center", fontsize = 7, color = INK_2)
            ax.set_xlim(0, 33)
        axes[0, col].set_title(ARCH_LABEL[arch])
        axes[1, col].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Validation accuracy (%)")
    axes[1, 0].set_ylabel("Train accuracy (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol = 4, loc = "upper center", fontsize = 8)
    fig.tight_layout(rect = (0, 0, 1, 0.95))
    fig.savefig(out_dir / "learning_curves.png", dpi = 200)
    plt.close(fig)


def plot_cost(rows, out_dir):
    # Accuracy against two cost measures, one panel each (same y-axis, no dual axis)
    fig, axes = plt.subplots(1, 2, figsize = (7.6, 3.2), sharey = True)
    done = [r for r in rows if not r["missing"]]
    y = [r["test_acc"] * 100 for r in done]
    for ax, key in ((axes[0], "train_min"), (axes[1], "trainable")):
        xs = [r[key] for r in done]
        for r, xval, yval in zip(done, xs, y):
            ax.scatter(xval, yval, s = 46, color = MODE_COLOR[r["mode"]], marker = ARCH_MARKER[r["arch"]],
                       edgecolor = SURFACE, linewidth = 1.5, zorder = 3)
        # Labels of points that share almost the same x are spread vertically to avoid overlaps
        x_key = np.log10(xs) if key == "trainable" else np.array(xs) / max(xs)
        y_lab = np.array(y, dtype = float)
        for cluster in {round(v * 8) for v in x_key}:
            idx = [i for i, v in enumerate(x_key) if round(v * 8) == cluster]
            y_lab[idx] = spread([y[i] for i in idx], (max(y) - min(y)) * 0.055 + 0.3)
        for r, xval, yval, yl in zip(done, xs, y, y_lab):
            # x: 6 points right of the marker, y: the spread label position in data units
            ax.annotate(f"{ARCH_LABEL[r['arch']].split('-')[0]} {r['mode']}", (xval, yval), xytext = (6, yl),
                        textcoords = ("offset points", "data"), fontsize = 7, color = INK_2, va = "center",
                        arrowprops = dict(arrowstyle = "-", color = AXIS, lw = 0.6, shrinkA = 0, shrinkB = 4))
        ax.margins(x = 0.25)
    axes[0].set_xlabel("Thời gian huấn luyện (phút)")
    axes[1].set_xlabel("Số tham số trainable (thang log)")
    axes[1].set_xscale("log")
    axes[0].set_ylabel("Test accuracy (%)")
    mode_handles = [plt.Line2D([], [], marker = "o", ls = "", color = MODE_COLOR[m], label = MODE_LABEL[m]) for m in MODES]
    arch_handles = [plt.Line2D([], [], marker = ARCH_MARKER[a], ls = "", color = MUTED, label = ARCH_LABEL[a]) for a in ARCH_LABEL]
    fig.legend(handles = mode_handles + arch_handles, ncol = 6, loc = "upper center", fontsize = 7.5)
    fig.tight_layout(rect = (0, 0, 1, 0.9))
    fig.savefig(out_dir / "accuracy_vs_cost.png", dpi = 200)
    plt.close(fig)


def plot_coarse_confusion(cm, coarse_names, title, path):
    # Row-normalized 20x20 confusion matrix over superclasses; sequential single-hue ramp
    from matplotlib.colors import LinearSegmentedColormap
    norm = cm / cm.sum(1, keepdims = True) * 100
    fig, ax = plt.subplots(figsize = (7.4, 6.6))
    ax.grid(False)
    im = ax.imshow(norm, cmap = LinearSegmentedColormap.from_list("seq", SEQ_BLUE), vmin = 0, vmax = 100)
    for i in range(len(norm)):
        for j in range(len(norm)):
            if norm[i, j] >= 1:
                ax.text(j, i, f"{norm[i, j]:.0f}", ha = "center", va = "center", fontsize = 6,
                        color = SURFACE if norm[i, j] > 50 else INK_2)
    short = [n.replace("_", " ") for n in coarse_names]
    ax.set_xticks(range(len(short)), short, rotation = 60, ha = "right", fontsize = 7)
    ax.set_yticks(range(len(short)), short, fontsize = 7)
    ax.set_xlabel("Siêu lớp dự đoán")
    ax.set_ylabel("Siêu lớp thật")
    ax.set_title(title)
    fig.colorbar(im, ax = ax, fraction = 0.04, pad = 0.02, label = "% ảnh của hàng")
    fig.tight_layout()
    fig.savefig(path, dpi = 200)
    plt.close(fig)


def plot_superclass_gap(acc_cnn, acc_vit, coarse_names, out_dir):
    # One series: accuracy difference ViT - CNN per superclass, sorted; the sign is read from the bar direction
    diff = (acc_vit - acc_cnn) * 100
    order = np.argsort(diff)
    fig, ax = plt.subplots(figsize = (7.2, 4.6))
    ax.barh(range(len(order)), diff[order], color = NEUTRAL_BAR, height = 0.7)
    for k, i in enumerate(order):
        ax.text(diff[i] + (0.3 if diff[i] >= 0 else -0.3), k, f"{diff[i]:+.1f}".replace(".", ","), va = "center",
                ha = "left" if diff[i] >= 0 else "right", fontsize = 7, color = INK_2)
    ax.set_yticks(range(len(order)), [coarse_names[i].replace("_", " ") for i in order], fontsize = 7.5)
    ax.axvline(0, color = AXIS, linewidth = 1)
    ax.set_xlabel("Chênh lệch accuracy theo siêu lớp: DeiT-Tiny tốt nhất trừ ResNet-18 tốt nhất (điểm %)")
    ax.grid(axis = "y", visible = False)
    lim = np.abs(diff).max() + 2
    ax.set_xlim(-lim, lim)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer = True))
    fig.tight_layout()
    fig.savefig(out_dir / "superclass_gap.png", dpi = 200)
    plt.close(fig)


def plot_error_examples(images, labels, pred_cnn, pred_vit, names, out_dir, per_group = 6, seed = config.SEED):
    # Rows: only ResNet wrong / only DeiT wrong / both wrong. Each caption: true label, ResNet guess, DeiT guess
    rng = np.random.default_rng(seed)
    groups = [("Chỉ ResNet sai", (pred_cnn != labels) & (pred_vit == labels)),
              ("Chỉ DeiT sai", (pred_cnn == labels) & (pred_vit != labels)),
              ("Cả hai sai", (pred_cnn != labels) & (pred_vit != labels))]
    fig, axes = plt.subplots(3, per_group, figsize = (per_group * 1.35, 3 * 1.75))
    for row, (title, mask) in enumerate(groups):
        idx = np.flatnonzero(mask)
        pick = rng.choice(idx, size = min(per_group, len(idx)), replace = False) if len(idx) else []
        for col in range(per_group):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(pick):
                i = pick[col]
                ax.imshow(images[i])
                ax.set_title(f"thật: {names[labels[i]]}\nR: {names[pred_cnn[i]]}\nD: {names[pred_vit[i]]}", fontsize = 6, color = INK_2)
        axes[row, 0].text(-0.15, 0.5, title, transform = axes[row, 0].transAxes, rotation = 90, ha = "right", va = "center",
                          fontsize = 8, color = INK)
    fig.tight_layout()
    fig.savefig(out_dir / "error_examples.png", dpi = 200)
    plt.close(fig)


# ----------------------------------------------------------------------------- error analysis
def top_confused_pairs(labels, preds, names, k = 8):
    cm = confusion_matrix(labels, preds, labels = range(config.NUM_CLASSES))
    np.fill_diagonal(cm, 0)
    order = np.argsort(cm, axis = None)[::-1][:k]
    return [(names[i // config.NUM_CLASSES], names[i % config.NUM_CLASSES], int(cm.flat[i])) for i in order]


def same_superclass_share(labels, preds, coarse_of):
    wrong = preds != labels
    return float((coarse_of[preds[wrong]] == coarse_of[labels[wrong]]).mean())


def error_analysis(runs, best, out_dir, tex_dir):
    test = load_cifar100(train = False)
    labels, names = np.array(test.targets), test.classes
    coarse_names = test.coarse_classes
    coarse_of = np.zeros(config.NUM_CLASSES, dtype = int)
    for f, c in zip(test.targets, test.coarse_targets):
        coarse_of[f] = c
    labels_c = coarse_of[labels]

    extra, per_class = {}, {}
    for arch, name in best.items():
        preds = runs[name]["preds"]
        cm_c = confusion_matrix(labels_c, coarse_of[preds], labels = range(len(coarse_names)))
        plot_coarse_confusion(cm_c, coarse_names, f"{ARCH_LABEL[arch]} ({name}): confusion matrix 20 siêu lớp",
                              out_dir / f"confusion_coarse_{arch}.png")
        per_class[arch] = cm_c.diagonal() / cm_c.sum(1)
        key = ARCH_MACRO[arch]
        extra[f"Best{key}Name"] = tex_name(name)
        extra[f"Best{key}SameSuper"] = pct(same_superclass_share(labels, preds, coarse_of), 1)
        extra[f"Best{key}CoarseAcc"] = pct((coarse_of[preds] == labels_c).mean())
        class_acc = np.array([(preds[labels == c] == c).mean() for c in range(config.NUM_CLASSES)])
        worst, top = np.argsort(class_acc)[:5], np.argsort(class_acc)[::-1][:5]
        extra[f"Best{key}Worst"] = ", ".join(f"{names[c].replace('_', ' ')} ({pct(class_acc[c], 0)}\\%)" for c in worst)
        extra[f"Best{key}Top"] = ", ".join(f"{names[c].replace('_', ' ')} ({pct(class_acc[c], 0)}\\%)" for c in top)
        rows = [f"{t.replace('_', ' ')} & {p.replace('_', ' ')} & {n} \\\\" for t, p, n in top_confused_pairs(labels, preds, names)]
        (tex_dir / f"confused_pairs_{arch}.tex").write_text("\n".join(rows) + "\n")

    if len(best) == 2:
        pred_cnn, pred_vit = runs[best["resnet18"]]["preds"], runs[best["deit_tiny"]]["preds"]
        plot_superclass_gap(per_class["resnet18"], per_class["deit_tiny"], coarse_names, out_dir)
        plot_error_examples(test.data, labels, pred_cnn, pred_vit, names, out_dir)
        ok_c, ok_v = pred_cnn == labels, pred_vit == labels
        n = len(labels)
        extra.update({"AgreeBothRight": pct((ok_c & ok_v).sum() / n), "AgreeOnlyCnn": pct((ok_c & ~ok_v).sum() / n),
                      "AgreeOnlyVit": pct((~ok_c & ok_v).sum() / n), "AgreeBothWrong": pct((~ok_c & ~ok_v).sum() / n),
                      "OracleAcc": pct((ok_c | ok_v).sum() / n)})
    return extra


# ----------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description = "Tổng hợp kết quả A1.2: bảng, biểu đồ, phân tích lỗi")
    parser.add_argument("--run-dir", default = str(config.OUTPUT_DIR / "runs"))
    parser.add_argument("--out-dir", default = str(config.OUTPUT_DIR / "a1_2"))
    parser.add_argument("--tex-dir", default = str(config.ROOT / "report" / "generated"))
    args = parser.parse_args()
    run_dir, out_dir, tex_dir = Path(args.run_dir), Path(args.out_dir), Path(args.tex_dir)
    out_dir.mkdir(parents = True, exist_ok = True)
    tex_dir.mkdir(parents = True, exist_ok = True)

    runs = load_runs(run_dir)
    print(f"Found {len(runs)}/8 finished runs in {run_dir}")
    if not runs:
        return
    rows = summary_rows(runs)
    best = best_per_arch(runs)
    write_tables(rows, tex_dir, out_dir)
    plot_accuracy_by_mode(rows, out_dir)
    plot_learning_curves(runs, out_dir)
    plot_cost(rows, out_dir)
    extra = error_analysis(runs, best, out_dir, tex_dir)
    extra["RunsDone"] = str(len(runs))
    extra["GpuName"] = next(r["gpu"] for r in rows if not r["missing"])
    write_macros(rows, extra, tex_dir)
    for r in rows:
        if not r["missing"]:
            print(f"{r['name']:20s} test acc {r['test_acc'] * 100:6.2f} | F1 {r['test_f1'] * 100:6.2f} | {r['train_min']:6.1f} min")


if __name__ == "__main__":
    main()
