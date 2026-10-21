####################################################################
#   Build the A1.1 report (HTML -> PDF with headless Chrome)       #
#   Reads output/eda/summary.json and output/protocol_table.json   #
#   Output: Reports/A1_report.pdf                                  #
####################################################################

import base64
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
A1 = HERE.parent
sys.path.insert(0, str(A1))
import config  # noqa: E402

REPORT_PDF = A1.parent / "Reports" / "A1_report.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def img(path):
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}">'


def vn(number):
    # Vietnamese number format: 45.000 instead of 45,000
    return f"{number:,}".replace(",", ".")


def protocol_rows(rows):
    labels = {"scratch": "Học từ đầu", "head": "Chỉ train head", "partial": "Đóng băng một phần", "none": "Fine-tune toàn bộ"}
    out = []
    for r in rows:
        mode = "scratch" if not r["pretrained"] else r["freeze"]
        out.append(f"<tr><td><code>{r['name']}</code></td><td>{'ResNet-18' if r['arch'] == 'resnet18' else 'DeiT-Tiny'}</td>"
                   f"<td>{'ImageNet-1k' if r['pretrained'] else 'Ngẫu nhiên'}</td><td>{labels[mode]}</td>"
                   f"<td class='num'>{vn(r['params_trainable'])}</td><td class='num'>{str(round(r['trainable_ratio'] * 100, 1)).replace('.', ',')}%</td></tr>")
    return "\n".join(out)


def coarse_rows(groups):
    return "\n".join(f"<tr><td>{name.replace('_', ' ')}</td><td>{', '.join(c.replace('_', ' ') for c in fines)}</td></tr>"
                     for name, fines in groups.items())


def smoke_summary():
    # Describe the subset test runs (output/smoke/*.json); these are not experiment results
    logs = [json.loads(p.read_text()) for p in sorted((config.OUTPUT_DIR / "smoke").glob("*_*.json"))
            if not p.name.endswith("_test_preds.json")]
    if not logs:
        return "chưa chạy"
    parts = [f"<code>{log['experiment']['name']}</code> ({log['epochs']} epoch trên tập con, thiết bị {log['device']}, "
             f"{log['train_time_sec']:.0f} giây)" for log in logs]
    return "đã chạy " + "; ".join(parts) + " — huấn luyện, đánh giá validation, chọn checkpoint và đánh giá test đều chạy không lỗi"


def render(summary, rows):
    template = (HERE / "template.html").read_text(encoding = "utf-8")
    eda = config.OUTPUT_DIR / "eda"
    hp = config.TRAIN_HPARAMS
    by_name = {r["name"]: r for r in rows}
    values = {
        "TRAIN": vn(summary['split_sizes']['train']), "VAL": vn(summary['split_sizes']['val']),
        "TEST": vn(summary['split_sizes']['test']),
        "MEAN": ", ".join(f"{v:.3f}" for v in summary["pixel_stats_train"]["mean"]),
        "STD": ", ".join(f"{v:.3f}" for v in summary["pixel_stats_train"]["std"]),
        "RESNET_PARAMS": f"{by_name['resnet18_full']['params_total'] / 1e6:.1f}".replace('.', ','),
        "DEIT_PARAMS": f"{by_name['deit_tiny_full']['params_total'] / 1e6:.1f}".replace('.', ','),
        "PROTOCOL_ROWS": protocol_rows(rows), "COARSE_ROWS": coarse_rows(summary["coarse_groups"]),
        "IMG_COUNTS": img(eda / "class_counts.png"), "IMG_SAMPLES": img(eda / "samples_per_class.png"),
        "IMG_AUG": img(eda / "augmentation_examples.png"),
        "BATCH": hp["batch_size"], "WD": hp["weight_decay"], "LR_PRE": hp["lr_pretrained"], "LR_SCR": hp["lr_scratch"],
        "EP_PRE": hp["epochs_pretrained"], "EP_SCR": hp["epochs_scratch"], "LS": hp["label_smoothing"],
        "SEED": config.SEED, "IMG_SIZE": config.IMG_SIZE, "SMOKE": smoke_summary(),
    }
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", str(value))
    return template


def main():
    summary = json.loads((config.OUTPUT_DIR / "eda" / "summary.json").read_text())
    rows = json.loads((config.OUTPUT_DIR / "protocol_table.json").read_text())
    html_path = HERE / "A1_report.html"
    html_path.write_text(render(summary, rows), encoding = "utf-8")
    REPORT_PDF.parent.mkdir(parents = True, exist_ok = True)
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={REPORT_PDF}", html_path.as_uri()], check = True, capture_output = True)
    print(f"Đã tạo {REPORT_PDF}")


if __name__ == "__main__":
    main()
