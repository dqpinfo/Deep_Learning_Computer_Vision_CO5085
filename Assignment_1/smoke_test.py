#############################################################################
#   Quick check of all 8 experiment configurations before real training     #
#   - build each model, count total / trainable parameters                  #
#   - run one forward + backward step, check frozen parts get no gradient   #
#   Output: output/protocol_table.md and output/protocol_table.json         #
#############################################################################

import json
import time

import torch
import torch.nn as nn

import config
from models import build_model, count_parameters, set_train_mode


def check_experiment(exp, batch):
    images, labels = batch
    model = build_model(exp["arch"], exp["pretrained"], exp["freeze"])
    total, trainable = count_parameters(model)

    set_train_mode(model)
    start = time.perf_counter()
    loss = nn.CrossEntropyLoss()(model(images), labels)
    loss.backward()
    step_time = time.perf_counter() - start

    # Frozen parameters must have no gradient, trainable parameters must have one
    frozen_ok = all(p.grad is None for p in model.parameters() if not p.requires_grad)
    trainable_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    return {
        **exp,
        "params_total": total,
        "params_trainable": trainable,
        "trainable_ratio": round(trainable / total, 4),
        "loss_first_step": round(loss.item(), 4),
        "cpu_step_sec_batch8": round(step_time, 2),
        "freeze_check": frozen_ok and trainable_ok,
    }


def to_markdown(rows):
    lines = ["| Cấu hình | Kiến trúc | Khởi tạo | Đóng băng | Tham số tổng | Tham số trainable | Tỉ lệ | Kiểm tra |",
             "|---|---|---|---|---:|---:|---:|:---:|"]
    for r in rows:
        lines.append(f"| {r['name']} | {r['arch']} | {'pretrained' if r['pretrained'] else 'scratch'} | {r['freeze']} | "
                     f"{r['params_total']:,} | {r['params_trainable']:,} | {r['trainable_ratio']:.1%} | "
                     f"{'OK' if r['freeze_check'] else 'LỖI'} |")
    return "\n".join(lines) + "\n"


def main():
    torch.manual_seed(config.SEED)
    batch = (torch.randn(8, 3, config.IMG_SIZE, config.IMG_SIZE), torch.randint(0, config.NUM_CLASSES, (8,)))
    rows = []
    for exp in config.EXPERIMENTS:
        row = check_experiment(exp, batch)
        rows.append(row)
        print(f"{row['name']:<20} trainable {row['params_trainable']:>11,} / {row['params_total']:,}  check={row['freeze_check']}")

    config.OUTPUT_DIR.mkdir(parents = True, exist_ok = True)
    (config.OUTPUT_DIR / "protocol_table.json").write_text(json.dumps(rows, indent = 2))
    (config.OUTPUT_DIR / "protocol_table.md").write_text(to_markdown(rows))
    if not all(r["freeze_check"] for r in rows):
        raise SystemExit("Có cấu hình đóng băng sai, xem protocol_table.md")


if __name__ == "__main__":
    main()
