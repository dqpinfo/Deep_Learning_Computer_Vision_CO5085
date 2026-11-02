##########################################################################
#   A2.1 pipeline check: real DataLoader batch -> both models ->         #
#   loss (border ignored) -> backward -> metrics, plus a tiny overfit    #
#   test. Writes output/protocol_table.json and .md                      #
##########################################################################

import json
import time

import torch
import torch.nn as nn

import config
from data import get_data_loaders
from metrics import SegMetrics
from models import build_model, count_parameters, param_groups


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def check_model(arch, images, masks, device, steps = 10):
    torch.manual_seed(config.SEED)
    model = build_model(arch, pretrained = True).to(device)
    params = count_parameters(model)
    criterion = nn.CrossEntropyLoss(ignore_index = config.IGNORE_INDEX)

    # One forward/backward pass: output shape and gradients on both parameter groups
    model.train()
    logits = model(images)
    assert logits.shape == (images.size(0), config.NUM_CLASSES, *images.shape[-2:]), logits.shape
    criterion(logits, masks).backward()
    encoder, decoder = param_groups(model)
    has_grad = all(any(p.grad is not None and p.grad.abs().sum() > 0 for p in group) for group in (encoder, decoder))

    # Tiny overfit test on the same batch: the loss must go down if the pipeline is wired correctly
    hp = config.TRAIN_HPARAMS
    optimizer = torch.optim.AdamW([{"params": encoder, "lr": hp["lr_encoder"]}, {"params": decoder, "lr": hp["lr_decoder"]}],
                                  weight_decay = hp["weight_decay"])
    losses, start = [], time.perf_counter()
    for step in range(steps):
        optimizer.zero_grad(set_to_none = True)
        loss = criterion(model(images), masks)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"  [{arch}] bước {step + 1}/{steps}: loss {losses[-1]:.3f}", flush = True)
    sync(device)
    seconds = time.perf_counter() - start

    model.eval()
    metrics = SegMetrics()
    with torch.no_grad():
        metrics.update(model(images).argmax(1), masks)
    result = metrics.compute()
    del model, optimizer                                # Free memory before the next model (8 GB Mac)
    if device.type == "mps":
        torch.mps.empty_cache()
    return {"arch": arch, "encoder_name": config.ARCHS[arch], **params, "output_shape": list(logits.shape),
            "grad_ok": has_grad, "loss_first": losses[0], "loss_last": losses[-1], "overfit_steps": steps,
            "sec_per_step": seconds / steps, "pixel_acc_after": result["pixel_acc"], "pet_iou_after": result["pet_iou"]}


def main():
    device = pick_device()
    train_loader, val_loader, test_loader = get_data_loaders(batch_size = 2, num_workers = 0)
    images, masks = next(iter(train_loader))
    print(f"Thiết bị: {device} | batch ảnh {tuple(images.shape)} {images.dtype} | mask {tuple(masks.shape)} "
          f"giá trị {sorted(masks.unique().tolist())}")
    print(f"Số batch: train {len(train_loader)}, val {len(val_loader)}, test {len(test_loader)} (batch 2)")
    images, masks = images.to(device), masks.to(device)

    rows = []
    for arch in config.ARCHS:
        row = check_model(arch, images, masks, device)
        rows.append(row)
        print(f"[{arch}] tham số {row['total']:,} (encoder {row['encoder']:,}, decoder {row['decoder']:,}) | "
              f"gradient {'OK' if row['grad_ok'] else 'LỖI'} | loss {row['loss_first']:.3f} -> {row['loss_last']:.3f} "
              f"sau {row['overfit_steps']} bước | {row['sec_per_step']:.2f} s/bước")

    out = {"device": str(device), "batch_shape": list(images.shape), "models": rows}
    config.OUTPUT_DIR.mkdir(parents = True, exist_ok = True)
    (config.OUTPUT_DIR / "protocol_table.json").write_text(json.dumps(out, indent = 2))
    lines = ["| Mô hình | Encoder | Tổng tham số | Encoder | Decoder | Gradient | Loss đầu -> cuối |",
             "|---|---|---:|---:|---:|---|---|"]
    for r in rows:
        lines.append(f"| {r['arch']} | {r['encoder_name']} | {r['total']:,} | {r['encoder']:,} | {r['decoder']:,} | "
                     f"{'OK' if r['grad_ok'] else 'LỖI'} | {r['loss_first']:.3f} -> {r['loss_last']:.3f} |")
    (config.OUTPUT_DIR / "protocol_table.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
