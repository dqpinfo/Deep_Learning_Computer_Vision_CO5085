###################################
#   Shared configuration for A1   #
###################################

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"                    # CIFAR-100 download location (not committed to GitHub)
OUTPUT_DIR = ROOT / "output"                # EDA figures, protocol table, training logs
SPLIT_FILE = ROOT / "splits" / "cifar100_split.json"

SEED = 42                                   # Fixed seed so the data split and training are reproducible

# Data split: 50,000 original train images -> 45,000 train + 5,000 validation (50 per class); the 10,000 test images are kept as is
NUM_CLASSES = 100
VAL_PER_CLASS = 50

# Model input size. The original 32x32 images are upscaled to 224x224
# because the pretrained (ImageNet) weights and DeiT-Tiny (patch 16) are designed for 224x224.
# Every configuration uses the same size for a fair comparison.
IMG_SIZE = 224

# Normalize with ImageNet statistics for every configuration (including from-scratch),
# so differences between experiments come only from architecture / initialization / freezing.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Two architecture families (model names in the timm library)
ARCHS = {
    "resnet18": "resnet18.a1_in1k",               # CNN, ~11.2M parameters (100-class head)
    "deit_tiny": "deit_tiny_patch16_224.fb_in1k",  # Transformer, ~5.5M parameters (100-class head)
}

# A1.2 experiment matrix: 2 architectures x (from-scratch + 3 freezing modes on the pretrained model) = 8 runs
EXPERIMENTS = [
    {"name": f"{arch}_{mode}", "arch": arch, "pretrained": mode != "scratch", "freeze": "none" if mode in ("scratch", "full") else mode}
    for arch in ARCHS
    for mode in ("scratch", "head", "partial", "full")
]

# Planned A1.2 hyperparameters (may be tuned slightly, but kept identical within each group of configurations)
TRAIN_HPARAMS = {
    "batch_size": 128,
    "optimizer": "AdamW",
    "weight_decay": 0.05,
    "lr_pretrained": 1e-4,      # Fine-tuning / head training on the pretrained model
    "lr_scratch": 1e-3,         # Training from scratch needs a larger learning rate
    "epochs_pretrained": 10,
    "epochs_scratch": 30,
    "scheduler": "cosine + 1 epoch warmup",
    "label_smoothing": 0.1,
}
