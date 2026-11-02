###################################
#   Shared configuration for A2   #
###################################

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"                    # Oxford-IIIT Pet download location (not committed to GitHub)
RAW_DIR = DATA_DIR / "raw"                  # images.tar.gz + annotations.tar.gz
PET_DIR = DATA_DIR / "oxford-iiit-pet"      # Extracted images/ and annotations/
OUTPUT_DIR = ROOT / "output"                # EDA figures, protocol table, training logs
SPLIT_FILE = ROOT / "splits" / "pet_split.json"

SEED = 42                                   # Fixed seed so the data split and training are reproducible

# Official archives of the Oxford-IIIT Pet dataset (Parkhi et al., CVPR 2012).
# The robots.ox.ac.uk server is not reachable from our network, so the identical files are taken from a
# Hugging Face mirror; the MD5 checksums below are the ones torchvision uses for the original server.
ARCHIVES = {
    "images.tar.gz": {
        "urls": ["https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
                 "https://huggingface.co/datasets/xingslong/oxford-iiit-pet/resolve/main/images.tar.gz"],
        "md5": "5c4f3ee8e5d25df40f4fd59a7f44e54c",
    },
    "annotations.tar.gz": {
        "urls": ["https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
                 "https://huggingface.co/datasets/xingslong/oxford-iiit-pet/resolve/main/annotations.tar.gz"],
        "md5": "95a8c909bbe2e81eed6a22bccdf3f68f",
    },
}

# Task: 38-class semantic segmentation = background (0) + 37 breeds (1-37).
# Pet pixels take the breed id of the image; the "border" region of the official trimap is ignored.
NUM_BREEDS = 37
NUM_CLASSES = NUM_BREEDS + 1
IGNORE_INDEX = 255
TRIMAP_PET, TRIMAP_BACKGROUND, TRIMAP_BORDER = 1, 2, 3   # Pixel values of the official trimap PNGs

# Data split: the official trainval list (3,680 images) -> train + validation (10 images per breed);
# the official test list (3,669 images) is kept as is
VAL_PER_BREED = 10

# Training/evaluation resolution. Images are resized to a fixed square so both models see identical inputs
# and batches can be stacked; metrics are computed at this resolution.
IMG_SIZE = 384

# ImageNet statistics: both encoders were pretrained on ImageNet-1k with this normalization
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Two architecture families, both with ImageNet-1k pretrained encoders and a similar parameter budget
ARCHS = {
    "unet_r50": "resnet50.a1_in1k",         # CNN: U-Net decoder on a timm ResNet-50 encoder
    "segformer_b2": "nvidia/mit-b2",        # Transformer: SegFormer with the MiT-B2 encoder (Hugging Face)
}

# A2.2 experiment matrix: 2 architectures x 4 training-set fractions (data-efficiency study);
# the 100% runs are the main comparison
TRAIN_FRACTIONS = (1.0, 0.5, 0.25, 0.1)
EXPERIMENTS = [
    {"name": f"{arch}_f{int(frac * 100)}", "arch": arch, "train_fraction": frac}
    for arch in ARCHS
    for frac in TRAIN_FRACTIONS
]

# Planned A2.2 hyperparameters (identical for both architectures)
TRAIN_HPARAMS = {
    "batch_size": 16,
    "optimizer": "AdamW",
    "weight_decay": 0.01,
    "lr_encoder": 6e-5,         # Pretrained encoder: small learning rate
    "lr_decoder": 6e-4,         # Randomly initialized decoder / head: 10x larger
    "epochs": 40,
    "scheduler": "poly (power 1.0) + 1 epoch warmup",
    "loss": "cross-entropy (ignore border)",
}
