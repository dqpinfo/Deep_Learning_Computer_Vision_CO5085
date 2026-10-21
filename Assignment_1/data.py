###############################################################
#   Load CIFAR-100, split train/val/test, build DataLoaders   #
###############################################################

import io
import json
import ssl
import urllib.request

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

import config

# Official CIFAR-100 mirror on Hugging Face (same images, labels and order as the University of Toronto original).
# Used because the original cs.toronto.edu server downloads very slowly from Vietnam.
HF_URL = "https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/{split}-00000-of-00001.parquet"
CACHE_FILE = config.DATA_DIR / "cifar100.npz"

# python.org builds of Python on macOS may lack root certificates -> use certifi's bundle (SSL verification stays on)
try:
    import certifi
    SSL_CONTEXT = ssl.create_default_context(cafile = certifi.where())
except ImportError:
    SSL_CONTEXT = ssl.create_default_context()


def download_parquet(split):
    path = config.DATA_DIR / "cifar100_hf" / f"{split}.parquet"
    if not path.exists():
        path.parent.mkdir(parents = True, exist_ok = True)
        print(f"Đang tải CIFAR-100 ({split}) từ Hugging Face...")
        with urllib.request.urlopen(HF_URL.format(split = split), context = SSL_CONTEXT) as resp:
            path.write_bytes(resp.read())
    return path


def build_cache():
    # Decode the PNG images in the parquet files into numpy arrays once and cache them for fast later loads
    import pyarrow.parquet as pq

    arrays = {}
    for split in ("train", "test"):
        table = pq.read_table(download_parquet(split))
        rows = table.to_pydict()
        arrays[f"x_{split}"] = np.stack([np.array(Image.open(io.BytesIO(img["bytes"])).convert("RGB")) for img in rows["img"]])
        arrays[f"y_{split}"] = np.array(rows["fine_label"], dtype = np.int64)
        arrays[f"c_{split}"] = np.array(rows["coarse_label"], dtype = np.int64)
        features = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]
    arrays["fine_names"] = np.array(features["fine_label"]["names"])
    arrays["coarse_names"] = np.array(features["coarse_label"]["names"])
    np.savez_compressed(CACHE_FILE, **arrays)


class CIFAR100(Dataset):
    # Same interface as torchvision.datasets.CIFAR100: .data (N, 32, 32, 3) uint8, .targets, .classes

    def __init__(self, train, transform = None):
        if not CACHE_FILE.exists():
            build_cache()
        split = "train" if train else "test"
        cache = np.load(CACHE_FILE)
        self.data = cache[f"x_{split}"]
        self.targets = cache[f"y_{split}"].tolist()
        self.coarse_targets = cache[f"c_{split}"].tolist()
        # The Hugging Face metadata misspells "crab" as "cra" -> restore the original class name
        self.classes = ["crab" if name == "cra" else name for name in cache["fine_names"].tolist()]
        self.coarse_classes = cache["coarse_names"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[index]


def train_transform():
    # Light augmentation, identical for every configuration:
    # random crop with 4-pixel padding + horizontal flip at the native 32x32 resolution, then upscale to 224x224
    return transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(config.IMG_SIZE, interpolation = transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD),
    ])


def eval_transform():
    # Validation/test: no augmentation, only resize and normalize
    return transforms.Compose([
        transforms.Resize(config.IMG_SIZE, interpolation = transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD),
    ])


def load_cifar100(train, transform = None):
    return CIFAR100(train = train, transform = transform)


def make_split(targets):
    # Stratified split: VAL_PER_CLASS random images per class go to validation, the rest to train
    rng = np.random.default_rng(config.SEED)
    targets = np.asarray(targets)
    train_idx, val_idx = [], []
    for c in range(config.NUM_CLASSES):
        idx = rng.permutation(np.flatnonzero(targets == c))
        val_idx.extend(idx[:config.VAL_PER_CLASS].tolist())
        train_idx.extend(idx[config.VAL_PER_CLASS:].tolist())
    return sorted(train_idx), sorted(val_idx)


def get_split():
    # Load the saved split (so every experiment uses exactly the same split); create and save it if missing
    if config.SPLIT_FILE.exists():
        split = json.loads(config.SPLIT_FILE.read_text())
        return split["train"], split["val"]

    train_idx, val_idx = make_split(load_cifar100(train = True).targets)
    config.SPLIT_FILE.parent.mkdir(parents = True, exist_ok = True)
    config.SPLIT_FILE.write_text(json.dumps({"seed": config.SEED, "train": train_idx, "val": val_idx}))
    return train_idx, val_idx


def get_datasets():
    train_idx, val_idx = get_split()
    train_set = Subset(load_cifar100(train = True, transform = train_transform()), train_idx)
    val_set = Subset(load_cifar100(train = True, transform = eval_transform()), val_idx)
    test_set = load_cifar100(train = False, transform = eval_transform())
    return train_set, val_set, test_set


def get_data_loaders(batch_size = 128, num_workers = 2):
    train_set, val_set, test_set = get_datasets()
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = num_workers, drop_last = True)
    val_loader = DataLoader(val_set, batch_size = batch_size, shuffle = False, num_workers = num_workers)
    test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = False, num_workers = num_workers)
    return train_loader, val_loader, test_loader
