#####################################################################
#   Download Oxford-IIIT Pet, build 38-class masks, split, loaders  #
#####################################################################

import hashlib
import json
import shutil
import ssl
import tarfile
import urllib.request

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

import config

# python.org builds of Python on macOS may lack root certificates -> use certifi's bundle (SSL verification stays on)
try:
    import certifi
    SSL_CONTEXT = ssl.create_default_context(cafile = certifi.where())
except ImportError:
    SSL_CONTEXT = ssl.create_default_context()


def md5sum(path):
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(name, info):
    path = config.RAW_DIR / name
    if path.exists() and md5sum(path) == info["md5"]:
        return path
    path.parent.mkdir(parents = True, exist_ok = True)
    for url in info["urls"]:
        try:
            print(f"Đang tải {name} từ {url} ...")
            with urllib.request.urlopen(url, context = SSL_CONTEXT, timeout = 30) as resp, open(path, "wb") as f:
                shutil.copyfileobj(resp, f, 1 << 20)
        except OSError as err:
            print(f"  Không tải được ({err}), thử nguồn tiếp theo")
            continue
        if md5sum(path) == info["md5"]:
            return path
        print("  Sai MD5, thử nguồn tiếp theo")
    raise RuntimeError(f"Không tải được {name} với MD5 đúng")


def prepare_dataset():
    # Download + verify both archives and extract them once into data/oxford-iiit-pet/
    if (config.PET_DIR / "annotations" / "trainval.txt").exists():
        return
    for name, info in config.ARCHIVES.items():
        with tarfile.open(download_archive(name, info)) as tar:
            tar.extractall(config.PET_DIR, filter = "data")


def read_list(split):
    # Official list format: "<image name> <breed id 1-37> <species 1=cat 2=dog> <breed index within species>"
    rows = []
    for line in (config.PET_DIR / "annotations" / f"{split}.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            name, breed, species, _ = line.split()
            rows.append({"name": name, "breed": int(breed), "species": int(species)})
    return rows


def breed_names():
    # Breed name = image name without the trailing number (e.g. "Bengal_12" -> "Bengal"), indexed by breed id
    names = {}
    for row in read_list("trainval") + read_list("test"):
        names[row["breed"]] = row["name"].rsplit("_", 1)[0]
    return [names[i] for i in range(1, config.NUM_BREEDS + 1)]


def class_names():
    return ["background"] + breed_names()


def trimap_to_mask(trimap, breed):
    # Trimap 1 = pet, 2 = background, 3 = border -> 38-class mask (0 = background, breed id, 255 = ignore)
    mask = np.full(trimap.shape, config.IGNORE_INDEX, dtype = np.uint8)
    mask[trimap == config.TRIMAP_BACKGROUND] = 0
    mask[trimap == config.TRIMAP_PET] = breed
    return mask


def load_image(name):
    # A few files are grayscale, RGBA or CMYK JPEGs -> always convert to RGB
    return Image.open(config.PET_DIR / "images" / f"{name}.jpg").convert("RGB")


def load_trimap(name):
    return np.array(Image.open(config.PET_DIR / "annotations" / "trimaps" / f"{name}.png"))


class PetSegmentation(Dataset):
    # Returns (image tensor 3xHxW normalized, mask tensor HxW int64 with values 0-37 or 255)

    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = tv_tensors.Image(v2.functional.pil_to_tensor(load_image(row["name"])))
        mask = tv_tensors.Mask(torch.from_numpy(trimap_to_mask(load_trimap(row["name"]), row["breed"])))
        image, mask = self.transform(image, mask)
        return image, mask.long()


def train_transform():
    # Geometric transforms are applied jointly to image and mask (nearest-neighbour for the mask);
    # colour jitter only touches the image
    return v2.Compose([
        v2.RandomResizedCrop(config.IMG_SIZE, scale = (0.5, 1.0), ratio = (3 / 4, 4 / 3), antialias = True),
        v2.RandomHorizontalFlip(),
        v2.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2),
        v2.ToDtype(torch.float32, scale = True),
        v2.Normalize(config.MEAN, config.STD),
    ])


def eval_transform():
    # Validation/test: plain resize of the whole image to IMG_SIZE x IMG_SIZE, no augmentation
    return v2.Compose([
        v2.Resize((config.IMG_SIZE, config.IMG_SIZE), antialias = True),
        v2.ToDtype(torch.float32, scale = True),
        v2.Normalize(config.MEAN, config.STD),
    ])


def make_split(rows):
    # Stratified split: VAL_PER_BREED random trainval images per breed go to validation, the rest to train
    rng = np.random.default_rng(config.SEED)
    train, val = [], []
    for breed in range(1, config.NUM_BREEDS + 1):
        names = sorted(r["name"] for r in rows if r["breed"] == breed)
        order = rng.permutation(len(names))
        val.extend(names[i] for i in order[:config.VAL_PER_BREED])
        train.extend(names[i] for i in order[config.VAL_PER_BREED:])
    return sorted(train), sorted(val)


def subsample(names, rows_by_name, fraction):
    # Stratified training subset for the data-efficiency study (same seed -> nested, reproducible subsets)
    if fraction >= 1.0:
        return names
    rng = np.random.default_rng(config.SEED)
    keep = []
    for breed in range(1, config.NUM_BREEDS + 1):
        group = sorted(n for n in names if rows_by_name[n]["breed"] == breed)
        order = rng.permutation(len(group))
        keep.extend(group[i] for i in order[:max(1, round(fraction * len(group)))])
    return sorted(keep)


def empty_trimaps(rows):
    # A few official trimaps mark the whole image as background although a pet is clearly visible
    # (annotation errors); these images would teach / score "pet = background", so they are excluded
    return sorted(r["name"] for r in rows if not (load_trimap(r["name"]) == config.TRIMAP_PET).any())


def get_split():
    # Load the saved split (so every experiment uses exactly the same split); create and save it if missing
    prepare_dataset()
    if config.SPLIT_FILE.exists():
        split = json.loads(config.SPLIT_FILE.read_text())
        return split["train"], split["val"], split["test"]

    trainval, test_rows = read_list("trainval"), read_list("test")
    excluded = empty_trimaps(trainval + test_rows)
    train, val = make_split([r for r in trainval if r["name"] not in excluded])
    test = sorted(r["name"] for r in test_rows if r["name"] not in excluded)
    config.SPLIT_FILE.parent.mkdir(parents = True, exist_ok = True)
    config.SPLIT_FILE.write_text(json.dumps({"seed": config.SEED, "excluded_empty_trimap": excluded,
                                             "train": train, "val": val, "test": test}))
    return train, val, test


def get_rows():
    prepare_dataset()
    return {r["name"]: r for r in read_list("trainval") + read_list("test")}


def get_datasets(train_fraction = 1.0):
    rows = get_rows()
    train, val, test = get_split()
    train = subsample(train, rows, train_fraction)
    return (PetSegmentation([rows[n] for n in train], train_transform()),
            PetSegmentation([rows[n] for n in val], eval_transform()),
            PetSegmentation([rows[n] for n in test], eval_transform()))


def get_data_loaders(batch_size = 16, num_workers = 2, train_fraction = 1.0):
    train_set, val_set, test_set = get_datasets(train_fraction)
    kwargs = {"batch_size": batch_size, "num_workers": num_workers, "persistent_workers": num_workers > 0}
    return (DataLoader(train_set, shuffle = True, drop_last = True, **kwargs),
            DataLoader(val_set, shuffle = False, **kwargs),
            DataLoader(test_set, shuffle = False, **kwargs))
