import random
import re
from collections import Counter

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import utils

from kd_config import (
    AUG_DISPLAY_NAMES,
    AUG_TYPES,
    BATCH_SIZE,
    CLASS_NAMES,
    CROP_SIZE,
    EMOTION_DIR,
    EMOTION_ID_TO_NAME,
    EXTENDED_DIR,
    IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SEED,
    TEST_SPLIT,
    VAL_SPLIT,
)


def resize_and_center(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    left = (IMG_SIZE - CROP_SIZE) // 2
    top = (IMG_SIZE - CROP_SIZE) // 2
    right = left + CROP_SIZE
    bottom = top + CROP_SIZE
    return img.crop((left, top, right, bottom))


def load_ckplus_samples():
    samples = []
    for subject_dir in sorted(EMOTION_DIR.iterdir()):
        if not subject_dir.is_dir():
            continue
        for seq_dir in sorted(subject_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            txt_files = sorted(seq_dir.glob("*.txt"))
            if not txt_files:
                continue
            emotion_id = int(re.search(r"\d+", txt_files[0].read_text().strip()).group())
            if emotion_id not in EMOTION_ID_TO_NAME:
                continue
            img_seq_dir = EXTENDED_DIR / seq_dir.relative_to(EMOTION_DIR)
            image_files = sorted(
                p for p in img_seq_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
            )
            if image_files:
                samples.append((str(image_files[-1]), emotion_id - 1))
    return samples


def describe_samples(samples):
    counts = Counter(label for _, label in samples)
    return {CLASS_NAMES[idx]: counts[idx] for idx in sorted(counts)}


def get_aug_transform(aug_type):
    if aug_type == "original":
        return resize_and_center
    if aug_type == "horizontal_flip":
        return lambda img: transforms.functional.hflip(resize_and_center(img))
    if aug_type == "sharpen":
        def sharpen(img):
            arr = np.array(resize_and_center(img))
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            return Image.fromarray(cv2.filter2D(arr, -1, kernel))
        return sharpen
    if aug_type == "rotate15":
        return lambda img: transforms.functional.rotate(resize_and_center(img), 15)
    if aug_type == "rotate_minus15":
        return lambda img: transforms.functional.rotate(resize_and_center(img), -15)
    if aug_type == "autocontrast":
        return lambda img: transforms.functional.autocontrast(resize_and_center(img))
    if aug_type == "equalize":
        return lambda img: transforms.functional.equalize(resize_and_center(img))
    if aug_type == "colorjitter":
        return lambda img: transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)(resize_and_center(img))
    if aug_type == "randomaffine":
        return lambda img: transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10)(resize_and_center(img))
    if aug_type == "randomerasing":
        def apply_erasing(img):
            tensor_img = transforms.ToTensor()(resize_and_center(img))
            erased = transforms.RandomErasing(p=1.0, scale=(0.1, 0.2), ratio=(0.3, 3.3))(tensor_img)
            return transforms.ToPILImage()(erased)
        return apply_erasing
    if aug_type == "randomresizedcrop":
        return lambda img: transforms.RandomResizedCrop(CROP_SIZE, scale=(0.7, 1.0), ratio=(0.75, 1.33))(img.resize((IMG_SIZE, IMG_SIZE)))
    if aug_type == "hist_eq":
        def hist_eq(img):
            arr = np.array(resize_and_center(img))
            img_yuv = cv2.cvtColor(arr, cv2.COLOR_RGB2YUV)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            return Image.fromarray(cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB))
        return hist_eq
    if aug_type == "clahe":
        def clahe(img):
            arr = np.array(resize_and_center(img))
            lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            clahe_op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_channel = clahe_op.apply(l_channel)
            merged = cv2.merge((l_channel, a_channel, b_channel))
            return Image.fromarray(cv2.cvtColor(merged, cv2.COLOR_LAB2RGB))
        return clahe
    if aug_type == "gaussian_blur":
        return lambda img: Image.fromarray(cv2.GaussianBlur(np.array(resize_and_center(img)), (7, 7), 0))
    if aug_type == "adjust_sharpness":
        return lambda img: transforms.functional.adjust_sharpness(resize_and_center(img), sharpness_factor=2.0)
    if aug_type == "gaussian_noise":
        def gaussian_noise(img):
            arr = np.array(resize_and_center(img))
            noise = np.random.normal(0, 25, arr.shape)
            return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
        return gaussian_noise
    raise ValueError(f"Unknown aug_type: {aug_type}")


class RandomAugImageFolder(Dataset):
    def __init__(self, samples, aug_types):
        self.samples = samples
        self.aug_types = aug_types
        self.normalizer = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __len__(self):
        return len(self.samples)

    def _normalize(self, image):
        tensor = transforms.ToTensor()(image)
        return self.normalizer(tensor)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        aug_type = random.choice(self.aug_types)
        pil_img = Image.open(img_path).convert("RGB")

        img = get_aug_transform(aug_type)(pil_img)
        return self._normalize(img), label


def build_dataloaders(samples, device, train_aug_types=None):
    train_aug_types = train_aug_types or AUG_TYPES
    dataset_size = len(samples)
    test_size = int(TEST_SPLIT * dataset_size)
    val_size = int(VAL_SPLIT * dataset_size)
    train_size = dataset_size - val_size - test_size

    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]
    test_samples = [samples[i] for i in test_indices]

    train_dataset = RandomAugImageFolder(train_samples, train_aug_types)
    val_dataset = RandomAugImageFolder(val_samples, ["original"])
    test_dataset = RandomAugImageFolder(test_samples, ["original"])

    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "train_loader": DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda")),
        "val_loader": DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda")),
        "test_loader": DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda")),
    }


def all_augmentations(img):
    return [(name, get_aug_transform(aug_type)(img)) for name, aug_type in zip(AUG_DISPLAY_NAMES, AUG_TYPES)]


def make_batch_grid(images, nrow=7, max_images=None):
    if max_images is not None:
        images = images[:max_images]
    return utils.make_grid(images, nrow=nrow)
