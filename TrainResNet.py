"""Train ResNet18 to distinguish two image sets (binary classification).

Files:
  - load images from `{BASE_DIR}/exp/{CLASS0_NAME}` (need >= NUM_IMAGES)
  - load images from `{BASE_DIR}/exp/{CLASS1_NAME}` (need >= NUM_IMAGES)
  - save results to `exp/resnet_detector_results.txt`
"""

from __future__ import annotations

import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

BASE_DIR = r"/workspace/W-Back/Data"
# BASE_DIR = r"D:\W-Back\Data"
CLASS0_NAME = "plain"
CLASS1_NAME = "prc"

EXP_DIR = os.path.join(BASE_DIR, "exp")
RESULT_FILE = os.path.join(EXP_DIR, "resnet_detector_results.txt")

CLASS0_DIR = os.path.join(EXP_DIR, CLASS0_NAME)
CLASS1_DIR = os.path.join(EXP_DIR, CLASS1_NAME)

NUM_IMAGES = 5000
VAL_RATIO = 0.2
SEED = 42

EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-4
NUM_WORKERS = 4

IMG_EXTS = {".png", ".jpg", ".jpeg"}


def list_image_paths(images_dir: str) -> list[str]:

    if not os.path.isdir(images_dir): return []
    names = [
        n for n in os.listdir(images_dir)
        if os.path.splitext(n)[1].lower() in IMG_EXTS
        and os.path.isfile(os.path.join(images_dir, n))
    ]

    def sort_key(n: str):

        stem = os.path.splitext(n)[0]
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    names = sorted(names, key=sort_key)

    return [os.path.join(images_dir, n) for n in names]


def require_image_paths(images_dir: str, n: int, tag: str) -> list[str]:

    paths = list_image_paths(images_dir)
    if len(paths) < n: raise FileNotFoundError(f"need {n} images in {tag} dir {images_dir}, found {len(paths)}")

    return paths[:n]


class PairFolderDataset(Dataset):
    """Paired class-0 / class-1 images, indexed 0..n-1."""

    def __init__(self, CLASS0_NAMEs: list[str], CLASS1_NAMEs: list[str], indices: list[int], tfm):

        self.samples: list[tuple[str, int]] = []

        for i in indices:
    
            self.samples.append((CLASS0_NAMEs[i], 0))
            self.samples.append((CLASS1_NAMEs[i], 1))

        self.tfm = tfm

    def __len__(self) -> int: return len(self.samples)

    def __getitem__(self, idx: int):

        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        return self.tfm(img), label


@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:

    model.eval()
    correct = 0
    total = 0

    for x, y in loader:

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x).argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())

    return correct / total if total else 0.0


def build_model(device: torch.device) -> nn.Module:

    try:

        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)

    except Exception: model = models.resnet18(pretrained=True)

    model.fc = nn.Linear(model.fc.in_features, 2)

    return model.to(device)


def append_result_block(lines: list[str]) -> None:

    block = ["", "=" * 72, *lines, "=" * 72]
    with open(RESULT_FILE, "a", encoding="utf-8") as f: f.write("\n".join(block) + "\n")


def main() -> None:

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    class0_paths = require_image_paths(CLASS0_DIR, NUM_IMAGES, "class0")
    class1_paths = require_image_paths(CLASS1_DIR, NUM_IMAGES, "class1")

    indices = list(range(NUM_IMAGES))
    random.shuffle(indices)
    n_val = max(1, int(NUM_IMAGES * VAL_RATIO))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = PairFolderDataset(class0_paths, class1_paths, train_idx, tfm)
    val_ds = PairFolderDataset(class0_paths, class1_paths, val_idx, tfm)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(
        f"device={device} | train_pairs={len(train_idx)} val_pairs={len(val_idx)} | "
        f"train_imgs={len(train_ds)} val_imgs={len(val_ds)} | "
        f"epochs={EPOCHS} batch={BATCH_SIZE} lr={LR}"
    )
    print(f"class0={CLASS0_NAME}\nclass1={CLASS1_NAME}")

    history: list[dict[str, float]] = []
    t0 = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):

        model.train()
        running_loss = 0.0
        n_batches = 0

        for x, y in train_loader:

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            n_batches += 1

        train_acc = accuracy(model, train_loader, device)
        val_acc = accuracy(model, val_loader, device)
        mean_loss = running_loss / max(n_batches, 1)
        row = {
            "epoch": float(epoch),
            "loss": mean_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}/{EPOCHS} | loss={mean_loss:.4f} | "
            f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f}"
        )

    elapsed = time.perf_counter() - t0
    summary = {
        "n_pairs": NUM_IMAGES,
        "n_train_pairs": len(train_idx),
        "n_val_pairs": len(val_idx),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "elapsed_sec": elapsed,
        "final_train_acc": history[-1]["train_acc"],
        "final_val_acc": history[-1]["val_acc"],
        "class0": CLASS0_NAME,
        "class1": CLASS1_NAME,
    }
    print("\n=== summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    append_result_block([
        f"ResNet18 {CLASS0_NAME} vs {CLASS1_NAME} | {time.strftime('%m-%d %H:%M')}",
        json.dumps(summary, ensure_ascii=False),
        "history:",
        json.dumps(history, ensure_ascii=False),
    ])
    print(f"wrote {RESULT_FILE}")


if __name__ == "__main__": main()
