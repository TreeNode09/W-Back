from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Lambda, ToPILImage, ToTensor

from WaterLo.src.loader import padding
from WaterLo.src.models import Generator, Bob
from WaterLo.src.utils import Models, crop_padding

# --- edit these in the IDE ---
BASE_DIR = r"D:\W-Back\Data"
# BASE_DIR = r"/workspace/W-Back/Data"

MODELS_DIR = os.path.join(BASE_DIR, "models")
OUT_DIR = os.path.join(BASE_DIR, "exp", "waterlo_structure")

IMAGE_PATHS = [os.path.join(BASE_DIR, "exp", "plain", "12.png")]    # Input image(s) to visualize
SCALE = 0.1 # Amplification for midgray view: 0.5 + SCALE * w


def load_models(models_dir: str) -> tuple[Models, torch.device]:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = Models(
        Generator(in_channels=3, out_channels=3).to(device),
        Bob(in_channels=3, out_channels=1).to(device),
    )

    models.load(models_dir)
    models.eval()

    return models, device


def tensor_to_uint8_rgb(t: torch.Tensor) -> Image.Image:
    """`t`: CHW float in [0, 1] -> PIL RGB."""

    arr = (t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)

    return Image.fromarray(arr, mode="RGB")


def residual_to_images(w: torch.Tensor, *, scale: float) -> dict[str, Image.Image]:
    """Build display variants from residual `w` (CHW).

    - midgray: 0.5 + scale * w
    - signed: per-channel min-max to [0,1]
    - magnitude: mean |w| over channels, then min-max (as RGB gray)
    """

    w = w.detach().float().cpu()

    mid = (0.5 + scale * w).clamp(0, 1)
    mid_img = tensor_to_uint8_rgb(mid)

    w_min = w.amin(dim=(-2, -1), keepdim=True)
    w_max = w.amax(dim=(-2, -1), keepdim=True)
    signed = (w - w_min) / (w_max - w_min + 1e-8)
    signed_img = tensor_to_uint8_rgb(signed)

    mag = w.abs().mean(dim=0, keepdim=True)
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    mag_rgb = mag.expand(3, -1, -1)
    mag_img = tensor_to_uint8_rgb(mag_rgb)

    return {"midgray": mid_img, "signed": signed_img, "magnitude": mag_img}


def visualize_one(image_path: str, models: Models, device: torch.device, out_dir: str, *, scale: float) -> None:

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]

    pil = Image.open(image_path).convert("RGB")
    ow, oh = pil.size

    tfm = Compose([ToTensor(), Lambda(lambda x: padding(x, size=512))])
    batch = tfm(pil).unsqueeze(0).to(device)

    with torch.no_grad():

        w = models.G(batch)
        w_crop, _ = crop_padding(w, ([ow], [oh]))
        img_crop, _ = crop_padding(batch, ([ow], [oh]))

    w0 = w_crop[0]
    variants = residual_to_images(w0, scale=scale)

    pil.save(os.path.join(out_dir, f"{stem}_original.png"))
    ToPILImage()(img_crop[0].cpu().clamp(0, 1)).save(os.path.join(out_dir, f"{stem}_padded_crop.png"))

    for name, im in variants.items():

        path = os.path.join(out_dir, f"{stem}_residual_{name}.png")
        im.save(path)
        print(f"saved {path}")

    npy_path = os.path.join(out_dir, f"{stem}_residual.npy")
    np.save(npy_path, w0.detach().cpu().numpy())
    print(f"saved {npy_path}")

    return


def main() -> None:

    if not os.path.isfile(os.path.join(MODELS_DIR, "G.pt")): raise FileNotFoundError(f"Missing G.pt under {MODELS_DIR}")

    models, device = load_models(MODELS_DIR)
    print(f"device={device} models={MODELS_DIR} out={OUT_DIR} scale={SCALE}")

    for path in IMAGE_PATHS:

        if not os.path.isfile(path): raise FileNotFoundError(path)

        print(f"\n=== {path} ===")
        visualize_one(path, models, device, OUT_DIR, scale=SCALE)

    return


if __name__ == "__main__": main()