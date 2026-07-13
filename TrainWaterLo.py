"""Generate plain SD training images and retrain WaterLo with JPEG augmentation enabled."""

import json
import os
import time

from Interface import applyPRC, getPrompts, makeWaterLoDataset, trainWaterLo

# --- edit paths on the remote machine if needed ---
BASE_DIR = r"/workspace/W-Back/Data"
# BASE_DIR = r"D:\W-Back\Data"

EXP_DIR = os.path.join(BASE_DIR, "exp")
MODEL_CACHE_DIR = os.path.join(BASE_DIR, "models")
WEIGHTS_DIR = MODEL_CACHE_DIR

MODEL_ID = "sd-research/stable-diffusion-2-1-base"
PROMPT_DATASET = "Gustavosta/Stable-Diffusion-Prompts"

NUM_IMAGES = 7500
PLAIN_DIR = os.path.join(EXP_DIR, "plain_sd_train")
DATASET_DIR = os.path.join(EXP_DIR, "waterlo_train")
PROMPTS_FILE = os.path.join(EXP_DIR, "prompts.json")


def make_progress_logger(title: str):

    start_time = time.perf_counter()
    total = None

    def on_progress(current: int, total_count: int) -> None:

        nonlocal total
        if total is None:
            total = total_count

        elapsed = time.perf_counter() - start_time
        avg = elapsed / current
        it_per_sec = current / elapsed if elapsed > 0 else 0.0
        eta = (total - current) * avg

        print(
            f"[{title}] {current}/{total} | "
            f"elapsed={elapsed:.2f}s | avg={avg:.2f}s/img | "
            f"speed={it_per_sec:.2f} img/s | eta={eta:.2f}s"
        )

    return on_progress


def count_images(images_dir: str) -> int:

    if not os.path.isdir(images_dir):
        return 0

    exts = {".png", ".jpg", ".jpeg", ".JPEG"}
    return sum(
        1 for name in os.listdir(images_dir)
        if os.path.splitext(name)[1] in exts and os.path.isfile(os.path.join(images_dir, name))
    )


def load_prompts() -> list[str]:

    if os.path.isfile(PROMPTS_FILE):
        with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
            prompts = json.load(f)
        if len(prompts) >= NUM_IMAGES:
            print(f"reuse prompts from {PROMPTS_FILE} ({len(prompts)} total)")
            return prompts[:NUM_IMAGES]

    os.makedirs(EXP_DIR, exist_ok=True)
    prompts = getPrompts(PROMPT_DATASET, NUM_IMAGES, EXP_DIR)
    print(f"fetched {len(prompts)} prompts into {PROMPTS_FILE}")
    return prompts


if __name__ == "__main__":

    os.makedirs(EXP_DIR, exist_ok=True)

    n_existing = count_images(PLAIN_DIR)
    if n_existing >= NUM_IMAGES:
        print(f"skip generation: {n_existing} images already in {PLAIN_DIR}")
    else:
        if n_existing > 0:
            print(f"warning: only {n_existing}/{NUM_IMAGES} images in {PLAIN_DIR}, regenerating all")

        print(f"=== generate {NUM_IMAGES} plain SD images ===")
        prompts = load_prompts()
        os.makedirs(PLAIN_DIR, exist_ok=True)
        applyPRC(
            BASE_DIR,
            "",
            MODEL_ID,
            prompts,
            out_path=PLAIN_DIR,
            watermark=False,
            on_progress=make_progress_logger("plain_sd"),
        )

    print("=== build WaterLo train/valid split ===")
    n_train, n_valid = makeWaterLoDataset(PLAIN_DIR, DATASET_DIR)
    print(f"dataset ready: train={n_train}, valid={n_valid} -> {DATASET_DIR}")

    print("=== train WaterLo (compression=True) ===")
    trainWaterLo(
        DATASET_DIR,
        WEIGHTS_DIR,
        alpha=0.005,
        lambd=4,
        epochs=32,
        batch_size=8,
        loss="ssim",
        compression=True,
    )
    print(f"done: weights saved to {WEIGHTS_DIR}/G.pt and {WEIGHTS_DIR}/B.pt")
