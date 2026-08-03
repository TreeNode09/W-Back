"""Generate plain SD training images and retrain WaterLo with JPEG augmentation."""

import json
import os
import random
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

TEST_PROMPTS_FILE = os.path.join(EXP_DIR, "prompts.json")        # Experiment.py test set
TRAIN_PROMPTS_FILE = os.path.join(EXP_DIR, "prompts_train.json")
TEST_NUM_PROMPTS = 500   # match Experiment NUM_IMAGES
PROMPT_SEED = 2509

NUM_IMAGES = 5000   # 7,500 exceeds the HF dataset capacity when margin=512
PRC_DIR = os.path.join(EXP_DIR, "prc_sd_train")
DATASET_DIR = os.path.join(EXP_DIR, "waterlo_train")

# JPEG aug: probability of applying JPEG when compression=True (old default ≈ 0.8)
JPEG_PROB = 0.9


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


def _save_json(path: str, data: list[str]) -> None:

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def load_train_prompts() -> list[str]:
    """Fetch test+train prompts in one shot, split with no string overlap."""

    if os.path.isfile(TRAIN_PROMPTS_FILE):
        with open(TRAIN_PROMPTS_FILE, "r", encoding="utf-8") as f:
            prompts = json.load(f)
        if len(prompts) >= NUM_IMAGES:
            print(f"train prompts: reuse {TRAIN_PROMPTS_FILE} ({len(prompts)} total)")
            return prompts[:NUM_IMAGES]

    n_test = TEST_NUM_PROMPTS
    if os.path.isfile(TEST_PROMPTS_FILE):
        with open(TEST_PROMPTS_FILE, "r", encoding="utf-8") as f:
            test_prompts = json.load(f)
        n_test = max(n_test, len(test_prompts))
        print(f"test prompts: reuse {TEST_PROMPTS_FILE} ({len(test_prompts)} total)")
    else:
        test_prompts = None

    # Extra margin: HF set has duplicate Prompt strings, and a random draw of
    # (n_test + NUM_IMAGES) often still collides with an existing test file.
    margin = max(512, n_test)
    total = n_test + NUM_IMAGES + margin
    print(f"fetch {total} prompts from {PROMPT_DATASET} (seed={PROMPT_SEED}, margin={margin})")
    random.seed(PROMPT_SEED)
    all_prompts = getPrompts(PROMPT_DATASET, total, out_path=None)

    if test_prompts is None:
        test_prompts = all_prompts[:TEST_NUM_PROMPTS]
        train_prompts = all_prompts[TEST_NUM_PROMPTS:TEST_NUM_PROMPTS + NUM_IMAGES]
        _save_json(TEST_PROMPTS_FILE, test_prompts)
        print(f"wrote test prompts -> {TEST_PROMPTS_FILE}")
    else:
        banned = set(test_prompts)
        train_prompts = [p for p in all_prompts if p not in banned][:NUM_IMAGES]
        if len(train_prompts) < NUM_IMAGES:
            raise RuntimeError(
                f"only {len(train_prompts)}/{NUM_IMAGES} non-overlapping train prompts "
                f"after fetching {total}; increase margin"
            )

    _save_json(TRAIN_PROMPTS_FILE, train_prompts)
    print(f"wrote train prompts -> {TRAIN_PROMPTS_FILE} ({len(train_prompts)})")
    return train_prompts


if __name__ == "__main__":

    os.makedirs(EXP_DIR, exist_ok=True)

    n_existing = count_images(PRC_DIR)
    if n_existing >= NUM_IMAGES:
        print(f"skip generation: {n_existing} images already in {PRC_DIR}")
    else:
        if n_existing > 0:
            print(f"warning: only {n_existing}/{NUM_IMAGES} images in {PRC_DIR}, regenerating all")

        prompts = load_train_prompts()

        print(f"=== generate {NUM_IMAGES} plain SD images ===")
        os.makedirs(PRC_DIR, exist_ok=True)
        applyPRC(
            BASE_DIR,
            "",
            MODEL_ID,
            prompts,
            out_path=PRC_DIR,
            on_progress=make_progress_logger("plain_sd"),
        )

    print("=== build WaterLo train/valid split ===")
    n_train, n_valid = makeWaterLoDataset(PRC_DIR, DATASET_DIR)
    print(f"dataset ready: train={n_train}, valid={n_valid} -> {DATASET_DIR}")

    print(f"=== train WaterLo (compression=True, jpeg_prob={JPEG_PROB}) ===")
    trainWaterLo(
        DATASET_DIR,
        WEIGHTS_DIR,
        alpha=0.005,
        lambd=4,
        epochs=32,
        batch_size=8,
        loss="ssim",
        compression=True,
        jpeg_prob=JPEG_PROB,
    )
    print(f"done: weights saved to {WEIGHTS_DIR}/G.pt and {WEIGHTS_DIR}/B.pt")
