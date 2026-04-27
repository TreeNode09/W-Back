import os
import random
import json
import time
from PIL import Image

from Interface import *

BASE_DIR = r"D:/W-Back/Data"
EXP_DIR = os.path.join(BASE_DIR, "exp")
PRC_IMG_DIR = os.path.join(EXP_DIR, "prc_images")
PROMPTS_FILE = os.path.join(EXP_DIR, "prompts.json")

PROMPT_DATASET = "Gustavosta/Stable-Diffusion-Prompts"
MODEL_ID = "sd-research/stable-diffusion-2-1-base"

def make_progress_logger():

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
            f"[PRC] {current}/{total} | "
            f"elapsed={elapsed:.2f}s | avg={avg:.2f}s/img | "
            f"speed={it_per_sec:.2f} img/s | eta={eta:.2f}s"
        )

    return on_progress

def preparePrompts(num_prompts: int, seed: int = 42) -> list[str]:

    os.makedirs(EXP_DIR, exist_ok=True)
    random.seed(seed)
    return getPrompts(PROMPT_DATASET, num_prompts, EXP_DIR)


def generatePRCImages(key_id: str, *, prompts_path: str = PROMPTS_FILE) -> None:

    with open(prompts_path, "r", encoding="utf-8") as f: prompts = json.load(f)

    os.makedirs(PRC_IMG_DIR, exist_ok=True)
    applyPRC(BASE_DIR, key_id, MODEL_ID, prompts,
        message_fn=lambda i: i, out_path=PRC_IMG_DIR, watermark=True, on_progress=make_progress_logger())

    return


def detectPRCImages(key_id: str, num_images: int, *, images_dir: str = PRC_IMG_DIR) -> dict[str, float]:

    names = [n for n in os.listdir(images_dir) if os.path.splitext(n)[1].lower() in {".png", ".jpg", ".jpeg"}]
    names = sorted(names, key=lambda n: int(os.path.splitext(n)[0]))
    names = names[:num_images]
    images = [Image.open(os.path.join(images_dir, n)).convert("RGB") for n in names]

    results = decodePRC(BASE_DIR, key_id, MODEL_ID, images, on_progress=make_progress_logger())

    detect_acc = sum(1 for detect, _ in results if detect) / len(results)

    msg_width = max(1, (len(results) - 1).bit_length())
    decode_hits = 0
    decode_valid = 0
    for i, (_, bits) in enumerate(results):
        if bits is None:
            continue
        decode_valid += 1
        decoded_value = int(bits[:msg_width], 2)
        if decoded_value == i:
            decode_hits += 1

    decode_acc_all = decode_hits / len(results)
    decode_acc_valid = (decode_hits / decode_valid) if decode_valid > 0 else 0.0

    metrics = {
        "detect_accuracy": detect_acc,
        "decode_accuracy_all": decode_acc_all,
        "decode_accuracy_valid": decode_acc_valid,
        "decode_valid_rate": (decode_valid / len(results)) if results else 0.0,
    }
    print(metrics)
    return metrics


if __name__ == "__main__":

    prompts = preparePrompts(1000)
    print(f"Prepared {len(prompts)} prompt(s) at {PROMPTS_FILE}")

    generatePRCImages("")