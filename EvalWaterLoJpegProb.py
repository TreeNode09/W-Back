"""Evaluate WaterLo noise-rect (+ JPEG) for one manually installed checkpoint.

Copy the target run's G.pt / B.pt into Data/models/ yourself before running.
Set JPEG_PROB to match that training run so results.txt records it.

Output image folder is resolved only via Experiment.waterlo_out_dir / generateWaterLoImages
(plain_wl_jpeg_prob_{JPEG_PROB}_alpha_{ALPHA}).

I/O (relative to BASE_DIR):
  Read:
    models/G.pt, models/B.pt
    exp/plain/*                  — if REGENERATE_WL
    exp/plain_wl_jpeg_prob_.../* — watermarked images used for eval
  Write:
    exp/plain_wl_jpeg_prob_.../* — if REGENERATE_WL
    exp/results.txt
"""

from __future__ import annotations

import os

from Experiment import (
    BASE_DIR,
    EXP_DIR,
    RESULT_FILE,
    generateWaterLoImages,
    waterlo_jpeg_rect_eval,
    waterlo_out_dir,
)

# --- edit per run ---
JPEG_PROB = 0.0
ALPHA = 0.005
NUM_IMAGES = 500
SIDE_RATIO = 0.3
JPEG_QUALITIES: list[int | None] = [None, 70, 40]
REGENERATE_WL = True

PLAIN_DIR = os.path.join(EXP_DIR, "plain")
TITLE = f"PLAIN_WL_{ALPHA:g} | jpeg_prob={JPEG_PROB:g}"


def main() -> None:

    models_dir = os.path.join(BASE_DIR, "models")
    for name in ("G.pt", "B.pt"):

        path = os.path.join(models_dir, name)
        if not os.path.isfile(path): raise FileNotFoundError(f"missing {path}; copy the weights here first")

    if REGENERATE_WL:

        images_dir = generateWaterLoImages(ALPHA, NUM_IMAGES, PLAIN_DIR, jpeg_prob=JPEG_PROB,)
        print(f"=== embedded -> {images_dir} ===")

    else:

        images_dir = waterlo_out_dir(PLAIN_DIR, ALPHA, jpeg_prob=JPEG_PROB)
        if not os.path.isdir(images_dir):

            raise FileNotFoundError(f"missing {images_dir}; set REGENERATE_WL=True or generate images first")

    print(
        f"eval | jpeg_prob={JPEG_PROB:g} | images={images_dir} | "
        f"n={NUM_IMAGES} | side={SIDE_RATIO} | qualities={JPEG_QUALITIES}"
    )
    print(f"weights={models_dir} | results -> {RESULT_FILE}")

    waterlo_jpeg_rect_eval(NUM_IMAGES, images_dir, TITLE, SIDE_RATIO, JPEG_QUALITIES)
    print("done")


if __name__ == "__main__": main()
