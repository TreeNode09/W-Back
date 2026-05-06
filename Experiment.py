import os
import random
import json
import time
import io
import cv2
from PIL import Image, ImageFilter
import math
import numpy as np
from sklearn.metrics import roc_auc_score

from Interface import *

BASE_DIR = r"D:/W-Back/Data"
# BASE_DIR = r"/workspace/W-Back/Data"
EXP_DIR = os.path.join(BASE_DIR, "exp")

PROMPTS_FILE = os.path.join(EXP_DIR, "prompts.json")
RESULT_FILE = os.path.join(EXP_DIR, "results.txt")

PROMPT_DATASET = "Gustavosta/Stable-Diffusion-Prompts"
MODEL_ID = "sd-research/stable-diffusion-2-1-base"

def compress_image_jpeg(image: Image.Image, quality: int) -> Image.Image:
 
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    out = Image.open(buf).convert("RGB")
    out.load()
    buf.close()

    return out


def compress_images_jpeg(images: list[Image.Image], quality: int) -> list[Image.Image]:

    return [compress_image_jpeg(im, quality) for im in images]


def blur_image_gaussian(image: Image.Image, radius: float) -> Image.Image:

    if radius is None or radius <= 0: return image

    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def blur_images_gaussian(images: list[Image.Image], radius: float) -> list[Image.Image]:

    if radius is None or radius <= 0: return images

    return [blur_image_gaussian(im, radius) for im in images]


def attack_image_noise_rect(image: Image.Image, side_ratio: float, seed: int) -> tuple[Image.Image, np.ndarray]:

    arr = np.array(image.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    if side_ratio <= 0 or side_ratio >= 1:
        gt = np.ones((h, w), dtype=np.uint8)
        return Image.fromarray(arr, mode="RGB"), gt

    rw = max(1, int(w * side_ratio))
    rh = max(1, int(h * side_ratio))

    rng = random.Random(seed)
    x0 = rng.randint(0, max(0, w - rw))
    y0 = rng.randint(0, max(0, h - rh))
    x1, y1 = x0 + rw, y0 + rh

    noise = np.random.default_rng(seed).integers(0, 256, size=(rh, rw, 3), dtype=np.uint8)
    arr[y0:y1, x0:x1, :] = noise

    gt = np.ones((h, w), dtype=np.uint8)
    gt[y0:y1, x0:x1] = 0

    return Image.fromarray(arr, mode="RGB"), gt


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


def sorted_image_names(images_dir: str, num_images: int) -> list[str]:

    names = [n for n in os.listdir(images_dir) if os.path.splitext(n)[1].lower() in {".png", ".jpg", ".jpeg"}]
    names = sorted(names, key=lambda n: int(os.path.splitext(n)[0]))[:num_images]
    return names


def load_sorted_rgb_images(images_dir: str, num_images: int) -> list[Image.Image]:

    names = sorted_image_names(images_dir, num_images)
    return [Image.open(os.path.join(images_dir, name)).convert("RGB") for name in names]


def preparePrompts(numprompts: int, seed: int = 42) -> list[str]:

    os.makedirs(EXP_DIR, exist_ok=True)
    random.seed(seed)
    return getPrompts(PROMPT_DATASET, numprompts, EXP_DIR)


def generatePRCImages(num: int, watermark: bool, key_id: str | None = None) -> None:

    with open(PROMPTS_FILE, "r", encoding="utf-8") as f: prompts = json.load(f)
    if watermark and not key_id: raise ValueError("key_id is required when watermark=True")

    out_dir_name = "prc" if watermark else "plain"
    out_path = os.path.join(EXP_DIR, out_dir_name)
    os.makedirs(out_path, exist_ok=True)

    applyPRC(BASE_DIR, key_id or "", MODEL_ID, prompts[:num], message_fn=lambda i: str(i), 
        out_path=out_path, watermark=watermark, on_progress=make_progress_logger(out_dir_name))

    return


def detectPRCImages(key_id: str, num_images: int, images_dir: str, *,
    jpeg_quality: int | None = None, blur_radius: float | None = None) -> dict[str, float]:

    images = load_sorted_rgb_images(images_dir, num_images)

    if blur_radius is not None: images = blur_images_gaussian(images, blur_radius)

    if jpeg_quality is not None: images = compress_images_jpeg(images, jpeg_quality)

    results = decodePRC(BASE_DIR, key_id, MODEL_ID, images, on_progress=make_progress_logger("decode_prc"))

    detect_acc = sum(1 for detect, _ in results if detect) / len(results)

    decode_hits = 0
    decode_valid = 0

    for i, (_, bits) in enumerate(results):

        if bits is None: continue

        decoded_text = decodeBitsToText(bits)
        if decoded_text is None: continue

        decode_valid += 1
        if decoded_text == str(i): decode_hits += 1

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


def generateWaterLoImages(alpha: float, num: int, in_path: str) -> None:

    images = load_sorted_rgb_images(in_path, num)

    in_tag = os.path.basename(os.path.normpath(in_path))
    out_path = os.path.join(EXP_DIR, f"{in_tag}_wl_{alpha:g}")
    os.makedirs(out_path, exist_ok=True)

    applyWaterLo(images, BASE_DIR, alpha=alpha, out_path=out_path, on_progress=make_progress_logger(f"wl_{alpha:g}"))

    return


def generateTreeRingImages(num: int, key_id: str, *, out_dir_name: str = "treering", key_gen_seed: int = 0) -> None:

    out_path = os.path.join(EXP_DIR, out_dir_name)

    with open(PROMPTS_FILE, "r", encoding="utf-8") as f: prompts = json.load(f)[:num]

    applyTreeRing(
        BASE_DIR, MODEL_ID, prompts, out_path=out_path, key_id=key_id, key_gen_seed=key_gen_seed,
        on_progress=make_progress_logger("treering")
    )

    return


def detectTreeRingImages(key_id: str, num_images: int, images_dir: str, *, keys_path: str | None = None,
    jpeg_quality: int | None = None, blur_radius: float | None = None) -> dict[str, float]:

    images = load_sorted_rgb_images(images_dir, num_images)

    if blur_radius is not None: images = blur_images_gaussian(images, blur_radius)

    if jpeg_quality is not None: images = compress_images_jpeg(images, jpeg_quality)

    rows = detectTreeRing(BASE_DIR, MODEL_ID, images, key_id, keys_path=keys_path,
        on_progress=make_progress_logger("TreeRing_detect"))

    dists = [d for d, _ in rows]
    detects = [det for _, det in rows]

    metrics = {
        "detect_accuracy": float(sum(1 for d in detects if d) / len(detects)) if detects else 0.0,
        "mean_dist": float(np.mean(dists)) if dists else float("nan"),
        "std_dist": float(np.std(dists)) if dists else float("nan"),
        "count": float(len(detects)),
    }
    print(metrics)

    return metrics


def treering_jpeg_eval(key_id: str, num_images: int, images_dir: str, title: str, jpeg_qualities: list[int], *,
    keys_path: str | None = None) -> dict[int, dict[str, float]]:

    block_lines = [
        "", "=" * 72, f"{title} TreeRing under JPEG | {time.strftime('%m-%d %H:%M')}",
        f"qualities={jpeg_qualities}", ""
    ]

    all_metrics: dict[int, dict[str, float]] = {}
    for q in jpeg_qualities:

        print(f"\n=== TreeRing JPEG quality={q} ===\n")

        m = detectTreeRingImages(key_id, num_images, images_dir, keys_path=keys_path, jpeg_quality=q)

        all_metrics[q] = m
        block_lines.append(f"q={q} {json.dumps(m, ensure_ascii=False)}")

    block_lines.append("=" * 72)
    with open(RESULT_FILE, "a", encoding="utf-8") as f: f.write("\n".join(block_lines))

    return all_metrics


def treering_blur_eval(key_id: str, num_images: int, images_dir: str, title: str, blur_radii: list[float], *,
    keys_path: str | None = None) -> dict[float, dict[str, float]]:

    block_lines = ["", "=" * 72, f"{title} TreeRing under BLUR | {time.strftime('%m-%d %H:%M')}", f"radii={blur_radii}", ""]

    all_metrics: dict[float, dict[str, float]] = {}
    for r in blur_radii:

        print(f"\n=== TreeRing blur radius={r} ===\n")

        m = detectTreeRingImages(key_id, num_images, images_dir, keys_path=keys_path, blur_radius=r)

        all_metrics[r] = m
        block_lines.append(f"blur_r={r} {json.dumps(m, ensure_ascii=False)}")

    block_lines.append("=" * 72)
    with open(RESULT_FILE, "a", encoding="utf-8") as f: f.write("\n".join(block_lines))

    return all_metrics


def prc_jpeg_eval(key_id: str, num_images: int, images_dir: str, title: str,
    jpeg_qualities: list[int]) -> dict[int, dict[str, float]]:

    block_lines = ["", "=" * 72, f"{title} under JPEG | {time.strftime('%m-%d %H:%M')}", f"qualities={jpeg_qualities}", ""]

    all_metrics: dict[int, dict[str, float]] = {}
    for q in jpeg_qualities:

        print(f"\n=== JPEG quality={q} ===\n")

        m = detectPRCImages(key_id, num_images, images_dir, jpeg_quality=q)

        all_metrics[q] = m
        block_lines.append(f"q={q} {json.dumps(m, ensure_ascii=False)}")

    block_lines.append("=" * 72)
    block_text = "\n".join(block_lines)

    with open(RESULT_FILE, "a", encoding="utf-8") as f: f.write(block_text)

    return all_metrics


def psnr_rgb(ref: Image.Image, dist: Image.Image, *, data_range: float = 255.0) -> float:

    a = np.asarray(ref.convert("RGB"), dtype=np.float64)
    b = np.asarray(dist.convert("RGB"), dtype=np.float64)

    if a.shape != b.shape: raise ValueError(f"PSNR shape mismatch {a.shape} vs {b.shape}")

    mse = float(np.mean((a - b) ** 2))
    if mse <= 0.0: return float("inf")

    return float(10.0 * math.log10((data_range ** 2) / mse))


def psnr_eval(num_images: int, ref_path: str, dist_path: str, title: str) -> dict[str, float]:

    names = sorted_image_names(ref_path, num_images)

    psnrs: list[float] = []
    for n in names:

        p_ref = os.path.join(ref_path, n)
        p_dist = os.path.join(dist_path, n)
        if not os.path.isfile(p_dist): raise FileNotFoundError(f"missing paired image: {p_dist}")

        ref = Image.open(p_ref).convert("RGB")
        dist = Image.open(p_dist).convert("RGB")
        psnrs.append(psnr_rgb(ref, dist))

    arr = np.array(psnrs, dtype=np.float64)
    finite = arr[np.isfinite(arr)]

    metrics: dict[str, float] = {
        "mean": float(np.mean(finite)) if finite.size else float("nan"),
        "std": float(np.std(finite)) if finite.size else float("nan"),
        "min": float(np.min(finite)) if finite.size else float("nan"),
        "max": float(np.max(finite)) if finite.size else float("nan"),
        "count": float(len(psnrs)),
        "inf_count": float(np.sum(np.isposinf(arr)))
    }

    print(metrics)

    finite_metrics: dict[str, float | None] = {}
    for k, v in metrics.items():

        if isinstance(v, float) and math.isfinite(v): finite_metrics[k] = v
        else: finite_metrics[k] = None

    block_lines = [
        "", "=" * 72, f"{title} PSNR | {time.strftime('%m-%d %H:%M')}", 
        json.dumps(finite_metrics, ensure_ascii=False), "=" * 72
    ]

    with open(RESULT_FILE, "a", encoding="utf-8") as f: f.write("\n".join(block_lines))

    return metrics


def fid_eval(num_images: int, real_dir: str, gen_dir: str, title: str, batch_size: int = 8) -> dict[str, float]:

    import torch
    from torchmetrics.image.fid import FrechetInceptionDistance

    real_names = sorted_image_names(real_dir, num_images)
    gen_names = sorted_image_names(gen_dir, num_images)

    n = min(len(real_names), len(gen_names))
    if n < 2: raise ValueError(f"FID needs at least 2 images per folder")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fid = FrechetInceptionDistance().to(device)

    def batch_to_uint8(pils: list[Image.Image]) -> torch.Tensor:

        arrs = [np.asarray(p.convert("RGB"), dtype=np.uint8) for p in pils]
        x = np.stack(arrs, axis=0)
        return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()

    bi = 0
    on_batch = make_progress_logger("fid_batches")
    total_batches = (n + batch_size - 1) // batch_size * 2

    for start in range(0, n, batch_size):

        chunk = [Image.open(os.path.join(real_dir, fn)).convert("RGB") for fn in real_names[start : start + batch_size]]
        fid.update(batch_to_uint8(chunk).to(device), real=True)
        bi += 1
        on_batch(bi, total_batches)

    for start in range(0, n, batch_size):

        chunk = [Image.open(os.path.join(gen_dir, fn)).convert("RGB") for fn in gen_names[start : start + batch_size]]
        fid.update(batch_to_uint8(chunk).to(device), real=False)
        bi += 1
        on_batch(bi, total_batches)

    fid_score = float(fid.compute())

    metrics: dict[str, float] = {"fid": fid_score, "n": float(n)}

    print(metrics)

    block_lines = [
        "", "=" * 72, f"{title} FID | {time.strftime('%m-%d %H:%M')}",
        json.dumps(metrics, ensure_ascii=False), "=" * 72
    ]

    with open(RESULT_FILE, "a", encoding="utf-8") as f: f.write("\n".join(block_lines))

    return metrics


def prc_blur_eval(key_id: str, num_images: int, images_dir: str, title: str,
    blur_radii: list[float],) -> dict[float, dict[str, float]]:

    block_lines = ["", "=" * 72, f"{title} under BLUR | {time.strftime('%m-%d %H:%M')}", f"radii={blur_radii}", ""]
    all_metrics: dict[float, dict[str, float]] = {}
    for r in blur_radii:

        print(f"\n=== Gaussian blur radius={r} ===\n")

        m = detectPRCImages(key_id, num_images, images_dir, blur_radius=r)

        all_metrics[r] = m
        block_lines.append(f"blur_r={r} {json.dumps(m, ensure_ascii=False)}")

    block_lines.append("=" * 72)
    block_text = "\n".join(block_lines)

    with open(RESULT_FILE, "a", encoding="utf-8") as f: f.write(block_text)

    return all_metrics


def waterlo_rect_eval(num_images: int, images_dir: str, title: str, side_ratio: float, *,
    threshold: float = 0.5) -> dict[str, float]:

    originals = load_sorted_rgb_images(images_dir, num_images)

    attacked, gt_masks = [], []
    for i, im in enumerate(originals):

        atk, gt = attack_image_noise_rect(im, side_ratio=side_ratio, seed=42+i)
        attacked.append(atk)
        gt_masks.append(gt)

    maps, preds = detectWaterLo(
        attacked, BASE_DIR, on_progress=make_progress_logger("detect_waterlo_rect"),
    )

    # for i, mp in enumerate(maps):

    #     bgr = cv2.cvtColor(np.array(mp.convert("RGB")), cv2.COLOR_RGB2BGR)
    #     cv2.imshow("waterlo_heatmap", bgr)
    #     key = cv2.waitKey(0)
    #     if key == 27: break

    # cv2.destroyAllWindows()

    # return

    ious, f1s, aucs = [], [], []
    for pred, gt in zip(preds, gt_masks):

        score_map = np.asarray(pred, dtype=np.float32)
        pred_mask = (score_map >= threshold).astype(np.uint8)
        pred_b = pred_mask.astype(bool)
        gt_b = gt.astype(bool)
        tp = np.logical_and(pred_b, gt_b).sum()
        fp = np.logical_and(pred_b, ~gt_b).sum()
        fn = np.logical_and(~pred_b, gt_b).sum()
        denom_iou = tp + fp + fn
        iou = float(tp / denom_iou) if denom_iou > 0 else 0.0
        denom_f1 = 2 * tp + fp + fn
        f1 = float((2 * tp) / denom_f1) if denom_f1 > 0 else 0.0
        ious.append(iou)
        f1s.append(f1)

        s = score_map.astype(np.float64).ravel()
        y = gt.astype(np.uint8).ravel()

        if y.any() and not y.all(): auc = float(roc_auc_score(y, s))
        else: auc = 0.5

        aucs.append(auc)

    metrics = {
        "count": float(len(preds)),
        "side_ratio": float(side_ratio),
        "threshold": float(threshold),
        "iou": float(np.mean(ious)) if ious else 0.0,
        "f1": float(np.mean(f1s)) if f1s else 0.0,
        "auc": float(np.mean(aucs)) if aucs else 0.5
    }
    
    print(metrics)

    block = [
        "", "=" * 72, f"{title} WaterLo Noise-Rect | {time.strftime('%m-%d %H:%M')}",
        f"{json.dumps(metrics, ensure_ascii=False)}", "=" * 72
    ]

    with open(RESULT_FILE, "a", encoding="utf-8") as f: f.write("\n".join(block))

    return metrics


def write_timing_block(title: str, meta: dict[str, object], stats: dict[str, dict[str, float]]) -> None:

    block = [
        "", "=" * 72, f"{title} | timing | {time.strftime('%m-%d %H:%M')}",
        json.dumps(meta, ensure_ascii=False), json.dumps(stats, ensure_ascii=False), "=" * 72
    ]

    with open(RESULT_FILE, "a", encoding="utf-8") as f: f.write("\n".join(block))

    print(f"\n==={title}===\n")
    print(json.dumps(stats, ensure_ascii=False))

    return


if __name__ == "__main__":

    NUM_IMAGES = 1
    KEY_ID = "45d7645463e6b61b8f0695b9284fac92"
    TREERING_KEY_ID = "8c86b304d80dd8c03a1390a49d25abae"


    # # Generate test image sets
    # ALPHAS = [0.002, 0.005, 0.01, 0.05]

    # prompts = preparePrompts(NUM_IMAGES, seed=42)
    # print(f"Prepared {len(prompts)} prompt(s) at {PROMPTS_FILE}")

    # # 1. no PRC, no WaterLo
    # print("\n=== PLAIN IMAGE ===\n")
    # generatePRCImages(NUM_IMAGES, watermark=False, key_id=None)

    # # 2. PRC, no WaterLo
    # print("\n=== PRC ===\n")
    # generatePRCImages(NUM_IMAGES, watermark=True, key_id=KEY_ID)

    # # 3. no PRC + WaterLo(alpha)
    # for alpha in ALPHAS:

    #     print(f"\n=== WATERLO alpha={alpha} ===\n")
    #     generateWaterLoImages(alpha, NUM_IMAGES, os.path.join(EXP_DIR, "plain"))

    # # 4. PRC + WaterLo(alpha)
    # for alpha in ALPHAS:

    #     print(f"\n=== PRC + WATERLO alpha={alpha} ===\n")
    #     generateWaterLoImages(alpha, NUM_IMAGES, os.path.join(EXP_DIR, "prc"))


    # # Generate Tree-Ring images
    # generateTreeRingImages(NUM_IMAGES, TREERING_KEY_ID)


    # # PRC evaluation under JPEG compression
    # JPEG_QUALITIES = [100, 70, 40]
    # prc_jpeg_eval(KEY_ID, NUM_IMAGES, os.path.join(EXP_DIR, "prc"), "PRC", JPEG_QUALITIES)
    # prc_jpeg_eval(KEY_ID, NUM_IMAGES, os.path.join(EXP_DIR, "prc_wl_0.002"), "PRC_WL_0.002", JPEG_QUALITIES)
    # prc_jpeg_eval(KEY_ID, NUM_IMAGES, os.path.join(EXP_DIR, "prc_wl_0.005"), "PRC_WL_0.005", JPEG_QUALITIES)
    # prc_jpeg_eval(KEY_ID, NUM_IMAGES, os.path.join(EXP_DIR, "prc_wl_0.01"), "PRC_WL_0.01", JPEG_QUALITIES)
    # prc_jpeg_eval(KEY_ID, NUM_IMAGES, os.path.join(EXP_DIR, "prc_wl_0.05"), "PRC_WL_0.05", JPEG_QUALITIES)

    # treering_jpeg_eval(TREERING_KEY_ID, NUM_IMAGES, os.path.join(EXP_DIR, "treering"), "TR", JPEG_QUALITIES)

    # # PRC evaluation under Gaussian blur
    # BLUR_RADII = [1, 2, 4]
    # prc_blur_eval(KEY_ID, NUM_IMAGES, os.path.join(EXP_DIR, "prc"), "PRC", BLUR_RADII)
    # prc_blur_eval(KEY_ID, NUM_IMAGES, os.path.join(EXP_DIR, "prc_wl_0.005"), "PRC_WL_0.005", BLUR_RADII)
    # prc_blur_eval(KEY_ID, NUM_IMAGES, os.path.join(EXP_DIR, "prc_wl_0.01"), "PRC_WL_0.01", BLUR_RADII)
    # prc_blur_eval(KEY_ID, NUM_IMAGES, os.path.join(EXP_DIR, "prc_wl_0.05"), "PRC_WL_0.05", BLUR_RADII)
    # prc_blur_eval(KEY_ID, NUM_IMAGES, os.path.join(EXP_DIR, "prc_wl_0.1"), "PRC_WL_0.1", BLUR_RADII)
    # prc_blur_eval(KEY_ID, NUM_IMAGES, os.path.join(EXP_DIR, "prc_wl_0.2"), "PRC_WL_0.2", BLUR_RADII)

    # treering_blur_eval(TREERING_KEY_ID, NUM_IMAGES, os.path.join(EXP_DIR, "treering"), "TR", BLUR_RADII)

    # # PRC evaluation with plain images in FID
    # fid_eval(NUM_IMAGES, os.path.join(EXP_DIR, "plain"), os.path.join(EXP_DIR, "prc"), "FID plain vs prc")


    # # WaterLo evaluation of image quality
    # psnr_eval(NUM_IMAGES, os.path.join(EXP_DIR, "plain"), os.path.join(EXP_DIR, "plain_wl_0.002"), "PLAIN vs WL_0.002")
    # psnr_eval(NUM_IMAGES, os.path.join(EXP_DIR, "plain"), os.path.join(EXP_DIR, "plain_wl_0.005"), "PLAIN vs WL_0.005")
    # psnr_eval(NUM_IMAGES, os.path.join(EXP_DIR, "plain"), os.path.join(EXP_DIR, "plain_wl_0.01"), "PLAIN vs WL_0.01")
    # psnr_eval(NUM_IMAGES, os.path.join(EXP_DIR, "plain"), os.path.join(EXP_DIR, "plain_wl_0.05"), "PLAIN vs WL_0.05")
    # psnr_eval(NUM_IMAGES, os.path.join(EXP_DIR, "plain"), os.path.join(EXP_DIR, "plain_wl_0.1"), "PLAIN vs WL_0.1")
    # psnr_eval(NUM_IMAGES, os.path.join(EXP_DIR, "plain"), os.path.join(EXP_DIR, "plain_wl_0.2"), "PLAIN vs WL_0.2")
    # psnr_eval(NUM_IMAGES, os.path.join(EXP_DIR, "prc"), os.path.join(EXP_DIR, "prc_wl_0.002"), "PRC vs WL_0.002")
    # psnr_eval(NUM_IMAGES, os.path.join(EXP_DIR, "prc"), os.path.join(EXP_DIR, "prc_wl_0.005"), "PRC vs WL_0.005")
    # psnr_eval(NUM_IMAGES, os.path.join(EXP_DIR, "prc"), os.path.join(EXP_DIR, "prc_wl_0.01"), "PRC vs WL_0.01")
    # psnr_eval(NUM_IMAGES, os.path.join(EXP_DIR, "prc"), os.path.join(EXP_DIR, "prc_wl_0.05"), "PRC vs WL_0.05")
    # psnr_eval(NUM_IMAGES, os.path.join(EXP_DIR, "prc"), os.path.join(EXP_DIR, "prc_wl_0.1"), "PRC vs WL_0.1")
    # psnr_eval(NUM_IMAGES, os.path.join(EXP_DIR, "prc"), os.path.join(EXP_DIR, "prc_wl_0.2"), "PRC vs WL_0.2")


    # # WaterLo evaluation after applying noise rectangle (optional: ``jpeg_quality=70`` etc.; ``101`` = no JPEG in detector)
    # SIDE_RATIOS = 0.3
    # waterlo_rect_eval(NUM_IMAGES, os.path.join(EXP_DIR, "plain"), "PLAIN", SIDE_RATIOS)
    # waterlo_rect_eval(NUM_IMAGES, os.path.join(EXP_DIR, "plain_wl_0.002"), "PLAIN_WL_0.002", SIDE_RATIOS)
    # waterlo_rect_eval(NUM_IMAGES, os.path.join(EXP_DIR, "plain_wl_0.005"), "PLAIN_WL_0.005", SIDE_RATIOS)
    # waterlo_rect_eval(NUM_IMAGES, os.path.join(EXP_DIR, "plain_wl_0.01"), "PLAIN_WL_0.01", SIDE_RATIOS)
    # waterlo_rect_eval(NUM_IMAGES, os.path.join(EXP_DIR, "prc"), "PRC", SIDE_RATIOS)
    # waterlo_rect_eval(NUM_IMAGES, os.path.join(EXP_DIR, "prc_wl_0.002"), "PRC_WL_0.002", SIDE_RATIOS)
    # waterlo_rect_eval(NUM_IMAGES, os.path.join(EXP_DIR, "prc_wl_0.005"), "PRC_WL_0.005", SIDE_RATIOS)
    # waterlo_rect_eval(NUM_IMAGES, os.path.join(EXP_DIR, "prc_wl_0.01"), "PRC_WL_0.01", SIDE_RATIOS)


    # # PRC & Tree-Ring evaluation on plain images
    # prc_jpeg_eval(KEY_ID, NUM_IMAGES, os.path.join(EXP_DIR, "plain"), "PLAIN", [100])
    # treering_jpeg_eval(TREERING_KEY_ID, NUM_IMAGES, os.path.join(EXP_DIR, "plain"), "PLAIN", [100])


    # # WaterLo threshold - F1 / IoU curve
    # WL_THRESHOLDS = [round(x, 2) for x in np.linspace(0.05, 0.95, 19)]
    # for th in WL_THRESHOLDS:
    #     waterlo_rect_eval(NUM_IMAGES, os.path.join(EXP_DIR, "plain"), f"PLAIN TH={th}", 0.3, threshold=th)
    # for th in WL_THRESHOLDS:
    #     waterlo_rect_eval(NUM_IMAGES, os.path.join(EXP_DIR, "plain_wl_0.002"), f"PLAIN_WL_0.002 TH={th}", 0.3, threshold=th)
    # for th in WL_THRESHOLDS:
    #     waterlo_rect_eval(NUM_IMAGES, os.path.join(EXP_DIR, "plain_wl_0.005"), f"PLAIN_WL_0.005 TH={th}", 0.3, threshold=th)
    # for th in WL_THRESHOLDS:
    #     waterlo_rect_eval(NUM_IMAGES, os.path.join(EXP_DIR, "plain_wl_0.01"), f"PLAIN_WL_0.01 TH={th}", 0.3, threshold=th)
    # for th in WL_THRESHOLDS:
    #     waterlo_rect_eval(NUM_IMAGES, os.path.join(EXP_DIR, "plain_wl_0.05"), f"PLAIN_WL_0.05 TH={th}", 0.3, threshold=th)
    # for th in WL_THRESHOLDS:
    #     waterlo_rect_eval(NUM_IMAGES, os.path.join(EXP_DIR, "plain_wl_0.1"), f"PLAIN_WL_0.1 TH={th}", 0.3, threshold=th)
    # for th in WL_THRESHOLDS:
    #     waterlo_rect_eval(NUM_IMAGES, os.path.join(EXP_DIR, "plain_wl_0.2"), f"PLAIN_WL_0.2 TH={th}", 0.3, threshold=th)

    # # Timing evaluation
    # SKIP = 1
    # WL_ALPHA = 0.005

    # # 1. PRC applying
    # with open(PROMPTS_FILE, "r", encoding="utf-8") as f: prompts = json.load(f)[:NUM_IMAGES]
    # clear_timing_stats()
    # applyPRC(BASE_DIR, KEY_ID, MODEL_ID, prompts, message_fn=lambda i: str(i))
    # stats = get_timing_stats(skip_first=SKIP)
    # write_timing_block("PRC apply", {"n": len(prompts), "skip": SKIP}, stats)

    # # 2. Generation without PRC
    # clear_timing_stats()
    # applyPRC(BASE_DIR, "", MODEL_ID, prompts, watermark=False)
    # stats = get_timing_stats(skip_first=SKIP)
    # write_timing_block("PRC apply (plain SD)", {"n": len(prompts), "skip": SKIP}, stats)

    # # 3. PRC decoding
    # prc_images = load_sorted_rgb_images(os.path.join(EXP_DIR, "prc"), NUM_IMAGES)
    # clear_timing_stats()
    # decodePRC(BASE_DIR, KEY_ID, MODEL_ID, prc_images)
    # stats = get_timing_stats(skip_first=SKIP)
    # write_timing_block("PRC decode", {"n": len(prc_images), "skip": SKIP}, stats)

    # # 4. WaterLo applying
    # plain_images = load_sorted_rgb_images(os.path.join(EXP_DIR, "plain"), NUM_IMAGES)
    # clear_timing_stats()
    # applyWaterLo(plain_images, BASE_DIR, alpha=WL_ALPHA)
    # stats = get_timing_stats(skip_first=SKIP)
    # write_timing_block("WaterLo apply", {"alpha": WL_ALPHA, "n": len(plain_images), "skip": SKIP}, stats)

    # # 5. WaterLo detecting
    # wl_images = load_sorted_rgb_images(os.path.join(EXP_DIR, "plain_wl_0.005"), NUM_IMAGES)
    # clear_timing_stats()
    # detectWaterLo(wl_images, BASE_DIR)
    # stats = get_timing_stats(skip_first=SKIP)
    # write_timing_block("WaterLo detect", {"compression": 101, "n": len(wl_images), "skip": SKIP}, stats)