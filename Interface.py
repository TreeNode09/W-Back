from collections.abc import Callable
from typing import Any

import numpy as np
import os
import json
import random
import secrets
import shutil
import torch
import pickle

from datasets import load_dataset
from diffusers import DPMSolverMultistepScheduler

from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, Resize
from torch.nn.functional import interpolate


def generateKey(out_path: str) -> str:
    """Generate a PRC key pair and save under ``out_path``.

    ## File:
    - create `{out_path}/keys` if not found
    - save key to `{out_path}/keys/{key_id}.pkl`

    ## Return:
    - ``key_id``: cryptographically random hex string
    """

    from PRC.src.prc import KeyGen

    N = 4 * 64 * 64  # the length of a PRC codeword
    encoding_key, decoding_key = KeyGen(N)

    keys_dir = os.path.join(out_path, "keys")
    os.makedirs(keys_dir, exist_ok=True)

    while True:

        key_id = secrets.token_hex(16)
        key_path = os.path.join(keys_dir, f"{key_id}.pkl")
        if not os.path.exists(key_path): break

    with open(key_path, 'wb') as f: pickle.dump((encoding_key, decoding_key), f)

    with open(key_path, 'rb') as f: extracted, _ = pickle.load(f)
    if extracted[0].all() != encoding_key[0].all(): raise ValueError("saved file is invalid")

    return key_id


def _build_scheduler(solver_order: int = 1):

    return DPMSolverMultistepScheduler(beta_end=0.012, beta_schedule="scaled_linear", beta_start=0.00085,
        num_train_timesteps=1000, prediction_type="epsilon", steps_offset=1, trained_betas=None, solver_order=solver_order)


def _preparePRC(out_path: str, model_id: str, *, allow_download: bool = False) -> Any:
    """Get a Stable Diffusion pipeline of a HuggingFace model with `model_id`.
    Always try local cache first.

    ## File:
    - create folder `{out_path}/models` if not found
    - save HuggingFace cache to folder `{out_path}/models/[model_name]`

    ## Return:
    - `pipe`: an SD pipeline with `model_id`
    """

    from PRC.src.inverse_stable_diffusion import InversableStableDiffusionPipeline

    models_dir = os.path.join(out_path, "models")
    os.makedirs(models_dir, exist_ok=True)

    scheduler = _build_scheduler(solver_order=1)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try: pipe = InversableStableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float32,
            cache_dir=models_dir, local_files_only=True)

    except Exception as e:

        if not allow_download: raise FileNotFoundError(f"Model not found locally in {models_dir}: {model_id}") from e

        pipe = InversableStableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float32,
            cache_dir=models_dir, local_files_only=False)

    pipe.set_progress_bar_config(disable=True)

    return pipe.to(device)


def getPrompts(dataset_id: str, num: int, out_path: str | None = None) -> list[str]:
    """Fetch `num` prompts from a HuggingFace dataset.
    Save fetched prompts if `out_path` is given.

    ## File:
    - if `out_path` is given:
        - create folder `{save_path}` if not found
        - save `{save_path}/prompts.json`

    ## Return:
    - `prompts`: list of gathered prompts
    """

    if num <= 0: raise ValueError(f"num={num} is not positive")

    if out_path is not None: os.makedirs(out_path, exist_ok=True)

    ds = load_dataset(dataset_id, split="test")
    if len(ds) < num: raise ValueError(f"dataset has less than num={num} record(s)")

    sample = ds[0]
    if isinstance(sample, dict):

        if "Prompt" in sample: key = "Prompt"
        elif "prompt" in sample: key = "prompt"
        else: raise ValueError(f"prompt not found in sample keys={list(sample.keys())}")

    else: raise ValueError("dataset is not dict-like")

    random.seed(42)
    indexes = random.sample(range(len(ds)), num)
    prompts = [ds[i][key] for i in indexes]

    if out_path is not None:

        prompts_path = os.path.join(out_path, "prompts.json")
        with open(prompts_path, "w", encoding="utf-8") as f: json.dump(prompts, f, ensure_ascii=False)

    return prompts


def _seed_everything(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def applyPRC(in_path: str, key_id: str, model_id: str, prompts: list[str],
    out_path: str | None = None, watermark: bool = True, on_progress: Callable[[int, int], None] | None = None) -> list[Any]:
    """Apply PRC watermark by generating images from watermarked latents, or plain SD when `watermark` is False.
    Only called when key, model and prompts are ready (key is ignored when `watermark` is False).
    Save generated images if `out_path` is given.

    `on_progress` is a callback: `(current: int, total: int) -> None`

    ## File:
    - if `watermark`: load key from `{in_path}/keys/{key_id}.pkl`
    - load model from `{in_path}/models/[model_name]`, where `model_name` matches `model_id`
    - if `out_path` is given:
        - save generated images to `{out_path}/{i}.png`

    ## Return:
    - `images`: list of generated images

    **Original Code:** `PRC/src/encode.py`
    """

    from PRC.inversion import generate

    if not prompts: return []

    INF_STEPS = 50

    # Must use a local model
    model_cache_dir = os.path.join(in_path, "models")
    pipe = _preparePRC(model_cache_dir, model_id, allow_download=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if watermark:

        from PRC.src.prc import Encode
        import PRC.src.pseudogaussians as prc_gaussians

        # Must generate a key before watermarking
        key_file = os.path.join(in_path, "keys", f"{key_id}.pkl")
        if not os.path.exists(key_file): raise FileNotFoundError(f"Key not found: {key_file}")
        with open(key_file, "rb") as f: encoding_key, _ = pickle.load(f)

    if out_path is not None: os.makedirs(out_path, exist_ok=True)

    images = []
    for i, prompt in enumerate(prompts):

        _seed_everything(i)

        if watermark:

            prc_codeword = Encode(encoding_key)
            init_latents = prc_gaussians.sample(prc_codeword).reshape(1, 4, 64, 64).to(device)

        else:

            init_latents_np = np.random.randn(1, 4, 64, 64)
            init_latents = torch.from_numpy(init_latents_np).to(torch.float64).to(device)

        image, _, _ = generate(prompt=prompt, init_latents=init_latents, num_inference_steps=INF_STEPS, solver_order=1,
            pipe=pipe)

        images.append(image)

        if out_path is not None: image.save(os.path.join(out_path, f"{i}.png"))

        if on_progress is not None:  on_progress(i + 1, len(prompts))

    return images


def decodePRC(in_path: str, key_id: str, model_id: str, images: list[Any], *,
    inf_steps: int = 50, decoder_inv_steps: int = 20) -> list[tuple[bool, bool, bool]]:
    """Detect / decode PRC watermark on a list of images.

    Reduce `decoder_inv_steps` to speed up processing.

    ## File:
    - load key from `{in_path}/keys/{key_id}.pkl`
    - load model from `{in_path}/models/[model_name]`, where `model_name` matches `model_id`

    ## Return:
    - tuples of `combined`, `decode` and `detect`

    **Original Code:** `PRC/src/decode.py`
    """

    from PRC.src.prc import Detect, Decode
    import PRC.src.pseudogaussians as prc_gaussians
    from PRC.inversion import exact_inversion

    if not images: return []

    key_file = os.path.join(in_path, "keys", f"{key_id}.pkl")
    if not os.path.exists(key_file): raise FileNotFoundError(f"Key not found: {key_file}")
    with open(key_file, "rb") as f: _, decoding_key = pickle.load(f)

    model_cache_dir = os.path.join(in_path, "models")
    pipe = _preparePRC(model_cache_dir, model_id, allow_download=False)

    results: list[tuple[bool, bool, bool]] = []
    for img in images:

        reversed_latents: torch.Tensor = exact_inversion(img, prompt="", test_num_inference_steps=inf_steps, inv_order=0, 
            pipe=pipe, decoder_inv_steps=decoder_inv_steps)

        reversed_prc = prc_gaussians.recover_posteriors(reversed_latents.to(torch.float64).flatten().cpu()).flatten().cpu()

        detect = Detect(decoding_key, reversed_prc)
        decode = Decode(decoding_key, reversed_prc) is not None
        combined = detect or decode

        results.append((combined, detect, decode))

    return results


def makeWaterLoDataset(in_path: str, out_path: str, train_ratio: float = 0.8) -> tuple[int, int]:
    """Prepare WaterLo train/valid folders from an image directory.

    ## File:
    - load images from `{in_path}`
    - save train images as `{out_path}/train/train_img012.png`
    - save valid images as `{out_path}/valid/valid_img012.png`

    ## Return:
    - numers of images in train/valid folders
    """

    if not os.path.isdir(in_path): raise FileNotFoundError(f"in_path='{in_path}' not found")
    if not (0.0 < train_ratio < 1.0): raise ValueError(f"train_ratio={train_ratio} must be in range (0, 1)")

    image_extensions = {".jpg", ".png", ".jpeg", ".JPEG"}
    images = []
    for name in os.listdir(in_path):

        src = os.path.join(in_path, name)
        if os.path.isfile(src) and os.path.splitext(name)[1] in image_extensions: images.append(src)

    if not images: raise FileNotFoundError(f"no images found in: {in_path}")

    images = sorted(images, key=lambda p: (os.path.splitext(p)[1].lower(), os.path.basename(p)))
    random.seed(42)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * train_ratio)
    train_list = images[:n_train]
    valid_list = images[n_train:]
    width = max(2, len(str(n)))

    train_dir = os.path.join(out_path, "train")
    valid_dir = os.path.join(out_path, "valid")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)

    for i, src_path in enumerate(train_list, start=1):

        ext = os.path.splitext(src_path)[1]
        dst = os.path.join(train_dir, f"train_img{i:0{width}d}{ext}")
        shutil.copy2(src_path, dst)

    for i, src_path in enumerate(valid_list, start=1):

        ext = os.path.splitext(src_path)[1]
        dst = os.path.join(valid_dir, f"valid_img{i:0{width}d}{ext}")
        shutil.copy2(src_path, dst)

    return len(train_list), len(valid_list)


def trainWaterLo(in_path: str, out_path: str, *,
    alpha: float = 0.005, lambd: float = 4, epochs: int = 32, batch_size: int = 8, loss: str = "ssim",
    compression: bool = False, weights_path: str | None = None) -> None:
    """Train WaterLo models with given train and valid images.

    `loss` only supports `"mse"` or `"ssim"`.

    ## File:
    - load train images from `{in_path}/train`
    - save best checkpoint to `{out_path}/G.pt` and `{out_path}/B.pt`
    - save training preview images in `{out_path}` (periodically overwritten):
        - `original.png`, `watermark.png`, `watermark_noise.png`
        - `inner_mask_gt.png`, `outer_mask_gt.png`
        - `inner_mask_bob.png`, `outer_mask_bob.png`, `bob_real.png`, `bob_fake.png`
    - if `weights_path` is given:
        - load models and resume training from `{weights_path}/G.pt` and `{weights_path}/B.pt`

    **Original Code:** `WaterLo/src/main.py`, in `main()`
    """

    from WaterLo.src.loader import loader_with_padding
    from WaterLo.src.models import Generator, Bob
    from WaterLo.src.loss import GeneratorLoss, BobLoss
    from WaterLo.src.utils import Models, Losses, Optimizers
    from WaterLo.src.main import fit

    if epochs <= 0: raise ValueError(f"epochs={epochs} is not positive")
    if batch_size <= 0: raise ValueError(f"batch_size={batch_size} is not positive")
    if loss not in {"mse", "ssim"}: raise ValueError(f"loss='{loss}' is not 'mse' or 'ssim'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(1)
    os.makedirs(out_path, exist_ok=True)

    generator = Generator(in_channels=3, out_channels=3).to(device)
    bob = Bob(in_channels=3, out_channels=1).to(device)
    models = Models(generator, bob)
    if weights_path is not None: models.load(weights_path)

    generator_loss = GeneratorLoss(device, alpha=alpha, loss=loss)
    bob_loss = BobLoss(device)
    criterions = Losses(generator_loss, bob_loss)

    generator_optimizer = torch.optim.AdamW(models.G.parameters(), lr=2e-4)
    bob_optimizer = torch.optim.AdamW(models.B.parameters(), lr=2e-4)
    optimizers = Optimizers(generator_optimizer, bob_optimizer)

    # Input size fixed to 512
    train_loader = loader_with_padding(in_path, 512, batch_size, "train")
    valid_loader = loader_with_padding(in_path, 512, batch_size, "valid")

    fit(models, epochs, criterions, optimizers, device, train_loader, valid_loader, out_path, compression, lambd, alpha)


def _prepareWaterLo(in_path: str) -> tuple[Any, torch.device]:
    """Load WaterLo G & B weights from `{in_path}`.
    
    **Original Code:** `WaterLo/src/apply_watermark.py`, in `main()`
    """

    from WaterLo.src.models import Generator, Bob
    from WaterLo.src.utils import Models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator(in_channels=3, out_channels=3).to(device)
    bob = Bob(in_channels=3, out_channels=1).to(device)
    models = Models(generator, bob)
    models.load(in_path)

    return models, device


def applyWaterLo(images: list[Any], in_path: str, alpha: float = 0.005,
    out_path: str | None = None, batch_size: int = 8, on_progress: Callable[[int, int], None] | None = None) -> list[Any]:
    """Apply WaterLo invisible watermark to RGB images.
    Save watermarked images if `out_path` is given.

    `on_progress` is a callback: `(current: int, total: int) -> None` (``current`` in ``1..len(images)``).

    ## File:
    - load `{in_path}/models/G.pt`
    - load `{in_path}/models/B.pt`
    - if `out_path` is given:
        - create `{out_path}` if not found
        - save watermarked images to `{out_path}/{i}.png`

    ## Return:
    - list of watermarked `PIL.Image`

    **Original Code:** `WaterLo/src/apply_watermark.py`
    """

    from WaterLo.src.apply_watermark import forward_watermark
    from WaterLo.src.loader import padding

    if not images: return []
    if batch_size <= 0: raise ValueError(f"batch_size={batch_size} is not positive")

    models, device = _prepareWaterLo(os.path.join(in_path, "models"))
    models.eval()

    to_pil = ToPILImage()
    out_images: list[Any] = []

    if out_path is not None: os.makedirs(out_path, exist_ok=True)

    tfm = Compose([ToTensor(), Lambda(lambda x: padding(x, size=512))])

    with torch.no_grad():

        for start in range(0, len(images), batch_size):

            chunk = images[start : min(start + batch_size, len(images))]

            batch = torch.stack([tfm(im.convert("RGB")) for im in chunk], dim=0)
            size = ([im.size[0] for im in chunk], [im.size[1] for im in chunk])
            croped = forward_watermark(models, batch, size, device, alpha)

            for i, t in enumerate(croped):

                out_image = to_pil(t.cpu())
                out_images.append(out_image)

                if out_path is not None: out_image.save(os.path.join(out_path, f"{start + i}.png"))

                if on_progress is not None: on_progress(start + i + 1, len(images))

    return out_images


def detectWaterLo(images: list[Any], in_path: str, compression: int = 101,
    out_path: str | None = None, batch_size: int = 8,) -> tuple[list[Any], list[np.ndarray]]:
    """Detect WaterLo watermark maps from RGB images.
    Save visualized results if `out_path` is given.

    `compression` > 100 means no compression.

    ## File:
    - load `{in_path}/models/G.pt`
    - load `{in_path}/models/B.pt`
    - if `out_path` is given:
        - create `{out_path}` if not found
        - save maps to `{out_path}/{i}.png`

    ## Return:
    - `(maps, preds)`: heatmaps and predicted values

    **Original Code:** `WaterLo/src/detect_watermark.py`
    """

    from WaterLo.src.jpeg import add_jpeg_noise
    from WaterLo.src.utils import rgb_to_ycbcr

    if not images:
        return [], []
    if batch_size <= 0: raise ValueError(f"batch_size={batch_size} is not positive")

    models, device = _prepareWaterLo(os.path.join(in_path, "models"))
    models.eval()

    tfm = Compose([Resize((512, 512)), ToTensor()])
    to_pil = ToPILImage()
    maps: list[Any] = []
    preds: list[np.ndarray] = []

    if out_path is not None: os.makedirs(out_path, exist_ok=True)

    with torch.no_grad():

        for start in range(0, len(images), batch_size):

            chunk = images[start : min(start + batch_size, len(images))]
            batch = torch.stack([tfm(im.convert("RGB")) for im in chunk], dim=0).to(device)

            if compression <= 100: batch = add_jpeg_noise(batch, device, compression)

            bob_preds = models.B(batch)
            pred: torch.Tensor = interpolate(bob_preds, size=batch.shape[2:], mode="nearest")

            for i in range(batch.shape[0]):

                preds.append(pred[i, 0].detach().float().cpu().numpy().copy())

                img_map = rgb_to_ycbcr(batch[i].clone())
                img_map[1] = pred[i, 0]
                img_map[2] = 1 - pred[i, 0]

                result = to_pil(img_map.detach().cpu().clamp(0, 1))
                maps.append(result)

                if out_path is not None: result.save(os.path.join(out_path, f"{start + i}.png"))

    return maps, preds