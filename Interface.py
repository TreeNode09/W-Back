from typing import Any

import numpy as np
import os
import json
import random
import torch
import pickle
from datasets import load_dataset
from diffusers import DPMSolverMultistepScheduler


def _generateKeyID(user_id: str):    #PENDING

    return user_id


def generateKey(out_path: str, user_id: str):
    """Generate a key to apply and detect watermarks, save the key in `out_path`.
    Only called if the current user doesn't have a key.

    ## File:
    - create `{out_path}/keys` if not found
    - save key to `{out_path}/keys/{key_path}.pkl`

    ## Return:
    - `key_id`: ID of the generated key
    """

    from PRC.src.prc import KeyGen

    N = 4 * 64 * 64  # the length of a PRC codeword
    encoding_key, decoding_key = KeyGen(N)

    key_id = _generateKeyID(user_id)
    keys_dir = os.path.join(out_path, "keys")
    os.makedirs(keys_dir, exist_ok=True)

    key_path = os.path.join(keys_dir, f"{key_id}.pkl")
    with open(key_path, 'wb') as f: pickle.dump((encoding_key, decoding_key), f)

    with open(key_path, 'rb') as f: extracted, _ = pickle.load(f)
    if extracted[0].all() != encoding_key[0].all(): raise ValueError("saved file is invalid")

    return key_id


def _build_scheduler(solver_order: int = 1):

    return DPMSolverMultistepScheduler(beta_end=0.012, beta_schedule="scaled_linear", beta_start=0.00085,
        num_train_timesteps=1000, prediction_type="epsilon", steps_offset=1, trained_betas=None, solver_order=solver_order)


def _prepareModel(out_path: str, model_id: str, *, allow_download: bool = False) -> Any:
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

    return pipe.to(device)


def _getPrompts(dataset_id: str, num: int, out_path: str | None = None) -> list[str]:
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


def applyPRC(in_path: str, key_id: str, model_id: str, prompts: list[str], out_path: str | None = None):
    """Apply PRC watermark by generating images from watermarked latents.
    Only called when key, model and prompts are ready.
    Save watermarked images if `out_path` is given.

    ## File:
    - load key from `{in_path}/keys/{key_id}.pkl`
    - load model from `{in_path}/models/[model_name]`, where `model_name` matches `model_id`
    - if `out_path` is given:
        - save generated images to `{out_path}/{i}.png`

    ## Return:
    - `images`: list of generated watermarked images
    """

    from PRC.src.prc import Encode
    import PRC.src.pseudogaussians as prc_gaussians
    from PRC.inversion import generate

    if not prompts: return []

    INF_STEPS = 50

    # Must generate a key before watermarking
    key_file = os.path.join(in_path, "keys", f"{key_id}.pkl")
    if not os.path.exists(key_file): raise FileNotFoundError(f"Key not found: {key_file}")
    with open(key_file, "rb") as f: encoding_key, _ = pickle.load(f)

    # Must use a local model
    model_cache_dir = os.path.join(in_path, "models")
    pipe = _prepareModel(model_cache_dir, model_id, allow_download=False)
    pipe.set_progress_bar_config(disable=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if out_path is not None: os.makedirs(out_path, exist_ok=True)

    images = []
    for i, prompt in enumerate(prompts):

        _seed_everything(i)

        prc_codeword = Encode(encoding_key)
        init_latents = prc_gaussians.sample(prc_codeword).reshape(1, 4, 64, 64).to(device)

        image, _, _ = generate(prompt=prompt, init_latents=init_latents, num_inference_steps=INF_STEPS, solver_order=1,
            pipe=pipe)

        images.append(image)

        if out_path is not None: image.save(os.path.join(out_path, f"{i}.png"))

    return images