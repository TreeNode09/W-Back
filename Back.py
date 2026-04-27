import base64
import io
import os
import secrets
import threading
import time
from typing import Any

from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
from jsonschema import ValidationError, validate

from Interface import *

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

BASE_DIR = r"D:\W-Back\Data"

PROMPT_DATASET = "Gustavosta/Stable-Diffusion-Prompts"
BATCH_SIZE = 8

TESTING = True  # when enabled, will use fixed images instead of real models and prompts


def _get_job_id() -> str:
    return secrets.token_hex(6)  # 12 hex chars


def _load_n_images_from_test(n: int) -> list[Any]:

    d = os.path.join(BASE_DIR, "test")
    paths: list[str] = []
    if os.path.isdir(d):

        exts = {".png", ".jpg", ".jpeg", ".webp"}
        names = sorted(f for f in os.listdir(d) if os.path.splitext(f)[1].lower() in exts)
        paths = [os.path.join(d, f) for f in names]

    if not paths: raise ValueError("TESTING: no image files in Data/test (*.png, *.jpg, *.jpeg, *.webp)")

    out: list[Any] = []
    for i in range(n): out.append(Image.open(paths[i % len(paths)]).convert("RGB"))
    
    return out


def _json_dict() -> dict[str, Any] | None:

    d = request.get_json(silent=True)

    return d if isinstance(d, dict) else None


def _png_to_b64(img: Any) -> str:

    buf = io.BytesIO()
    img.save(buf, format="PNG")

    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _load_images_from_request(field: str = "images") -> tuple[list[Any], str | None]:

    parts = request.files.getlist(field)
    loaded: list[Any] = []

    for part in parts:

        if not part or not part.filename: continue

        raw = part.read()
        if not raw: continue

        try: loaded.append(Image.open(io.BytesIO(raw)).convert("RGB"))
        except Exception: return [], f"invalid image file: {part.filename!r}"

    if not loaded: return [], f"non-empty `{field}` file required"
    return loaded, None


def _run_generate_job(job_id: str, sid: str, model_id: str, prompts: list[str], use_prc: bool, use_waterlo: bool,
    alpha: float, key_id: str) -> None:

    def on_prc(current: int, total: int) -> None:

        socketio.emit("generate_prc", {"job_id": job_id, "current": current, "total": total}, to=sid)

    def on_wl(current: int, total: int) -> None:

        socketio.emit("generate_waterlo", {"job_id": job_id, "current": current, "total": total}, to=sid)

    with app.app_context():

        try:

            images = applyPRC(BASE_DIR, key_id, model_id, prompts, watermark=use_prc, out_path=None, on_progress=on_prc)

            if use_waterlo:

                images = applyWaterLo(images, BASE_DIR, alpha=alpha, batch_size=BATCH_SIZE, out_path=None, on_progress=on_wl)

            b64_list = [_png_to_b64(im) for im in images]

            socketio.emit("generate_done", {"job_id": job_id, "images": b64_list, "count": len(b64_list)}, to=sid,)

        except Exception as e: socketio.emit("generate_error", {"job_id": job_id, "error": str(e)}, to=sid)


def _run_waterlo_images_job(job_id: str, sid: str, pil_images: list[Any], alpha: float) -> None:

    def on_wl(current: int, total: int) -> None:

        socketio.emit("generate_waterlo", {"job_id": job_id, "current": current, "total": total}, to=sid)

    with app.app_context():

        try:

            out = applyWaterLo(pil_images, BASE_DIR, alpha=alpha, batch_size=BATCH_SIZE, out_path=None, on_progress=on_wl)
            b64_list = [_png_to_b64(im) for im in out]

            socketio.emit("generate_done", {"job_id": job_id, "images": b64_list, "count": len(b64_list)}, to=sid)

        except Exception as e: socketio.emit("generate_error", {"job_id": job_id, "error": str(e)}, to=sid)


def _run_decode_prc_job(job_id: str, sid: str, pil_images: list[Any], key_id: str, model_id: str) -> None:

    def on_prc(current: int, total: int) -> None:

        socketio.emit("decode_prc", {"job_id": job_id, "current": current, "total": total}, to=sid)

    with app.app_context():

        try:

            results = decodePRC(BASE_DIR, key_id, model_id, pil_images, on_progress=on_prc)
            payload = [{"detect": detect, "decode_bits": bits} for (detect, bits) in results]

            socketio.emit("decode_done", {"job_id": job_id, "method": "prc", "results": payload, "count": len(payload)}, to=sid)

        except Exception as e: socketio.emit("decode_error", {"job_id": job_id, "method": "prc", "error": str(e)}, to=sid)


def _run_decode_waterlo_job(job_id: str, sid: str, pil_images: list[Any]) -> None:

    def on_wl(current: int, total: int) -> None:

        socketio.emit("decode_waterlo", {"job_id": job_id, "current": current, "total": total}, to=sid)

    with app.app_context():

        try:

            maps, preds = detectWaterLo(pil_images, BASE_DIR, on_progress=on_wl)
            maps_out = [_png_to_b64(im) for im in maps]
            preds_out = [p.tolist() for p in preds]

            socketio.emit("decode_done", {
                "job_id": job_id, "method": "waterlo", "maps": maps_out, "preds": preds_out, "count": len(maps_out)
            }, to=sid)

        except Exception as e: socketio.emit("decode_error", {"job_id": job_id, "method": "waterlo", "error": str(e)}, to=sid)


def _run_generate_job_test(job_id: str, sid: str, model_id: str, prompts: list[str], use_prc: bool, use_waterlo: bool,
    alpha: float, key_id: str) -> None:

    n = len(prompts)
    total = max(1, n)

    with app.app_context():

        try:

            for current in range(1, total + 1):

                socketio.emit("generate_prc", {"job_id": job_id, "current": current, "total": total}, to=sid)
                if current <= total: time.sleep(1.0)

            if use_waterlo:

                for current in range(1, total + 1):

                    socketio.emit("generate_waterlo", {"job_id": job_id, "current": current, "total": total}, to=sid)
                    if current <= total: time.sleep(1.0)

            images = _load_n_images_from_test(n)
            b64_list = [_png_to_b64(im) for im in images]

            socketio.emit("generate_done", {"job_id": job_id, "images": b64_list, "count": len(b64_list)}, to=sid)

        except Exception as e: socketio.emit("generate_error", {"job_id": job_id, "error": str(e)}, to=sid)


def _run_waterlo_images_job_test(job_id: str, sid: str, pil_images: list[Any], alpha: float) -> None:

    n = len(pil_images)
    total = max(1, n)

    with app.app_context():

        try:

            for current in range(1, total + 1):

                socketio.emit("generate_waterlo", {"job_id": job_id, "current": current, "total": total}, to=sid)
                if current < total: time.sleep(1.0)

            out = _load_n_images_from_test(n)
            b64_list = [_png_to_b64(im) for im in out]

            socketio.emit("generate_done", {"job_id": job_id, "images": b64_list, "count": len(b64_list)}, to=sid)

        except Exception as e: socketio.emit("generate_error", {"job_id": job_id, "error": str(e)}, to=sid)


def _run_decode_prc_job_test(job_id: str, sid: str, pil_images: list[Any], key_id: str, model_id: str) -> None:

    n = len(pil_images)
    total = max(1, n)

    with app.app_context():

        try:

            for current in range(1, total + 1):

                socketio.emit("decode_prc", {"job_id": job_id, "current": current, "total": total}, to=sid)
                if current <= total: time.sleep(1.0)

            _load_n_images_from_test(n)
            payload = [{"detect": True, "decode_bits": "0" * 64} for _ in range(n)]
            socketio.emit("decode_done", {"job_id": job_id, "method": "prc", "results": payload, "count": len(payload)}, to=sid)

        except Exception as e: socketio.emit("decode_error", {"job_id": job_id, "method": "prc", "error": str(e)}, to=sid)


def _run_decode_waterlo_job_test(job_id: str, sid: str, pil_images: list[Any]) -> None:

    n = len(pil_images)
    total = max(1, n)

    with app.app_context():

        try:
            from torchvision.transforms import ToPILImage, ToTensor
            from WaterLo.src.utils import rgb_to_ycbcr

            for current in range(1, total + 1):

                socketio.emit("decode_waterlo", {"job_id": job_id, "current": current, "total": total}, to=sid)
                if current <= total: time.sleep(1.0)

            maps_src = _load_n_images_from_test(n)

            # convert color space to YCbCr
            tfm = ToTensor()
            to_pil = ToPILImage()
            maps: list[Any] = []
            preds_out: list[list[float]] = []
            for im in maps_src:
                t = tfm(im.convert("RGB"))
                pred = t[0].clone()
                img_map = rgb_to_ycbcr(t.clone())
                img_map[1] = pred
                img_map[2] = 1 - pred
                maps.append(to_pil(img_map.detach().cpu().clamp(0, 1)))
                preds_out.append(pred.detach().float().cpu().numpy().tolist())

            maps_out = [_png_to_b64(im) for im in maps]
            socketio.emit("decode_done", {
                "job_id": job_id, "method": "waterlo", "maps": maps_out, "preds": preds_out, "count": len(maps_out)
            }, to=sid)

        except Exception as e: socketio.emit("decode_error", {"job_id": job_id, "method": "waterlo", "error": str(e)}, to=sid)


@socketio.on("connect")
def handle_socket_connect(): print(f"[Socket] client connected: sid={request.sid}")


@socketio.on("disconnect")
def handle_socket_disconnect(): print(f"[Socket] client disconnected: sid={request.sid}")


@app.route("/test", methods=["GET"])
def handle_test():
    return jsonify({"data": secrets.token_urlsafe(16)})


@app.route("/key", methods=["GET"])
def handle_key():
    """Generate a PRC key.
    
    ## Return:
    - key_id of the generated key
    """

    try: return jsonify({"key_id": generateKey(BASE_DIR)})

    except ValueError as e: return jsonify({"error": str(e)}), 400


@app.route("/prompts", methods=["GET"])
def handle_prompts():
    """Get `num` promtps from the given prompt dataset.
    
    ## Query:
    - `num`: `int` in [1, 8]
    - (optional) `dataset_id`: non-empty `string`, default = 'Gustavosta/Stable-Diffusion-Prompts'

    ## Return:
    - list of fetched prompts, length = `num`
    """

    SCHEMA_PROMPTS: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "num": {"type": "string", "pattern": "^[1-8]$"},
            "dataset_id": {"type": "string", "minLength": 1}
        },
        "required": ["num"]
    }

    raw = {k: request.args[k] for k in request.args}

    try: validate(instance=raw, schema=SCHEMA_PROMPTS)
    except ValidationError as e: return jsonify({"error": e.message}), 400

    num = int(raw["num"])
    dataset_id = raw.get("dataset_id", PROMPT_DATASET).strip()

    try: prompts = getPrompts(dataset_id, num)
    except ValueError as e: return jsonify({"error": str(e)}), 400

    return jsonify({"prompts": prompts, "count": len(prompts)})


@app.route("/generate/prompts", methods=["POST"])
def handle_generate_by_prompts():
    """Generate images with `prompts` and a model with `model_id`, and apply selected watermarks.

    ## Body:
    - `model_id`: non-empty `string`
    - `prompts`: non-empty `array` of `string`
    - `use_prc`: `boolean`
    - `use_waterlo`: `boolean`
    - `socket_id`: non-empty `string`
    - `key_id`:
        - if `use_prc` is `True`: non-empty `string`
        - if `use_prc` is `False`: ignored
    - (optional) `alpha`: `float` in (0, 1], default = 0.005
    
    ## Return:
    - `202 Accepted`: `{"job_id": str, "status": "accepted"}`

    ## Socket:
    - `generate_prc`: `{"job_id": str, "current": int, "total": int}`
    - `generate_waterlo`: `{"job_id": str, "current": int, "total": int}`
    - `generate_done`: `{"job_id": str, "images": base64, "count": int}`
    - `generate_error`: `{"job_id": str, "error": str}`
    """

    data = _json_dict()
    if data is None: return jsonify({"error": "JSON object body required"}), 400

    SCHEMA_GENERATE_BY_PROMTPS: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["model_id", "prompts", "use_prc", "use_waterlo", "socket_id"],
        "properties": {
            "model_id": {"type": "string", "minLength": 1},
            "prompts": {"type": "array", "minItems": 1, "items": {"type": "string"}},
            "socket_id": {"type": "string", "minLength": 1},
            "use_prc": {"type": "boolean"},
            "use_waterlo": {"type": "boolean"},
            "key_id": {"type": ["string", "null"]},
            "alpha": {"type": "number", "exclusiveMinimum": 0, "maximum": 1}
        },
        "allOf": [
            {
                "if": {"properties": {"use_prc": {"const": True}}},
                "then": {
                    "required": ["key_id"],
                    "properties": {"key_id": {"type": "string", "minLength": 1}}
                }
            }
        ]
    }

    try: validate(instance=data, schema=SCHEMA_GENERATE_BY_PROMTPS)
    except ValidationError as e: return jsonify({"error": e.message}), 400

    model_id = data["model_id"].strip()
    prompts = data["prompts"]
    use_prc = data["use_prc"]
    use_waterlo = data["use_waterlo"]
    alpha = float(data.get("alpha", 0.005))

    if use_prc: key_id = str(data["key_id"])
    else: key_id = ""

    sid = str(data["socket_id"]).strip()
    job_id = _get_job_id()

    run = _run_generate_job_test if TESTING else _run_generate_job
    threading.Thread(target=run, daemon=True,
        kwargs= {
            "job_id": job_id, "sid": sid, "model_id": model_id, "prompts": list(prompts),
            "use_prc": use_prc, "use_waterlo": use_waterlo, "alpha": alpha, "key_id": key_id
        }
    ).start()

    return jsonify({"job_id": job_id, "status": "accepted"}), 202


@app.route("/generate/images", methods=["POST"])
def handle_generate_by_images():
    """Apply WaterLo to uploaded RGB images.

    ## Request (`multipart/form-data`):
    - `images`: images readable to PIL
    - `socket_id`: non-empty `string`
    - (optional) `alpha`: `float` in (0, 1], default = 0.005

    ## Return:
    - `202 Accepted`: `{"job_id": str, "status": "accepted"}`

    ## Socket:
    - `generate_prc`: `{"job_id": str, "current": int, "total": int}`
    - `generate_waterlo`: `{"job_id": str, "current": int, "total": int}`
    - `generate_done`: `{"job_id": str, "images": base64, "count": int}`
    - `generate_error`: `{"job_id": str, "error": str}`
    """

    loaded, err = _load_images_from_request("images")
    if err is not None: return jsonify({"error": err}), 400

    sid = (request.form.get("socket_id") or "").strip()
    if not sid: return jsonify({"error": "invalid `socket_id`"}), 400

    alpha_raw = request.form.get("alpha")
    if alpha_raw is None or str(alpha_raw).strip() == "": alpha = 0.005
    else:

        try: alpha = float(alpha_raw)
        except ValueError: return jsonify({"error": "`alpha` must be a number"}), 400

        if not (0.0 < alpha <= 1.0): return jsonify({"error": "`alpha` must be in (0, 1]"}), 400

    job_id = _get_job_id()
    run = _run_waterlo_images_job_test if TESTING else _run_waterlo_images_job
    threading.Thread(target=run, daemon=True,
        kwargs={
            "job_id": job_id, "sid": sid, "pil_images": loaded, "alpha": alpha
        }
    ).start()

    return jsonify({"job_id": job_id, "status": "accepted"}), 202


@app.route("/decode", methods=["POST"])
def handle_decode():
    """Run selected decode branches (PRC / WaterLo) on uploaded RGB images.

    ## Request (`multipart/form-data`):
    - `images`: images readable to PIL
    - `socket_id`: non-empty `string`
    - `use_prc`: `boolean`
    - `use_waterlo`: `boolean`
    - ·model_id`, `key_id`:
        - if `use_prc` is `True`: non-empty `string`
        - if `use_prc` is `False`: ignored

    ## Return:
    - `202 Accepted`: `{"job_id": str, "status": "accepted"}`

    ## Socket:
    - `decode_prc`: `{"job_id": str, "current": int, "total": int}`
    - `decode_waterlo`: `{"job_id": str, "current": int, "total": int}`
    - `decode_done`:
        - `{"job_id": str, "method": "prc", "results": [{"detect": bool, "decode_bits": str | null}, ...], "count": int}`
        - `{"job_id": str, "method": "waterlo", "maps": [base64, ...], "preds": [[float, ...], ...], "count": int}`
    - `decode_error`: `{"job_id": str, "method": "prc"|"waterlo", "error": str}`
    """

    loaded, err = _load_images_from_request("images")
    if err is not None: return jsonify({"error": err}), 400

    sid = (request.form.get("socket_id") or "").strip()
    if not sid: return jsonify({"error": "invalid `socket_id`"}), 400

    def parse_bool(name: str) -> bool | None:
        raw = (request.form.get(name) or "").strip().lower()
        if raw in ("true", "1"): return True
        if raw in ("false", "0"): return False
        return None

    use_prc = parse_bool("use_prc")
    if use_prc is None: return jsonify({"error": "invalid `use_prc` (expected true/false)"}), 400

    use_waterlo = parse_bool("use_waterlo")
    if use_waterlo is None: return jsonify({"error": "invalid `use_waterlo` (expected true/false)"}), 400

    if not (use_prc or use_waterlo): return jsonify({"error": "at least one of `use_prc` / `use_waterlo` must be true"}), 400

    model_id = (request.form.get("model_id") or "").strip()
    key_id = (request.form.get("key_id") or "").strip()
    if use_prc and not model_id: return jsonify({"error": "invalid `model_id`"}), 400
    if use_prc and not key_id: return jsonify({"error": "invalid `key_id`"}), 400

    pil_prc = [im.copy() for im in loaded] if use_prc else []
    pil_wl = [im.copy() for im in loaded] if use_waterlo else []

    job_id = _get_job_id()
    if use_prc:

        run = _run_decode_prc_job_test if TESTING else _run_decode_prc_job
        threading.Thread(target=run, daemon=True,
            kwargs={
                "job_id": job_id, "sid": sid, "pil_images": pil_prc, "key_id": key_id, "model_id": model_id
            }
        ).start()

    if use_waterlo:

        run = _run_decode_waterlo_job_test if TESTING else _run_decode_waterlo_job
        threading.Thread(target=run, daemon=True,
            kwargs={
                "job_id": job_id, "sid": sid, "pil_images": pil_wl
            }
        ).start()

    return jsonify({"job_id": job_id, "status": "accepted"}), 202


if __name__ == "__main__":

    socketio.run(app, host="0.0.0.0", port=5000, debug=True, allow_unsafe_werkzeug=True)