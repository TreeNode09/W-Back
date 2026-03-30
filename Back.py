import base64
import io
import threading
import uuid
from typing import Any

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
from jsonschema import ValidationError, validate

from Interface import *

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

BASE_DIR = r"D:\W\Data"

PROMPT_DATASET = "Gustavosta/Stable-Diffusion-Prompts"
BATCH_SIZE = 8


def _json_dict() -> dict[str, Any] | None:

    d = request.get_json(silent=True)

    return d if isinstance(d, dict) else None


def _png_to_b64(img: Any) -> str:

    buf = io.BytesIO()
    img.save(buf, format="PNG")

    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _run_generate_job(job_id: str, sid: str, model_id: str, prompts: list[str], use_prc: bool, use_waterlo: bool,
    alpha: float, key_id: str) -> None:
    """Runs after HTTP 202; emits progress / result / error to ``sid`` only."""

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
    - `num`: positive `int`
    - (optional) `dataset_id`: non-empty `string`, default = 'Gustavosta/Stable-Diffusion-Prompts'

    ## Return:
    - list of fetched prompts, length = `num`
    """

    SCHEMA_PROMPTS: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "num": {"type": "string", "pattern": "^[1-9][0-9]*$"},
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
    - ``202 Accepted``: ``{"job_id": str, "status": "accepted"}`` — generation runs in a background thread

    ## Socket:
    - `generate_prc`: `{"job_id": str, "current": int, "total": int}`
    - `generate_waterlo`: `{"job_id": str, "current": int, "total": int}`
    - `generate_done`: `{"job_id": str, "images": base64, "countl": int}`
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
    job_id = uuid.uuid4().hex

    threading.Thread(
        target=_run_generate_job, daemon=True,
        kwargs= {
            "job_id": job_id, "sid": sid, "model_id": model_id, "prompts": list(prompts),
            "use_prc": use_prc, "use_waterlo": use_waterlo, "alpha": alpha, "key_id": key_id
        }
    ).start()

    return jsonify({"job_id": job_id, "status": "accepted"}), 202


if __name__ == "__main__":

    socketio.run(app, host="0.0.0.0", port=5000, debug=True, allow_unsafe_werkzeug=True)