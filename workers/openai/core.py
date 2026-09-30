"""Shared core for the OpenAI-compatible workers (vllm/sglang/llama/openai).

They all proxy the same OpenAI-compatible API, so the logic lives
here and the per-engine adapters just pass an EngineDefaults. Every default is
env-overridable: the image is version-locked to the engine, so it owns the
engine/version-specific values (log path, health endpoint, log grammar)."""

import base64
import binascii
import json
import math
import os
import re
import wave
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from vastai import Worker, WorkerConfig, HandlerConfig, LogActionConfig, BenchmarkConfig
from vastai.serverless.server.lib.data_types import ApiPayload, JsonDataException

from workers.openai.benchmark import (
    BENCHMARKS,
    DEFAULT_BENCHMARK_ROUTE,
    REF_AUDIO_GEN_SECONDS,
    REF_AUDIO_SECONDS,
    REF_EMBED_CHARS,
    REF_IMAGE_SIDE,
    REF_RERANK_CHARS,
    RERANK_DOC_CHARS,
    resolve_model_name as _resolve_model_name,
    benchmark_speech,
    synthetic_png,
    _words,
)


def _env_lines(name, default):
    """Newline-delimited env var -> list of stripped lines; default if unset/empty."""
    raw = os.environ.get(name)
    return [s for ln in raw.splitlines() if (s := ln.strip())] if raw else default


@dataclass(frozen=True)
class EngineDefaults:
    """Per-engine baked defaults; each is overridden by the matching env var if set."""

    name: str                 # engine id for the startup banner
    model_log_file: str       # MODEL_LOG
    load_log_msgs: List[str]  # MODEL_LOAD_LOG_MSG — model-loaded markers
    error_log_msgs: List[str]  # MODEL_ERROR_LOG_MSGS — failed-load markers
    info_log_msgs: List[str] = field(default_factory=lambda: ['"message":"Download'])  # MODEL_INFO_LOG_MSGS


MODEL_SERVER_URL = "http://127.0.0.1"
MODEL_SERVER_PORT = 18000


def request_parser(request):
    return request["input"] if request.get("input") is not None else request


DEFAULT_AUDIO_FILENAME = "audio.wav"
DEFAULT_IMAGE_FILENAME = "image.png"

# Uploads are decoded before the SDK checks the signature, so they are bounded first.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024        # per file: the OpenAI limit
MAX_REQUEST_UPLOAD_BYTES = 64 * 1024 * 1024
MAX_UPLOAD_FILES = 16

# Engines pick a decoder from the content type, which comes from the extension.
AUDIO_TYPES = {
    "flac": "audio/flac", "m4a": "audio/mp4", "mp3": "audio/mpeg", "mp4": "audio/mp4",
    "mpeg": "audio/mpeg", "mpga": "audio/mpeg", "ogg": "audio/ogg", "wav": "audio/wav",
    "webm": "audio/webm",
}
IMAGE_TYPES = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
               "webp": "image/webp"}
VIDEO_TYPES = {"mp4": "video/mp4", "mov": "video/quicktime", "webm": "video/webm"}
VISUAL_TYPES = {**IMAGE_TYPES, **VIDEO_TYPES}
MEDIA_TYPES = {**AUDIO_TYPES, **VISUAL_TYPES}     # mp4 and webm are video
JSON_TYPES = {"json": "application/json"}

# References are fetched by the engine: only these schemes, so never a path on the instance.
REFERENCE_SCHEMES = ("http", "https", "data")
MAX_DATA_URI_PREFIX = 128           # "data:audio/wav;base64" and the like

# The SDK divides every route's in-flight workload by the throughput of the benchmark
# route alone, so every route counts in benchmark requests, clamped.
BENCHMARK_MAX_TOKENS = 500          # completions_benchmark_generator's max_tokens
MIN_REQUEST_MULTIPLE = 0.1
MAX_REQUEST_MULTIPLE = 8.0

REF_IMAGE_PIXELS = REF_IMAGE_SIDE ** 2
REF_SPEECH_CHARS = 500
CHARS_PER_TOKEN = 4
REF_VIDEO_WIDTH, REF_VIDEO_HEIGHT, REF_VIDEO_FRAMES = 832, 480, 33
REF_VIDEO_FPS = 24                  # a typical default; the engine's is per model


def _in_request_units(size: float, reference: float) -> float:
    multiple = min(max(size / reference, MIN_REQUEST_MULTIPLE), MAX_REQUEST_MULTIPLE)
    return BENCHMARK_MAX_TOKENS * multiple


def _number(value: Any, default: float) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return n if math.isfinite(n) else default


def _decode_b64(value: Any, field: str) -> bytes:
    """base64 or a data: URI -> bytes; the size is checked before decoding."""
    if not isinstance(value, str):
        raise JsonDataException({field: "must be a base64 string"})
    start = 0
    if value[:5].lower() == "data:":
        comma = value.find(",", 0, MAX_DATA_URI_PREFIX)
        if comma < 0:
            raise JsonDataException({field: "malformed data: URI"})
        start = comma + 1
    size = len(value) - start
    if size > (MAX_UPLOAD_BYTES + 2) // 3 * 4 + size // 76 * 2 + 4:   # + line breaks
        raise JsonDataException({field: f"larger than {MAX_UPLOAD_BYTES} bytes"})
    value = re.sub(r"\s+", "", value[start:])
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise JsonDataException({field: "not valid base64"})
    if not raw:
        raise JsonDataException({field: "decoded to zero bytes"})
    if len(raw) > MAX_UPLOAD_BYTES:
        raise JsonDataException({field: f"larger than {MAX_UPLOAD_BYTES} bytes"})
    return raw


def _file_part(raw: bytes, filename: Any, default: str, types: Dict[str, str],
               field: str) -> tuple:
    """(filename, bytes, content_type), with a safe basename and an allowed extension."""
    name = os.path.basename(str(filename or "")).strip()
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name).lstrip(".") or default
    stem, dot, ext = name.rpartition(".")
    name = (stem[:128] + dot + ext) if dot else name[:128]     # keep the extension
    ext = ext.lower() if dot else ""
    if ext not in types:
        raise JsonDataException(
            {field: f"unsupported file type {ext or '(none)'!r}; "
                    f"expected one of {', '.join(sorted(types))}"})
    return (name, raw, types[ext])


def _file_parts(values: Any, names: Any, default: str, types: Dict[str, str],
                field: str, name_field: str) -> list:
    values = values if isinstance(values, list) else [values]
    if len(values) > MAX_UPLOAD_FILES:
        raise JsonDataException({field: f"at most {MAX_UPLOAD_FILES} files"})
    names = names if isinstance(names, list) else [names]
    names = names + [None] * (len(values) - len(names))
    return [_file_part(_decode_b64(v, field), n, default, types, name_field)
            for v, n in zip(values, names)]


# WAV is read exactly; other formats are estimated from a typical byte rate.
AUDIO_BYTES_PER_SECOND = {"wav": 32000, "flac": 12000, "ogg": 8000, "webm": 8000}
DEFAULT_AUDIO_BYTES_PER_SECOND = 16000     # mp3, m4a/mp4


def _audio_seconds(raw: bytes, filename: str) -> float:
    if raw.startswith(b"RIFF"):
        try:
            with wave.open(BytesIO(raw)) as w:
                return w.getnframes() / float(w.getframerate())
        except Exception:
            pass
    ext = str(filename).rpartition(".")[2].lower()
    per_second = AUDIO_BYTES_PER_SECOND.get(ext, DEFAULT_AUDIO_BYTES_PER_SECOND)
    return len(raw) / float(per_second)


def _check_budget(values: List[Any], field: str) -> None:
    encoded = sum(len(v) for v in values if isinstance(v, str))
    if encoded * 3 // 4 > MAX_REQUEST_UPLOAD_BYTES:
        raise JsonDataException(
            {field: f"request uploads exceed {MAX_REQUEST_UPLOAD_BYTES} bytes in total"})


def _flatten(values: List[Any]) -> List[Any]:
    return [v for value in values
            for v in (value if isinstance(value, list) else [value])]


def _check_reference(value: Any, field: str) -> None:
    """Only the scheme is checked; the engine refuses what it cannot fetch or decode."""
    values = value if isinstance(value, list) else [value]
    if len(values) > MAX_UPLOAD_FILES:
        raise JsonDataException({field: f"at most {MAX_UPLOAD_FILES} values"})
    for item in values:
        if item is None:
            continue
        if not isinstance(item, str):
            raise JsonDataException({field: "must be a string"})
        try:
            scheme = urlparse(item).scheme
        except ValueError:                  # e.g. "http://[::1"
            scheme = None
        if scheme not in REFERENCE_SCHEMES:
            raise JsonDataException({field: "must be an http(s) URL or a data: URI"})
        if scheme == "data" and (len(item) - MAX_DATA_URI_PREFIX) * 3 // 4 > MAX_UPLOAD_BYTES:
            raise JsonDataException({field: f"larger than {MAX_UPLOAD_BYTES} bytes"})


def _drop_empty(fields: Dict[str, Any], keys: tuple) -> None:
    for key in keys:
        if fields.get(key) == "":
            del fields[key]


def _parse_body(json_msg: Any) -> Dict[str, Any]:
    if not isinstance(json_msg, dict):
        raise JsonDataException({"payload": "must be an object"})
    fields = request_parser(json_msg)
    if not isinstance(fields, dict):
        raise JsonDataException({"input": "must be an object"})
    return dict(fields)


def _fill_model(fields: Dict[str, Any]) -> None:
    if not fields.get("model"):
        model = _resolve_model_name()
        if model:
            fields["model"] = model


class _UploadPayload(ApiPayload):
    """Sent to the engine as multipart form data: `files` as file parts, `fields` as fields."""

    def __init__(self, fields: Dict[str, Any], files: Dict[str, Any]):
        self.fields = fields
        self.files = files

    def generate_payload_json(self) -> Dict[str, Any]:
        raise NotImplementedError("this payload is sent as multipart, not JSON")

    def generate_payload_multipart(self) -> Optional[Dict[str, Any]]:
        return {**self.files, **self.fields}


class TranscriptionPayload(_UploadPayload):
    """{"file": "<base64>", "filename": "a.mp3", ...}"""

    @classmethod
    def for_test(cls) -> "TranscriptionPayload":
        fields: Dict[str, Any] = {}
        _fill_model(fields)
        return cls(fields, {"file": ("benchmark.wav", benchmark_speech(), "audio/wav")})

    @classmethod
    def from_json_msg(cls, json_msg: Any) -> "TranscriptionPayload":
        fields = _parse_body(json_msg)
        raw = fields.pop("file", None)
        if raw is None:
            raise JsonDataException({"file": "field missing"})
        part = _file_part(_decode_b64(raw, "file"), fields.pop("filename", None),
                          DEFAULT_AUDIO_FILENAME, AUDIO_TYPES, "filename")
        _fill_model(fields)
        return cls(fields, {"file": part})

    def count_workload(self) -> float:
        name, raw, _ = self.files["file"]
        return _in_request_units(_audio_seconds(raw, name), REF_AUDIO_SECONDS)


class ImageEditPayload(_UploadPayload):
    """{"image": "<b64>" | [...], "filename": ..., "mask": "<b64>", "mask_filename": ...,
    "prompt": ...}, or `url` / `url[]` in place of `image`."""

    @classmethod
    def for_test(cls) -> "ImageEditPayload":
        side = REF_IMAGE_SIDE
        fields: Dict[str, Any] = {"prompt": _words(60), "size": f"{side}x{side}", "n": 1}
        _fill_model(fields)
        return cls(fields, {"image": [("image.png", synthetic_png(side), "image/png")]})

    @classmethod
    def from_json_msg(cls, json_msg: Any) -> "ImageEditPayload":
        fields = _parse_body(json_msg)
        images = fields.pop("image", None)
        names = fields.pop("filename", None)
        mask = fields.pop("mask", None)
        mask_name = fields.pop("mask_filename", None)
        _drop_empty(fields, ("url", "url[]"))
        urls = [fields.get("url"), fields.get("url[]")]
        _check_budget([*_flatten([images]), mask, *_flatten(urls)], "image")
        for key in ("url", "url[]"):
            _check_reference(fields.get(key), key)

        files: Dict[str, list] = {}
        if images is not None:
            files["image"] = _file_parts(images, names, DEFAULT_IMAGE_FILENAME,
                                         IMAGE_TYPES, "image", "filename")
        if mask is not None:
            files["mask"] = [_file_part(_decode_b64(mask, "mask"), mask_name,
                                        DEFAULT_IMAGE_FILENAME, IMAGE_TYPES,
                                        "mask_filename")]
        if not files.get("image") and not (fields.get("url") or fields.get("url[]")):
            raise JsonDataException({"image": "field missing (or pass `url`)"})
        _fill_model(fields)
        return cls(fields, files)

    def count_workload(self) -> float:
        return _image_workload(self.fields)


# Upload field: (accepted types, name for an unnamed upload). Each is named by
# `<field>_filename`; only input_references repeats.
VIDEO_FILE_FIELDS = {
    "input_reference": (VISUAL_TYPES, DEFAULT_IMAGE_FILENAME),
    "input_references": (MEDIA_TYPES, DEFAULT_IMAGE_FILENAME),
    "control_reference": (VISUAL_TYPES, DEFAULT_IMAGE_FILENAME),
    "source_video": (VIDEO_TYPES, "video.mp4"),
    "source_audio": (AUDIO_TYPES, DEFAULT_AUDIO_FILENAME),
    "video_noise_mask": (JSON_TYPES, "mask.json"),
    "audio_noise_mask": (JSON_TYPES, "mask.json"),
}
# Reference object field: the key in it the engine fetches.
VIDEO_REFERENCE_FIELDS = {"image_reference": "image_url", "video_reference": "video_url",
                          "audio_reference": "audio_url"}


class VideoPayload(_UploadPayload):
    """/v1/videos/sync: the engine's form fields, with uploads base64'd."""

    @classmethod
    def for_test(cls) -> "VideoPayload":
        fields: Dict[str, Any] = {"prompt": _words(60), "width": REF_VIDEO_WIDTH,
                                  "height": REF_VIDEO_HEIGHT, "num_frames": REF_VIDEO_FRAMES}
        _fill_model(fields)
        return cls(fields, {})

    @classmethod
    def from_json_msg(cls, json_msg: Any) -> "VideoPayload":
        fields = _parse_body(json_msg)
        uploads = {name: fields.pop(name, None) for name in VIDEO_FILE_FIELDS}
        names = {name: fields.pop(f"{name}_filename", None) for name in VIDEO_FILE_FIELDS}
        urls = []
        for name, key in VIDEO_REFERENCE_FIELDS.items():
            value = fields.get(name)
            if value is None:
                continue
            refs = value if isinstance(value, list) else [value]
            if not all(isinstance(r, dict) for r in refs):
                raise JsonDataException({name: "must be an object or a list of objects"})
            _check_reference([r.get(key) for r in refs], f"{name}.{key}")
            urls += [r.get(key) for r in refs]
            if isinstance(value, list):
                fields[name] = json.dumps(value)    # the SDK would repeat a list
        _check_budget([*_flatten(list(uploads.values())), *urls], "input_references")

        files: Dict[str, Any] = {}
        for name, value in uploads.items():
            if value is None:
                continue
            if isinstance(value, list) and name != "input_references":
                raise JsonDataException({name: "one file"})
            types, default = VIDEO_FILE_FIELDS[name]
            parts = _file_parts(value, names[name], default, types, name, f"{name}_filename")
            files[name] = parts if name == "input_references" else parts[0]
        _fill_model(fields)
        return cls(fields, files)

    def count_workload(self) -> float:
        return _video_workload(self.fields)


def _unwrap_input(request: Any) -> Any:
    """For routes where `input` is a field itself: unwrap only a dict."""
    return request["input"] if isinstance(request, dict) and isinstance(
        request.get("input"), dict) else request


def speech_request_parser(request: Any) -> Dict[str, Any]:
    request = _unwrap_input(request)
    if not isinstance(request, dict):
        raise JsonDataException({"payload": "must be an object"})
    _drop_empty(request, ("ref_audio", "ref_audio_2"))
    refs = [request.get("ref_audio"), request.get("ref_audio_2")]
    _check_budget(_flatten(refs), "ref_audio")
    for key in ("ref_audio", "ref_audio_2"):
        _check_reference(request.get(key), key)
    return request


def speech_batch_request_parser(request: Any) -> Dict[str, Any]:
    request = _unwrap_input(request)
    if not isinstance(request, dict):
        raise JsonDataException({"payload": "must be an object"})
    items = request.get("items")
    if not isinstance(items, list) or not all(isinstance(i, dict) for i in items):
        raise JsonDataException({"items": "must be a list of objects"})
    for item in (request, *items):
        _drop_empty(item, ("ref_audio",))
    refs = [item.get("ref_audio") for item in (request, *items)]
    if any(r is not None and not isinstance(r, str) for r in refs):
        raise JsonDataException({"ref_audio": "must be a string"})
    _check_budget(refs, "ref_audio")
    for ref in refs:
        _check_reference(ref, "ref_audio")
    return request


def _image_workload(data: Dict[str, Any]) -> float:
    """n x pixels; `size` is "WxH", or "auto" when the engine decides."""
    pixels = REF_IMAGE_PIXELS
    w, sep, h = str(data.get("size") or "").lower().partition("x")
    if sep:
        pixels = _number(w, REF_IMAGE_SIDE) * _number(h, REF_IMAGE_SIDE)
    return _in_request_units(_number(data.get("n"), 1) * pixels, REF_IMAGE_PIXELS)


def _speech_workload(data: Dict[str, Any]) -> float:
    """Text, plus one reference request per voice-clone reference."""
    text = data.get("input")
    chars = len(text) if isinstance(text, str) else 0
    refs = _flatten([data.get("ref_audio"), data.get("ref_audio_2")])
    chars += REF_SPEECH_CHARS * sum(1 for r in refs if isinstance(r, str))
    return _in_request_units(chars, REF_SPEECH_CHARS)


def _speech_batch_workload(data: Dict[str, Any]) -> float:
    """Each item as on speech; the batch's ref_audio applies to items without one."""
    items = data.get("items")
    chars = 0
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict):
            text = item.get("input")
            chars += len(text) if isinstance(text, str) else 0
            if item.get("ref_audio") or data.get("ref_audio"):
                chars += REF_SPEECH_CHARS
    return _in_request_units(chars, REF_SPEECH_CHARS)


def _chat_batch_workload(data: Dict[str, Any]) -> float:
    messages = data.get("messages")
    count = len(messages) if isinstance(messages, list) else 0
    tokens = _number(data.get("max_tokens"), 0) * count
    return _in_request_units(tokens, BENCHMARK_MAX_TOKENS)


def _audio_generate_workload(data: Dict[str, Any]) -> float:
    seconds = _number(data.get("audio_length"), REF_AUDIO_GEN_SECONDS)
    return _in_request_units(seconds, REF_AUDIO_GEN_SECONDS)


def _video_workload(data: Dict[str, Any]) -> float:
    """Pixels x frames x outputs; anything the model picks counts as the reference."""
    width, height = data.get("width"), data.get("height")
    w, sep, h = str(data.get("size") or "").lower().partition("x")
    if sep:                                 # the engine lets `size` win
        width, height = w, h
    pixels = (_number(width, REF_VIDEO_WIDTH) * _number(height, REF_VIDEO_HEIGHT)
              if width and height else REF_VIDEO_WIDTH * REF_VIDEO_HEIGHT)
    frames = REF_VIDEO_FRAMES
    if data.get("num_frames"):
        frames = _number(data["num_frames"], REF_VIDEO_FRAMES)
    elif data.get("seconds"):
        frames = (_number(data["seconds"], 1)
                  * _number(data.get("fps"), REF_VIDEO_FPS))
    outputs = _number(data.get("num_outputs_per_prompt"), 1)
    return _in_request_units(pixels * frames * outputs,
                             REF_VIDEO_WIDTH * REF_VIDEO_HEIGHT * REF_VIDEO_FRAMES)


def _score_input_chars(value: Any) -> int:
    """Text by length; a multimodal input counts as one document."""
    return len(value) if isinstance(value, str) else RERANK_DOC_CHARS


def _pairs_workload(queries: List[Any], documents: Any) -> float:
    """Each (query, document) pair a reranker reads: one query against many, or two lists
    pairwise."""
    documents = documents if isinstance(documents, list) else [documents]
    pairs = ([(queries[0], d) for d in documents] if len(queries) == 1
             else list(zip(queries, documents)))
    chars = sum(_score_input_chars(q) + _score_input_chars(d) for q, d in pairs)
    return _in_request_units(chars, REF_RERANK_CHARS)


def _rerank_workload(data: Dict[str, Any]) -> float:
    return _pairs_workload([data.get("query")], data.get("documents"))


def _score_workload(data: Dict[str, Any]) -> float:
    """vLLM's shapes: queries with documents or items, text_1/text_2, data_1/data_2."""
    def first(*keys):
        return next((data[k] for k in keys if data.get(k) is not None), None)
    queries = first("queries", "text_1", "data_1")
    return _pairs_workload(queries if isinstance(queries, list) else [queries],
                           first("documents", "items", "text_2", "data_2"))


def _embeddings_workload(data: Dict[str, Any]) -> float:
    """`input`: a string, or a list of strings, tokens or token lists; items are summed."""
    value = data.get("input")
    items = value if isinstance(value, (list, tuple)) else [value]
    chars = 0
    for item in items:
        if isinstance(item, str):
            chars += len(item)
        elif isinstance(item, (list, tuple)):
            chars += len(item) * CHARS_PER_TOKEN
        elif isinstance(item, int):
            chars += CHARS_PER_TOKEN
    return _in_request_units(chars, REF_EMBED_CHARS)


UPLOAD_ROUTES = ("/v1/images/edits", "/v1/videos/sync",
                 "/v1/audio/transcriptions", "/v1/audio/translations")
# Served, with BENCHMARK_ROUTE, when OPENAI_ROUTES is unset: what the worker served before.
DEFAULT_ROUTES = ("/v1/completions", "/v1/chat/completions")


def benchmark_route() -> str:
    return os.environ.get("BENCHMARK_ROUTE", "").strip() or DEFAULT_BENCHMARK_ROUTE


def _served_routes(handlers: List[HandlerConfig]) -> List[HandlerConfig]:
    """OPENAI_ROUTES, less the upload routes on an SDK that cannot send multipart."""
    known = {h.route for h in handlers}
    wanted = {r.strip() for r in os.environ.get("OPENAI_ROUTES", "").split(",") if r.strip()}
    if wanted - known:
        print("WARNING: OPENAI_ROUTES names unknown routes: "
              + ", ".join(sorted(wanted - known)), flush=True)
    wanted = wanted or {*DEFAULT_ROUTES, benchmark_route()}
    unsendable = set()
    if "generate_payload_multipart" not in vars(ApiPayload):
        unsendable = wanted & set(UPLOAD_ROUTES)
        if unsendable:
            print("ERROR: installed vastai SDK cannot send multipart; not serving "
                  + ", ".join(sorted(unsendable)), flush=True)

    if benchmark_route() in unsendable:
        raise RuntimeError(
            f"the benchmark route {benchmark_route()} needs a vastai SDK with multipart "
            "support; upgrade the SDK or benchmark a JSON route")
    served = [h for h in handlers if h.route in wanted and h.route not in unsendable]

    benchmarked = [h.route for h in served if h.benchmark_config]
    if not benchmarked:
        raise RuntimeError(
            f"the benchmark route {benchmark_route()} is not served; set BENCHMARK_ROUTE "
            "to a route this deployment serves, or widen OPENAI_ROUTES")
    print("serving: " + ", ".join(h.route for h in served), flush=True)
    print(f"benchmarking: {benchmarked[0]}. If this model does not serve it, set "
          "BENCHMARK_ROUTE to the route it does.", flush=True)
    return served


def build_config(defaults: EngineDefaults, model_server_url: str = MODEL_SERVER_URL,
                 model_server_port: int = MODEL_SERVER_PORT) -> dict:
    """The WorkerConfig kwargs, apart from run() so the handler table can be tested."""
    # Relative path resolves against the server url+port; a full URL is used as-is.
    healthcheck_url = os.environ.get("MODEL_HEALTH_ENDPOINT", "/health")
    benchmarked = benchmark_route()
    if benchmarked not in BENCHMARKS:
        raise RuntimeError(f"BENCHMARK_ROUTE={benchmarked!r} is not a route; expected one of "
                           + ", ".join(BENCHMARKS))

    def route(path, **kw):
        if path == benchmarked:
            b = BENCHMARKS[path]
            kw["benchmark_config"] = BenchmarkConfig(
                generator=b.generator, concurrency=b.concurrency, runs=b.runs)
        return HandlerConfig(route=path, allow_parallel_requests=True,
                             max_queue_time=600.0, **kw)

    tokens = lambda data: data.get("max_tokens", 0)   # noqa: E731
    handlers = [
        route("/v1/completions", workload_calculator=tokens, request_parser=request_parser),
        route("/v1/chat/completions", workload_calculator=tokens,
              request_parser=request_parser),
        route("/v1/chat/completions/batch", workload_calculator=_chat_batch_workload,
              request_parser=request_parser),
        route("/v1/audio/speech", workload_calculator=_speech_workload,
              request_parser=speech_request_parser),
        route("/v1/audio/speech/batch", workload_calculator=_speech_batch_workload,
              request_parser=speech_batch_request_parser),
        route("/v1/audio/generate", workload_calculator=_audio_generate_workload,
              request_parser=_unwrap_input),
        route("/v1/embeddings", workload_calculator=_embeddings_workload,
              request_parser=_unwrap_input),
        route("/v1/rerank", workload_calculator=_rerank_workload, request_parser=request_parser),
        route("/v1/score", workload_calculator=_score_workload, request_parser=request_parser),
        route("/v1/images/generations", workload_calculator=_image_workload,
              request_parser=request_parser),
        route("/v1/images/edits", payload_class=ImageEditPayload),
        route("/v1/audio/transcriptions", payload_class=TranscriptionPayload),
        route("/v1/audio/translations", payload_class=TranscriptionPayload),
        route("/v1/videos/sync", payload_class=VideoPayload),
    ]

    config = dict(
        model_server_url=model_server_url,
        model_server_port=model_server_port,
        model_log_file=os.environ.get("MODEL_LOG", defaults.model_log_file),
        model_healthcheck_url=healthcheck_url,
        handlers=_served_routes(handlers),
        log_action_config=LogActionConfig(
            on_load=_env_lines("MODEL_LOAD_LOG_MSG", defaults.load_log_msgs),
            on_error=_env_lines("MODEL_ERROR_LOG_MSGS", defaults.error_log_msgs),
            on_info=_env_lines("MODEL_INFO_LOG_MSGS", defaults.info_log_msgs),
        ),
    )
    return config


def run(defaults: EngineDefaults) -> None:
    """Build the WorkerConfig from defaults (env-overridable) and run the worker."""
    # BACKEND=openai aliases vllm, so report the real engine and note the alias.
    backend = os.environ.get("BACKEND")
    alias = f" (BACKEND={backend})" if backend and backend != defaults.name else ""
    print(f"Using worker backend: {defaults.name}{alias}", flush=True)

    Worker(WorkerConfig(**build_config(defaults))).run()
