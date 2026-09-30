"""Shared core for the OpenAI-compatible workers (vllm/sglang/llama/openai).

They all proxy the same /v1/completions + /v1/chat/completions API, so the logic lives
here and the per-engine adapters just pass an EngineDefaults. Every default is
env-overridable: the image is version-locked to the engine, so it owns the
engine/version-specific values (log path, health endpoint, log grammar)."""

import base64
import binascii
import json
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
    REF_VIDEO_FRAMES,
    REF_VIDEO_HEIGHT,
    REF_VIDEO_WIDTH,
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

# Uploads are buffered, base64-decoded and re-encoded before the SDK checks the request
# signature, so they are bounded here: per file (25 MiB is the OpenAI limit) and per
# request, across every file and inline reference in it.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_REQUEST_UPLOAD_BYTES = 64 * 1024 * 1024
MAX_UPLOAD_FILES = 16

# The content type is chosen from the extension, and engines pick a decoder from it, so
# only the formats each spec route accepts are allowed -- and the table is fixed rather
# than read from the host's mime database, which differs between images.
AUDIO_TYPES = {
    "flac": "audio/flac", "m4a": "audio/mp4", "mp3": "audio/mpeg", "mp4": "audio/mp4",
    "mpeg": "audio/mpeg", "mpga": "audio/mpeg", "ogg": "audio/ogg", "wav": "audio/wav",
    "webm": "audio/webm",
}
IMAGE_TYPES = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
               "webp": "image/webp"}
VIDEO_TYPES = {"mp4": "video/mp4", "mov": "video/quicktime", "webm": "video/webm"}
# mp4 and webm are video where a field takes either: an extension has one content type.
MEDIA_TYPES = {**AUDIO_TYPES, **IMAGE_TYPES, **VIDEO_TYPES}
VISUAL_TYPES = {**IMAGE_TYPES, **VIDEO_TYPES}

# Reference fields an engine resolves for itself (url, ref_audio) accept only http(s) URLs
# and data: URIs. Anything else is refused -- a bare path, and bare base64 too, since "/" is
# a base64 character and a padded path decodes -- so no value reaches the engine as
# something it might open on the instance's own disk. http(s) URLs are fetched by the
# engine from inside the instance and are not filtered here, as with image_url on chat.
REFERENCE_SCHEMES = ("http", "https", "data")
MAX_DATA_URI_PREFIX = 128           # "data:audio/wav;base64" and the like

# Workload unit. The SDK's wait_time divides the in-flight workload of EVERY route by a
# max_throughput measured on one route (BENCHMARK_ROUTE), and 429s any request
# arriving while that exceeds max_queue_time. So every route must count in the same
# unit, or one upload (bytes, pixels) 429s the whole worker.
#
# Non-token routes count in benchmark requests: one reference-sized request weighs the
# same as one benchmark completion, and no request weighs more than
# MAX_REQUEST_MULTIPLE of them. The reference sizes are ESTIMATES, not measurements --
# the clamp is what keeps the gate safe, and it does not depend on them being right.
BENCHMARK_MAX_TOKENS = 500          # completions_benchmark_generator's max_tokens
MIN_REQUEST_MULTIPLE = 0.1
MAX_REQUEST_MULTIPLE = 8.0

REF_IMAGE_PIXELS = REF_IMAGE_SIDE ** 2   # also used when `size` is absent or "auto"
REF_SPEECH_CHARS = 500              # text to synthesise; each clone reference adds one
# REF_AUDIO_SECONDS and REF_EMBED_CHARS live in benchmark.py, whose payloads must be
# exactly one of each.
CHARS_PER_TOKEN = 4                 # sizes pre-tokenised embedding input
MAX_IMAGES = 10                     # the spec's ceiling on `n`
MAX_IMAGE_SIDE = 16384
MAX_AUDIO_GEN_SECONDS = 3600
REF_VIDEO_PIXELS = REF_VIDEO_WIDTH * REF_VIDEO_HEIGHT
REF_VIDEO_FPS = 16                  # frames per second when `seconds` is given without `fps`
MAX_VIDEO_FRAMES = 100_000
MAX_VIDEO_OUTPUTS = 10              # the engine's ceiling on num_outputs_per_prompt


def _in_request_units(size: float, reference: float) -> float:
    multiple = min(max(size / reference, MIN_REQUEST_MULTIPLE), MAX_REQUEST_MULTIPLE)
    return BENCHMARK_MAX_TOKENS * multiple


def _bounded_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        return min(max(int(value), low), high)
    except (TypeError, ValueError, OverflowError):
        return default


def _decode_b64(value: Any, field: str) -> bytes:
    """A base64 string -> bytes, or a JsonDataException naming the offending field.

    Accepts a data: URI and line-wrapped base64 (both common encoder outputs). The size
    limit is checked on the encoded length, before anything is decoded.
    """
    if not isinstance(value, str):
        raise JsonDataException({field: "must be a base64 string"})
    start = 0
    if value[:5].lower() == "data:":
        comma = value.find(",", 0, MAX_DATA_URI_PREFIX)    # bounded: never scans the data
        if comma < 0:
            raise JsonDataException({field: "malformed data: URI"})
        start = comma + 1
    size = len(value) - start
    if size > (MAX_UPLOAD_BYTES + 2) // 3 * 4 + size // 76 * 2 + 4:
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
    """(filename, bytes, content_type) -- the shape the SDK turns into a file part.

    The filename is reduced to a safe basename, and its extension must be one the route
    accepts; the content type comes from that extension.
    """
    name = os.path.basename(str(filename or "")).strip()
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name).lstrip(".") or default
    # Truncate the stem, not the name: cutting a long name mid-extension would reject a
    # legitimate upload for having no extension at all.
    stem, dot, ext = name.rpartition(".")
    name = (stem[:128] + dot + ext) if dot else name[:128]
    ext = ext.lower() if dot else ""
    if ext not in types:
        raise JsonDataException(
            {field: f"unsupported file type {ext or '(none)'!r}; "
                    f"expected one of {', '.join(sorted(types))}"})
    return (name, raw, types[ext])


# An ASR request is priced by seconds of audio, which is what it costs and what engines
# bill. WAV is read exactly; other containers are estimated from a per-format byte rate,
# which is within about 2x (bitrate varies) and bounded by the clamp either way.
AUDIO_BYTES_PER_SECOND = {
    "wav": 32000, "flac": 12000, "mp3": 16000, "mpeg": 16000, "mpga": 16000,
    "m4a": 16000, "mp4": 16000, "ogg": 8000, "webm": 8000,
}
DEFAULT_AUDIO_BYTES_PER_SECOND = 16000


def _audio_seconds(raw: bytes, filename: str) -> float:
    if raw.startswith(b"RIFF"):
        try:
            with wave.open(BytesIO(raw)) as w:
                return w.getnframes() / float(w.getframerate())
        except Exception:
            pass                               # not a WAV we can read: estimate it
    ext = str(filename).rpartition(".")[2].lower()
    per_second = AUDIO_BYTES_PER_SECOND.get(ext, DEFAULT_AUDIO_BYTES_PER_SECOND)
    return len(raw) / float(per_second)


def _check_budget(values: List[Any], field: str) -> None:
    """Refuse a request whose inline data, together, exceeds the per-request budget.

    Checked on the encoded length, before anything is decoded.
    """
    encoded = sum(len(v) for v in values if isinstance(v, str))
    if encoded * 3 // 4 > MAX_REQUEST_UPLOAD_BYTES:
        raise JsonDataException(
            {field: f"request uploads exceed {MAX_REQUEST_UPLOAD_BYTES} bytes in total"})


def _flatten(values: List[Any]) -> List[Any]:
    return [v for value in values
            for v in (value if isinstance(value, list) else [value])]


def _check_reference(value: Any, field: str) -> None:
    """Refuse reference values that are neither an http(s) URL nor a data: URI, and a data:
    URI over the per-file limit. The scheme is all that is checked: the engine parses the
    rest, and refuses what it cannot fetch or decode."""
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
        except ValueError:                  # e.g. an unclosed IPv6 host, "http://[::1"
            scheme = None
        if scheme not in REFERENCE_SCHEMES:
            raise JsonDataException({field: "must be an http(s) URL or a data: URI"})
        if scheme == "data" and (len(item) - MAX_DATA_URI_PREFIX) * 3 // 4 > MAX_UPLOAD_BYTES:
            raise JsonDataException({field: f"larger than {MAX_UPLOAD_BYTES} bytes"})


def _drop_empty(fields: Dict[str, Any], keys: tuple) -> None:
    """An empty reference means none: drop it rather than send the engine an empty value."""
    for key in keys:
        if fields.get(key) == "":
            del fields[key]


def _parse_body(json_msg: Any) -> Dict[str, Any]:
    """The request as a dict, unwrapping the `input` wrapper the other routes accept."""
    if not isinstance(json_msg, dict):
        raise JsonDataException({"payload": "must be an object"})
    fields = request_parser(json_msg)
    if not isinstance(fields, dict):
        raise JsonDataException({"input": "must be an object"})
    return dict(fields)


def _fill_model(fields: Dict[str, Any]) -> None:
    # The engine needs a model id; on-demand templates set only the engine var.
    if not fields.get("model"):
        model = _resolve_model_name()
        if model:
            fields["model"] = model


class _UploadPayload(ApiPayload):
    """Shared by the routes whose spec body is multipart: never sent as JSON."""

    def generate_payload_json(self) -> Dict[str, Any]:
        # Never reached: generate_payload_multipart() returns a mapping, so the backend
        # takes the multipart branch. Loud rather than posting JSON the engine rejects.
        raise NotImplementedError("this payload is sent as multipart, not JSON")


class TranscriptionPayload(_UploadPayload):
    """base64 in, multipart out — the envelope is JSON, the OpenAI spec wants a file.

        {"file": "<base64>", "filename": "a.mp3", "model": "...", "language": "en"}

    `filename` sets the content type; engines infer the audio format from it. Other
    fields pass through as form fields.
    """

    def __init__(self, fields: Dict[str, Any], audio: bytes, filename: str):
        self.fields = fields
        self.audio = audio
        self.filename = filename

    @classmethod
    def for_test(cls) -> "TranscriptionPayload":
        fields: Dict[str, Any] = {}
        _fill_model(fields)
        return cls(fields=fields, audio=benchmark_speech(), filename="benchmark.wav")

    @classmethod
    def from_json_msg(cls, json_msg: Any) -> "TranscriptionPayload":
        fields = _parse_body(json_msg)
        raw = fields.pop("file", None)
        if raw is None:
            raise JsonDataException({"file": "field missing"})
        audio = _decode_b64(raw, "file")
        part = _file_part(audio, fields.pop("filename", None), DEFAULT_AUDIO_FILENAME,
                          AUDIO_TYPES, "filename")
        _fill_model(fields)
        return cls(fields=fields, audio=audio, filename=part[0])

    def generate_payload_multipart(self) -> Optional[Dict[str, Any]]:
        part = _file_part(self.audio, self.filename, DEFAULT_AUDIO_FILENAME,
                          AUDIO_TYPES, "filename")
        return {"file": part, **self.fields}

    def count_workload(self) -> float:
        return _in_request_units(_audio_seconds(self.audio, self.filename),
                                 REF_AUDIO_SECONDS)


class ImageEditPayload(_UploadPayload):
    """base64 in, multipart out, for /v1/images/edits.

        {"image": "<b64>" | ["<b64>", ...], "filename": "a.png" | [...],
         "mask": "<b64>", "mask_filename": "m.png", "prompt": "..."}

    Engines also accept `url` (or `url[]`) instead of an upload; one of the two is
    required.
    """

    def __init__(self, fields: Dict[str, Any], files: Dict[str, list]):
        self.fields = fields
        self.files = files

    @classmethod
    def for_test(cls) -> "ImageEditPayload":
        """One reference-sized edit, weighed by the same _image_workload as generations."""
        side = REF_IMAGE_SIDE
        fields: Dict[str, Any] = {"prompt": _words(60), "size": f"{side}x{side}", "n": 1}
        _fill_model(fields)
        part = _file_part(synthetic_png(side), DEFAULT_IMAGE_FILENAME,
                          DEFAULT_IMAGE_FILENAME, IMAGE_TYPES, "filename")
        return cls(fields=fields, files={"image": [part]})

    @classmethod
    def from_json_msg(cls, json_msg: Any) -> "ImageEditPayload":
        fields = _parse_body(json_msg)
        files: Dict[str, list] = {}

        images = fields.pop("image", None)
        names = fields.pop("filename", None)
        mask = fields.pop("mask", None)
        mask_name = fields.pop("mask_filename", None)
        images = images if isinstance(images, list) or images is None else [images]
        if images is not None and len(images) > MAX_UPLOAD_FILES:
            raise JsonDataException({"image": f"at most {MAX_UPLOAD_FILES} files"})
        _drop_empty(fields, ("url", "url[]"))
        urls = [fields.get("url"), fields.get("url[]")]
        _check_budget([*(images or []), mask, *_flatten(urls)], "image")
        for key in ("url", "url[]"):
            _check_reference(fields.get(key), key)
        if images is not None:
            names = names if isinstance(names, list) else [names]
            names = names + [None] * (len(images) - len(names))
            files["image"] = [
                _file_part(_decode_b64(img, "image"), name, DEFAULT_IMAGE_FILENAME,
                           IMAGE_TYPES, "filename")
                for img, name in zip(images, names)
            ]

        if mask is not None:
            files["mask"] = [_file_part(_decode_b64(mask, "mask"), mask_name,
                                        DEFAULT_IMAGE_FILENAME, IMAGE_TYPES,
                                        "mask_filename")]

        if not files.get("image") and not (fields.get("url") or fields.get("url[]")):
            raise JsonDataException({"image": "field missing (or pass `url`)"})
        _fill_model(fields)
        return cls(fields=fields, files=files)

    def generate_payload_multipart(self) -> Optional[Dict[str, Any]]:
        return {**self.files, **self.fields}

    def count_workload(self) -> float:
        return _image_workload(self.fields)


# /v1/videos/sync file fields: the formats each takes, and the name an unnamed upload
# gets. `<field>_filename` names each upload, as `filename` does on image edits.
# input_references is the one field that repeats.
VIDEO_FILE_FIELDS = {
    "input_reference": (VISUAL_TYPES, DEFAULT_IMAGE_FILENAME),
    "input_references": (MEDIA_TYPES, DEFAULT_IMAGE_FILENAME),
    "control_reference": (VISUAL_TYPES, DEFAULT_IMAGE_FILENAME),
    "source_video": (VIDEO_TYPES, "video.mp4"),
    "source_audio": (AUDIO_TYPES, DEFAULT_AUDIO_FILENAME),
    "video_noise_mask": (VISUAL_TYPES, DEFAULT_IMAGE_FILENAME),
    "audio_noise_mask": (MEDIA_TYPES, DEFAULT_AUDIO_FILENAME),
}
REPEATED_VIDEO_FILE_FIELDS = ("input_references",)
# Reference objects ({"image_url": ...} or {"file_id": ...}, or a list of them), and the
# key in each that the engine fetches.
VIDEO_REFERENCE_FIELDS = {"image_reference": "image_url", "video_reference": "video_url",
                          "audio_reference": "audio_url"}
# Form fields the engine parses as JSON: sent as one JSON string each, since the SDK
# would repeat a list as separate fields.
VIDEO_JSON_FIELDS = (*VIDEO_REFERENCE_FIELDS, "lora", "extra_params")


class VideoPayload(_UploadPayload):
    """base64 in, form out, for /v1/videos/sync, which answers with the mp4 itself.

        {"prompt": "...", "input_reference": "<b64>", "input_reference_filename": "a.png",
         "image_reference": {"image_url": "https://..."}, "width": 832, "height": 480,
         "num_frames": 33}

    Uploads are the VIDEO_FILE_FIELDS; the other fields are form fields, as the engine
    takes them.
    """

    def __init__(self, fields: Dict[str, Any], files: Dict[str, Any]):
        self.fields = fields
        self.files = files

    @classmethod
    def for_test(cls) -> "VideoPayload":
        fields: Dict[str, Any] = {"prompt": _words(60), "width": REF_VIDEO_WIDTH,
                                  "height": REF_VIDEO_HEIGHT, "num_frames": REF_VIDEO_FRAMES}
        _fill_model(fields)
        return cls(fields=fields, files={})

    @classmethod
    def from_json_msg(cls, json_msg: Any) -> "VideoPayload":
        fields = _parse_body(json_msg)
        uploads = {name: fields.pop(name, None) for name in VIDEO_FILE_FIELDS}
        names = {name: fields.pop(f"{name}_filename", None) for name in VIDEO_FILE_FIELDS}
        refs = {}
        for name, key in VIDEO_REFERENCE_FIELDS.items():
            value = fields.get(name)
            items = value if isinstance(value, list) else [value]
            if value is not None and (len(items) > MAX_UPLOAD_FILES
                                      or not all(isinstance(i, dict) for i in items)):
                raise JsonDataException(
                    {name: f"must be an object, or a list of at most {MAX_UPLOAD_FILES}"})
            refs[name] = [i.get(key) for i in items if isinstance(i, dict)]
        _check_budget([*_flatten(list(uploads.values())), *_flatten(list(refs.values()))],
                      "input_reference")
        for name, urls in refs.items():
            _check_reference(urls, f"{name}.{VIDEO_REFERENCE_FIELDS[name]}")

        files: Dict[str, Any] = {}
        for name, value in uploads.items():
            if value is None:
                continue
            repeated = name in REPEATED_VIDEO_FILE_FIELDS
            values = value if isinstance(value, list) else [value]
            if len(values) > (MAX_UPLOAD_FILES if repeated else 1):
                raise JsonDataException(
                    {name: f"at most {MAX_UPLOAD_FILES} files" if repeated else "one file"})
            given = names[name] if isinstance(names[name], list) else [names[name]]
            given = given + [None] * (len(values) - len(given))
            types, default = VIDEO_FILE_FIELDS[name]
            parts = [_file_part(_decode_b64(v, name), n, default, types, f"{name}_filename")
                     for v, n in zip(values, given)]
            files[name] = parts if repeated else parts[0]

        for name in VIDEO_JSON_FIELDS:
            if isinstance(fields.get(name), (dict, list)):
                fields[name] = json.dumps(fields[name])
        _fill_model(fields)
        return cls(fields=fields, files=files)

    def generate_payload_multipart(self) -> Optional[Dict[str, Any]]:
        return {**self.files, **self.fields}

    def count_workload(self) -> float:
        return _video_workload(self.fields)


def _unwrap_input(request: Any) -> Any:
    """The shared parser's envelope, for routes where `input` is itself a field: unwrap
    only a dict, which is never valid text to speak or embed."""
    return request["input"] if isinstance(request, dict) and isinstance(
        request.get("input"), dict) else request


def speech_request_parser(request: Any) -> Dict[str, Any]:
    """Validation for /v1/audio/speech, where `input` is the text to synthesise."""
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
    """Validation for /v1/audio/speech/batch: `ref_audio` may be set for the batch and
    on each item, and every one of them is checked as on /v1/audio/speech."""
    request = _unwrap_input(request)
    if not isinstance(request, dict):
        raise JsonDataException({"payload": "must be an object"})
    items = request.get("items")
    if not isinstance(items, list) or not all(isinstance(i, dict) for i in items):
        raise JsonDataException({"items": "must be a list of objects"})
    for item in (request, *items):
        _drop_empty(item, ("ref_audio",))
    refs = [item.get("ref_audio") for item in (request, *items)]
    _check_budget(refs, "ref_audio")
    for ref in refs:
        _check_reference(ref, "ref_audio")
    return request


def _image_workload(data: Dict[str, Any]) -> float:
    """n x pixels, in request units. `size` is "WxH", or "auto" when the engine decides.
    Both are caller-declared, so they are bounded before use."""
    n = _bounded_int(data.get("n") or 1, 1, MAX_IMAGES, 1)
    pixels = REF_IMAGE_PIXELS
    w, sep, h = str(data.get("size") or "").lower().partition("x")
    if sep:
        pixels = (_bounded_int(w, 1, MAX_IMAGE_SIDE, 1024)
                  * _bounded_int(h, 1, MAX_IMAGE_SIDE, 1024))
    return _in_request_units(n * pixels, REF_IMAGE_PIXELS)


def _speech_chars(data: Dict[str, Any], batch_ref: Any = None) -> int:
    """Input text, plus one reference-sized request per voice-clone reference.

    A reference costs a speaker-encoder pass the text does not account for. `ref_audio`
    is a URL or a data: URI and may be a list, so references are counted rather than
    measured: a URL's length says nothing about the file it names. A batch's own
    `ref_audio` stands in for an item's.
    """
    text = data.get("input")
    chars = len(text) if isinstance(text, str) else 0
    refs = data.get("ref_audio") or batch_ref
    refs = refs if isinstance(refs, list) else [refs]
    refs = [*refs, data.get("ref_audio_2")]
    return chars + REF_SPEECH_CHARS * sum(1 for r in refs if isinstance(r, str) and r)


def _speech_workload(data: Dict[str, Any]) -> float:
    return _in_request_units(_speech_chars(data), REF_SPEECH_CHARS)


def _speech_batch_workload(data: Dict[str, Any]) -> float:
    """The batch's items, summed; clamped as one request, since it is one."""
    items = data.get("items")
    items = [i for i in items if isinstance(i, dict)] if isinstance(items, list) else []
    chars = sum(_speech_chars(item, data.get("ref_audio")) for item in items)
    return _in_request_units(chars, REF_SPEECH_CHARS)


def _chat_batch_workload(data: Dict[str, Any]) -> float:
    """max_tokens for each conversation in the batch, as on /v1/chat/completions."""
    messages = data.get("messages")
    count = len(messages) if isinstance(messages, list) else 0
    tokens = data.get("max_tokens") or data.get("max_completion_tokens") or 0
    return _bounded_int(tokens, 0, 10**9, 0) * count * _bounded_int(data.get("n") or 1, 1, 128, 1)


def _audio_generate_workload(data: Dict[str, Any]) -> float:
    """Seconds of audio asked for; the engine's own length when unset."""
    try:
        seconds = float(data.get("audio_length") or REF_AUDIO_GEN_SECONDS)
    except (TypeError, ValueError, OverflowError):
        seconds = REF_AUDIO_GEN_SECONDS
    if not seconds == seconds:                          # NaN
        seconds = REF_AUDIO_GEN_SECONDS
    seconds = min(max(seconds, 0.0), MAX_AUDIO_GEN_SECONDS)
    return _in_request_units(seconds, REF_AUDIO_GEN_SECONDS)


def _video_workload(data: Dict[str, Any]) -> float:
    """Pixels x frames x outputs, in request units. Sizes the model picks for itself
    (and `size`, `seconds`, `fps` left unset) count as the reference video."""
    pixels = REF_VIDEO_PIXELS
    w, sep, h = str(data.get("size") or "").lower().partition("x")
    if data.get("width") and data.get("height"):
        w, h, sep = data["width"], data["height"], "x"
    if sep:
        pixels = (_bounded_int(w, 1, MAX_IMAGE_SIDE, REF_VIDEO_WIDTH)
                  * _bounded_int(h, 1, MAX_IMAGE_SIDE, REF_VIDEO_HEIGHT))
    frames = REF_VIDEO_FRAMES
    if data.get("num_frames"):
        frames = _bounded_int(data["num_frames"], 1, MAX_VIDEO_FRAMES, REF_VIDEO_FRAMES)
    elif data.get("seconds"):
        fps = _bounded_int(data.get("fps") or REF_VIDEO_FPS, 1, 240, REF_VIDEO_FPS)
        frames = _bounded_int(data["seconds"], 1, MAX_VIDEO_FRAMES, 1) * fps
    outputs = _bounded_int(data.get("num_outputs_per_prompt") or 1, 1, MAX_VIDEO_OUTPUTS, 1)
    return _in_request_units(pixels * frames * outputs, REF_VIDEO_PIXELS * REF_VIDEO_FRAMES)


def _embeddings_workload(data: Dict[str, Any]) -> float:
    """Size of the embedding input, in request units.

    `input` is a string, an array of strings, an array of tokens, or an array of token
    arrays. A batch costs about its sum, so items are summed; tokens are converted to
    characters so every shape lands in one scale.
    """
    value = data.get("input")
    items = value if isinstance(value, (list, tuple)) else [value]
    chars = 0
    for item in items:
        if isinstance(item, str):
            chars += len(item)
        elif isinstance(item, (list, tuple)):
            chars += len(item) * CHARS_PER_TOKEN      # a token array
        elif isinstance(item, int):
            chars += CHARS_PER_TOKEN                  # a bare token
    return _in_request_units(chars, REF_EMBED_CHARS)


UPLOAD_ROUTES = ("/v1/images/edits", "/v1/videos/sync",
                 "/v1/audio/transcriptions", "/v1/audio/translations")
# Served when OPENAI_ROUTES is unset, as before this worker had other routes, plus
# whichever route BENCHMARK_ROUTE names.
DEFAULT_ROUTES = ("/v1/completions", "/v1/chat/completions")


def benchmark_route() -> str:
    """The route this deployment is benchmarked on: only the template knows which of the
    routes an engine serves the endpoint exists for."""
    route = os.environ.get("BENCHMARK_ROUTE", "").strip() or DEFAULT_BENCHMARK_ROUTE
    if route not in BENCHMARKS:
        raise RuntimeError(
            f"BENCHMARK_ROUTE={route!r} cannot be benchmarked; expected one of "
            + ", ".join(BENCHMARKS))
    return route


def _served_routes(handlers: List[HandlerConfig]) -> List[HandlerConfig]:
    """The handlers to serve.

    OPENAI_ROUTES (comma-separated) chooses them; unset serves DEFAULT_ROUTES plus the
    benchmarked route, so an existing deployment serves exactly what it did before. Every
    instance pulls this worker from main at boot, so on an SDK that cannot send multipart
    it degrades instead of failing: the upload routes are not served, and answer 404.
    """
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
    """The WorkerConfig kwargs for an engine. Separate from run() so the handler table
    can be inspected without starting a server."""

    # Relative path resolves against the server url+port; a full URL is used as-is.
    healthcheck_url = os.environ.get("MODEL_HEALTH_ENDPOINT", "/health")
    # Exactly one route carries a BenchmarkConfig, which is what the SDK requires.
    benchmarked = benchmark_route()

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
        # `input` is a real field on speech and embeddings: they unwrap only a dict.
        route("/v1/audio/speech", workload_calculator=_speech_workload,
              request_parser=speech_request_parser),
        route("/v1/audio/speech/batch", workload_calculator=_speech_batch_workload,
              request_parser=speech_batch_request_parser),
        # Text to music or sound: `input` is the prompt.
        route("/v1/audio/generate", workload_calculator=_audio_generate_workload,
              request_parser=_unwrap_input),
        route("/v1/embeddings", workload_calculator=_embeddings_workload,
              request_parser=_unwrap_input),
        route("/v1/images/generations", workload_calculator=_image_workload,
              request_parser=request_parser),
        # Payload classes apply their own parsing; request_parser is ignored with them.
        route("/v1/images/edits", payload_class=ImageEditPayload),
        route("/v1/audio/transcriptions", payload_class=TranscriptionPayload),
        route("/v1/audio/translations", payload_class=TranscriptionPayload),
        route("/v1/videos/sync", payload_class=VideoPayload),
    ]
    # Routes an engine does not implement answer 404 from behind the worker.

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
