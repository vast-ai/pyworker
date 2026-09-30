"""Benchmark payloads for the OpenAI worker, one per route that can be benchmarked.

BENCHMARK_ROUTE names the route a deployment is benchmarked on (default /v1/completions,
so LLM workers benchmark exactly what they did before). Every benchmark request weighs
one reference request in the worker's workload units, so the score means the same thing
whichever route is benchmarked.
"""

import os
import random
import struct
import wave
import zlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable, List, Optional

import nltk

# One template serves both lanes; on-demand templates set only the engine var, so the
# benchmark recovers the model id from it. Only one is ever set.
_MODEL_NAME_VARS = ("MODEL_NAME", "VLLM_MODEL", "SGLANG_MODEL", "LLAMA_MODEL")

nltk.download("words")
WORD_LIST = nltk.corpus.words.words()

REF_IMAGE_SIDE = 1024      # one reference image, per side
REF_AUDIO_SECONDS = 30.0   # one reference clip: one Whisper window
REF_AUDIO_GEN_SECONDS = 10.0   # one reference generated clip (/v1/audio/generate)
# One reference video: a short clip at a size every Wan-family pipeline accepts
# (sides a multiple of 32, 4k+1 frames).
REF_VIDEO_WIDTH, REF_VIDEO_HEIGHT, REF_VIDEO_FRAMES = 832, 480, 33
BATCH_BENCHMARK_ITEMS = 4  # a batch benchmark splits one reference request this many ways
# One reference embedding request. Sized for a 256-token encoder (all-MiniLM-L6): engines
# reject an over-length input rather than truncate it, and tokens per character vary
# with the draw, so a reference near the limit fails by luck.
REF_EMBED_CHARS = int(os.environ.get("BENCHMARK_EMBED_CHARS") or 600)   # empty = unset


def resolve_model_name() -> Optional[str]:
    return next((v for var in _MODEL_NAME_VARS if (v := os.environ.get(var))), None)


def _model() -> dict:
    model = resolve_model_name()
    return {"model": model} if model else {}


def _words(chars: int) -> str:
    """Random dictionary words, about `chars` characters long."""
    out: List[str] = []
    size = 0
    while size < chars:
        word = random.choice(WORD_LIST)
        out.append(word)
        size += len(word) + 1
    return " ".join(out)


def _voice() -> dict:
    # Engines disagree on voice names, so the benchmark sends none unless told which.
    voice = os.environ.get("BENCHMARK_SPEECH_VOICE")
    return {"voice": voice} if voice else {}


def synthetic_png(side: int = REF_IMAGE_SIDE, tile: int = 64) -> bytes:
    """A random RGB PNG, built from the stdlib: the input to the edit benchmark.

    A diffusion edit's cost depends on resolution and steps, not on the pixels, so
    synthetic input measures the same work as a photograph. Tiled from one random block
    so it is cheap to build (for_test() runs inside the timed window), and re-rolled per
    call because engines cache multimodal input by content hash.
    """
    rows = [os.urandom(tile * 3) * (side // tile) for _ in range(tile)]
    raw = b"".join(b"\x00" + rows[y % tile] for y in range(side))   # filter 0 per row

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0)   # 8-bit RGB
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(raw, 1)) + chunk(b"IEND", b""))


# Real speech for the transcription benchmark: noise leaves the decoder idle, which
# overstated throughput 2x on vLLM whisper-large-v3 and 27x on faster-whisper. 10.7 s of
# our own text, synthesised with openbmb/VoxCPM2 (Apache-2.0), 16 kHz mono 16-bit.
_SPEECH_PATH = Path(__file__).with_name("benchmark_speech.wav")
_speech = None


def benchmark_speech(seconds: float = REF_AUDIO_SECONDS) -> bytes:
    """The speech clip tiled to `seconds`, as WAV. Its last few samples are random per
    call, because engines cache multimodal input by content hash."""
    global _speech
    if _speech is None:
        with wave.open(str(_SPEECH_PATH)) as w:
            _speech = (w.getparams(), w.readframes(w.getnframes()))
    params, frames = _speech
    want = int(seconds * params.framerate) * params.sampwidth * params.nchannels
    data = (frames * (want // len(frames) + 1))[:want - 8] + os.urandom(8)
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setparams(params)
        w.writeframes(data)
    return buf.getvalue()


def completions_benchmark_generator() -> dict:
    model = resolve_model_name()
    if not model:
        raise ValueError("No model set: MODEL_NAME / VLLM_MODEL / SGLANG_MODEL / LLAMA_MODEL all empty")
    prompt = " ".join(random.choices(WORD_LIST, k=250))
    return {"model": model, "prompt": prompt, "temperature": 0.7, "max_tokens": 500}


def chat_benchmark_generator() -> dict:
    prompt = " ".join(random.choices(WORD_LIST, k=250))
    return {**_model(), "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7, "max_tokens": 500}


def embeddings_benchmark_generator() -> dict:
    return {**_model(), "input": _words(REF_EMBED_CHARS)}


def speech_benchmark_generator() -> dict:
    return {**_model(), "input": _words(500), **_voice()}


def chat_batch_benchmark_generator() -> dict:
    """One reference request split across a batch: BATCH_BENCHMARK_ITEMS conversations
    whose max_tokens sum to the chat benchmark's."""
    k = BATCH_BENCHMARK_ITEMS
    return {**_model(),
            "messages": [[{"role": "user", "content": " ".join(random.choices(WORD_LIST, k=250 // k))}]
                         for _ in range(k)],
            "temperature": 0.7, "max_tokens": 500 // k}


def speech_batch_benchmark_generator() -> dict:
    """One reference request split across a batch: the items' text sums to the speech
    benchmark's."""
    k = BATCH_BENCHMARK_ITEMS
    return {**_model(), "items": [{"input": _words(500 // k)} for _ in range(k)], **_voice()}


def audio_generate_benchmark_generator() -> dict:
    return {**_model(), "input": _words(60), "audio_length": REF_AUDIO_GEN_SECONDS}


def images_benchmark_generator() -> dict:
    return {**_model(), "prompt": _words(60), "size": f"{REF_IMAGE_SIDE}x{REF_IMAGE_SIDE}", "n": 1}


@dataclass(frozen=True)
class Benchmark:
    concurrency: int
    runs: int
    # None: the route's payload class builds its benchmark payloads (for_test).
    generator: Optional[Callable[[], dict]] = None


BENCHMARKS = {
    "/v1/completions": Benchmark(10, 3, completions_benchmark_generator),
    "/v1/chat/completions": Benchmark(10, 3, chat_benchmark_generator),
    "/v1/embeddings": Benchmark(10, 3, embeddings_benchmark_generator),
    "/v1/audio/speech": Benchmark(4, 2, speech_benchmark_generator),
    "/v1/audio/speech/batch": Benchmark(4, 2, speech_batch_benchmark_generator),
    "/v1/chat/completions/batch": Benchmark(10, 3, chat_batch_benchmark_generator),
    "/v1/audio/generate": Benchmark(2, 1, audio_generate_benchmark_generator),
    "/v1/images/generations": Benchmark(2, 1, images_benchmark_generator),
    # Uploads: the payload class builds these. Every served route has a benchmark, so a
    # deployment narrowed to any one route (an edit-only model, say) can become ready.
    "/v1/images/edits": Benchmark(2, 1),
    "/v1/audio/transcriptions": Benchmark(4, 2),
    "/v1/audio/translations": Benchmark(4, 2),
    "/v1/videos/sync": Benchmark(1, 1),
}
DEFAULT_BENCHMARK_ROUTE = "/v1/completions"
