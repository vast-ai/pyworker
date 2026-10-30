"""Benchmark payloads, one per route. Each weighs one reference request, so the score
means the same whichever route BENCHMARK_ROUTE names."""

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

from workers.openai import routes

# One template serves both lanes; on-demand templates set only the engine var, so the
# benchmark recovers the model id from it. Only one is ever set.
_MODEL_NAME_VARS = ("MODEL_NAME", "VLLM_MODEL", "SGLANG_MODEL", "LLAMA_MODEL")

nltk.download("words")
WORD_LIST = nltk.corpus.words.words()

REF_IMAGE_SIDE = 1024      # one reference image, per side
REF_AUDIO_SECONDS = 30.0   # one reference clip: one Whisper window
REF_AUDIO_GEN_SECONDS = 10.0
# Song models (SPEECH_KIND=music) count max_new_tokens in frames, 25 a second.
BENCHMARK_SONG_FRAMES = 750
BENCHMARK_SONG_STYLE = "pop, female vocal, piano, bass and drums, 110 BPM, uplifting"
BATCH_BENCHMARK_ITEMS = 4  # a batch benchmark splits one reference request this many ways
# A rerank or score request is one query against these documents; each pair fits a
# 512-token reranker.
RERANK_DOCS, RERANK_QUERY_CHARS, RERANK_DOC_CHARS = 16, 60, 250
REF_RERANK_CHARS = RERANK_DOCS * (RERANK_QUERY_CHARS + RERANK_DOC_CHARS)
# Sized for a 256-token encoder: engines refuse an over-length input.
try:
    REF_EMBED_CHARS = int(os.environ.get("BENCHMARK_EMBED_CHARS") or 600)   # empty = unset
except ValueError:
    REF_EMBED_CHARS = 0
if REF_EMBED_CHARS <= 0:
    print(f"WARNING: BENCHMARK_EMBED_CHARS={os.environ['BENCHMARK_EMBED_CHARS']!r} is not a "
          "positive integer; using 600", flush=True)
    REF_EMBED_CHARS = 600


def resolve_model_name() -> Optional[str]:
    return next((v for var in _MODEL_NAME_VARS if (v := os.environ.get(var))), None)


def _model() -> dict:
    model = resolve_model_name()
    return {"model": model} if model else {}


def _words(chars: int) -> str:
    """Random dictionary words, about `chars` characters long. Sized in characters, not
    words, because the embeddings, rerank, score and speech workloads count characters:
    a payload of about the reference length weighs about one request."""
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
    """A random RGB PNG, tiled so it is cheap to build and new per call (engines cache
    by content hash)."""
    rows = [os.urandom(tile * 3) * (side // tile) for _ in range(tile)]
    raw = b"".join(b"\x00" + rows[y % tile] for y in range(side))   # filter 0 per row

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0)   # 8-bit RGB
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(raw, 1)) + chunk(b"IEND", b""))


# Real speech: noise leaves the decoder idle and overstates throughput. See benchmark_speech.md.
_SPEECH_PATH = Path(__file__).with_name("benchmark_speech.wav")
_speech = None


def benchmark_speech(seconds: float = REF_AUDIO_SECONDS) -> bytes:
    """The speech clip tiled to `seconds`, with a random tail (engines cache by content hash)."""
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


def song_benchmark_generator() -> dict:
    """Lyrics in input, the style in instructions. A song ends at its own end token or at
    max_new_tokens, and is counted at max_new_tokens either way, so the lyrics run well past
    the frames asked for: a song that ended early would overstate throughput. No voice:
    song models refuse one."""
    sections = ("[Verse]", "[Chorus]", "[Verse]", "[Chorus]", "[Bridge]", "[Chorus]")
    lyrics = "\n\n".join(
        tag + "\n" + "\n".join(" ".join(random.choices(WORD_LIST, k=6)) for _ in range(4))
        for tag in sections)
    return {**_model(), "input": lyrics, "instructions": BENCHMARK_SONG_STYLE,
            "max_new_tokens": BENCHMARK_SONG_FRAMES}


def chat_batch_benchmark_generator() -> dict:
    k = BATCH_BENCHMARK_ITEMS
    return {**_model(),
            "messages": [[{"role": "user", "content": " ".join(random.choices(WORD_LIST, k=250 // k))}]
                         for _ in range(k)],
            "temperature": 0.7, "max_tokens": 500 // k}


def speech_batch_benchmark_generator() -> dict:
    k = BATCH_BENCHMARK_ITEMS
    return {**_model(), "items": [{"input": _words(500 // k)} for _ in range(k)], **_voice()}


def audio_generate_benchmark_generator() -> dict:
    return {**_model(), "input": _words(60), "audio_length": REF_AUDIO_GEN_SECONDS}


def rerank_benchmark_generator() -> dict:
    return {**_model(), "query": _words(RERANK_QUERY_CHARS),
            "documents": [_words(RERANK_DOC_CHARS) for _ in range(RERANK_DOCS)]}


def score_benchmark_generator() -> dict:
    return {**_model(), "queries": _words(RERANK_QUERY_CHARS),
            "items": [_words(RERANK_DOC_CHARS) for _ in range(RERANK_DOCS)]}


def images_benchmark_generator() -> dict:
    return {**_model(), "prompt": _words(60), "size": f"{REF_IMAGE_SIDE}x{REF_IMAGE_SIDE}", "n": 1}


@dataclass(frozen=True)
class Benchmark:
    concurrency: int
    runs: int
    # None: the route's payload class builds its benchmark payloads (for_test).
    generator: Optional[Callable[[], dict]] = None


BENCHMARKS = {
    routes.COMPLETIONS: Benchmark(10, 3, completions_benchmark_generator),
    routes.CHAT: Benchmark(10, 3, chat_benchmark_generator),
    routes.EMBEDDINGS: Benchmark(10, 3, embeddings_benchmark_generator),
    routes.RERANK: Benchmark(10, 3, rerank_benchmark_generator),
    routes.SCORE: Benchmark(10, 3, score_benchmark_generator),
    routes.SPEECH: Benchmark(4, 2, speech_benchmark_generator),
    routes.SPEECH_BATCH: Benchmark(4, 2, speech_batch_benchmark_generator),
    routes.CHAT_BATCH: Benchmark(10, 3, chat_batch_benchmark_generator),
    routes.AUDIO_GENERATE: Benchmark(2, 1, audio_generate_benchmark_generator),
    routes.IMAGES: Benchmark(2, 1, images_benchmark_generator),
    # Upload routes: the payload class builds these.
    routes.IMAGE_EDITS: Benchmark(2, 1),
    routes.TRANSCRIPTIONS: Benchmark(4, 2),
    routes.TRANSLATIONS: Benchmark(4, 2),
    routes.VIDEO: Benchmark(1, 1),
}
# /v1/audio/speech under SPEECH_KIND=music.
SONG_BENCHMARK = Benchmark(2, 1, song_benchmark_generator)
DEFAULT_BENCHMARK_ROUTE = routes.COMPLETIONS
