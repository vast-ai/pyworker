"""workers/openai/core: the handler table, and TranscriptionPayload.

Stdlib unittest so it needs no test dependency the worker does not already have:
    python -m unittest discover -s tests
"""

import base64
import contextlib
import io
import json
import os
import subprocess
import sys
import unittest
from unittest import mock

# This repo is cloned onto instances, where these are set for real; the suite must not
# read them.
os.environ["MODEL_NAME"] = "openai/whisper-large-v3"
for _leak in ("OPENAI_ROUTES", "BENCHMARK_ROUTE", "BENCHMARK_SPEECH_VOICE",
              "BENCHMARK_EMBED_CHARS", "MODEL_LOG", "MODEL_HEALTH_ENDPOINT", "BACKEND",
              "VLLM_MODEL", "SGLANG_MODEL", "LLAMA_MODEL"):
    os.environ.pop(_leak, None)

from vastai.serverless.server.lib.data_types import JsonDataException  # noqa: E402

import workers.openai.core as core  # noqa: E402
from workers.openai.benchmark import BENCHMARKS  # noqa: E402
from workers.openai.core import (  # noqa: E402
    BENCHMARK_MAX_TOKENS,
    MAX_REQUEST_MULTIPLE,
    MIN_REQUEST_MULTIPLE,
    EngineDefaults,
    ImageEditPayload,
    TranscriptionPayload,
    build_config,
    request_parser,
)

WAV = b"RIFF\x24\x00\x00\x00WAVEfmt "


def b64(data=WAV):
    return base64.b64encode(data).decode()


class TestTranscriptionPayloadValidation(unittest.TestCase):
    def test_rejects_bad_file_field(self):
        for label, payload in [
            ("missing", {"model": "m"}),
            ("not a string", {"file": 123}),
            ("not base64", {"file": "!!!"}),
            ("empty", {"file": ""}),
        ]:
            with self.subTest(label), self.assertRaises(JsonDataException):
                TranscriptionPayload.from_json_msg(payload)

    def test_rejects_non_dict_payload(self):
        with self.assertRaises(JsonDataException):
            TranscriptionPayload.from_json_msg(["not", "a", "dict"])


class TestTranscriptionPayloadParsing(unittest.TestCase):
    def test_a_tiny_clip_counts_as_the_floor(self):
        p = TranscriptionPayload.from_json_msg({"file": b64()})
        self.assertEqual(p.count_workload(), BENCHMARK_MAX_TOKENS * MIN_REQUEST_MULTIPLE)

    def test_model_falls_back_to_the_engine_env_var(self):
        p = TranscriptionPayload.from_json_msg({"file": b64()})
        self.assertEqual(p.fields["model"], "openai/whisper-large-v3")
        p = TranscriptionPayload.from_json_msg({"file": b64(), "model": "given"})
        self.assertEqual(p.fields["model"], "given")

    def test_input_wrapped_request_is_unwrapped(self):
        p = TranscriptionPayload.from_json_msg({"input": {"file": b64(), "model": "m"}})
        self.assertEqual(p.fields["model"], "m")


class TestTranscriptionPayloadMultipart(unittest.TestCase):
    def test_file_part_carries_filename_and_type_from_the_extension(self):
        fields = TranscriptionPayload.from_json_msg(
            {"file": b64(), "filename": "clip.mp3"}
        ).generate_payload_multipart()
        self.assertEqual(fields["file"], ("clip.mp3", WAV, "audio/mpeg"))

    def test_unsupported_extension_is_rejected(self):
        for name in ("clip.unknown", "clip.html", "clip"):
            with self.subTest(name), self.assertRaises(JsonDataException):
                TranscriptionPayload.from_json_msg({"file": b64(), "filename": name})

    def test_null_filename_uses_the_default(self):
        fields = TranscriptionPayload.from_json_msg(
            {"file": b64(), "filename": None}).generate_payload_multipart()
        self.assertEqual(fields["file"][:1], ("audio.wav",))
        self.assertEqual(fields["file"][2], "audio/wav")

    def test_filename_is_reduced_to_a_safe_basename(self):
        for given, expected in [("../../etc/cron.d/x.wav", "x.wav"),
                                ('a"\r\nX: 1.wav', "a___X__1.wav"),
                                (".hidden.wav", "hidden.wav")]:
            with self.subTest(given):
                fields = TranscriptionPayload.from_json_msg(
                    {"file": b64(), "filename": given}).generate_payload_multipart()
                self.assertEqual(fields["file"][0], expected)

    def test_other_fields_pass_through(self):
        fields = TranscriptionPayload.from_json_msg(
            {"file": b64(), "language": "en", "temperature": 0}
        ).generate_payload_multipart()
        self.assertEqual(fields["language"], "en")
        self.assertEqual(fields["temperature"], 0)

    def test_json_body_is_refused(self):
        # The backend takes the multipart branch; posting JSON would be silently wrong.
        p = TranscriptionPayload.from_json_msg({"file": b64()})
        with self.assertRaises(NotImplementedError):
            p.generate_payload_json()


DEFAULTS = EngineDefaults(
    name="stub", model_log_file="/tmp/stub.log",
    load_log_msgs=["loaded"], error_log_msgs=["failed"],
)


def _wav(seconds, rate=16000):
    """Silent 16-bit mono WAV: for pricing, where only the header's duration matters."""
    import wave
    from io import BytesIO
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return buf.getvalue()


ALL_ROUTES = {"/v1/completions", "/v1/chat/completions", "/v1/audio/speech",
              "/v1/embeddings", "/v1/images/generations", "/v1/images/edits",
              "/v1/audio/transcriptions", "/v1/audio/translations",
              "/v1/chat/completions/batch", "/v1/audio/speech/batch",
              "/v1/audio/generate", "/v1/videos/sync"}


def handlers():
    """The handler table, serving every route unless the test set OPENAI_ROUTES itself
    (an empty value means the default set)."""
    env = {} if "OPENAI_ROUTES" in os.environ else {"OPENAI_ROUTES": ",".join(sorted(ALL_ROUTES))}
    # The startup lines go to a buffer; a test that asserts on them patches print itself.
    with mock.patch.dict(os.environ, env), contextlib.redirect_stdout(io.StringIO()):
        return {h.route: h for h in build_config(DEFAULTS)["handlers"]}


class TestHandlerTable(unittest.TestCase):
    def test_input_bearing_routes_do_not_use_the_shared_request_parser(self):
        """`input` is a real field in the speech spec (and in embeddings), but the
        shared parser treats a top-level `input` as a wrapper and unwraps it -- which
        would reduce {"model","input","voice"} to the bare text and drop the rest."""
        body = {"model": "t", "input": "Hi", "voice": "a"}
        self.assertEqual(request_parser(body), "Hi")
        for route in ("/v1/audio/speech", "/v1/embeddings"):
            with self.subTest(route):
                parser = handlers()[route].request_parser
                self.assertIsNot(parser, request_parser)
                if parser is not None:
                    self.assertEqual(parser(dict(body)), body)

    def test_speech_and_embeddings_unwrap_a_dict_input(self):
        """The other routes accept the {"input": {...}} envelope; these two must too, but
        only when `input` is a dict, which is never valid text to speak or embed."""
        inner = {"model": "t", "input": "Hi", "voice": "a"}
        for route in ("/v1/audio/speech", "/v1/embeddings"):
            with self.subTest(route):
                parser = handlers()[route].request_parser
                self.assertEqual(parser({"input": dict(inner)}), inner)
                self.assertEqual(parser(dict(inner)), inner)

    def test_wrapper_routes_keep_the_shared_request_parser(self):
        for route in ("/v1/completions", "/v1/chat/completions"):
            with self.subTest(route):
                self.assertIsNotNone(handlers()[route].request_parser)

    def test_exactly_one_route_carries_a_benchmark(self):
        """The SDK requires exactly one, which is why this worker needs no SDK change
        to pick a route."""
        carrying = [r for r, h in handlers().items() if h.benchmark_config]
        self.assertEqual(carrying, ["/v1/completions"])   # the default: LLM unchanged

    def test_benchmark_route_moves_the_benchmark(self):
        """Every route can be the benchmark, so a deployment narrowed to any one of them
        (an edit-only model, say) can become ready."""
        for route in BENCHMARKS:
            with self.subTest(route), mock.patch.dict(os.environ, {"BENCHMARK_ROUTE": route}):
                carrying = [r for r, h in handlers().items() if h.benchmark_config]
                self.assertEqual(carrying, [route])

    def test_an_empty_benchmark_route_is_the_default(self):
        """Templates often list every variable, empty; empty must mean unset, as it
        does for OPENAI_ROUTES, or an existing deployment refuses to boot."""
        with mock.patch.dict(os.environ, {"BENCHMARK_ROUTE": ""}):
            carrying = [r for r, h in handlers().items() if h.benchmark_config]
        self.assertEqual(carrying, ["/v1/completions"])

    def test_a_route_that_cannot_be_benchmarked_is_refused(self):
        with mock.patch.dict(os.environ, {"BENCHMARK_ROUTE": "/v1/images/variations"}):
            with self.assertRaises(RuntimeError):
                handlers()

    def test_a_benchmark_route_that_is_not_served_is_refused(self):
        """Otherwise the worker boots, fails its benchmark on a route it does not
        serve, and reports an engine problem."""
        with mock.patch.dict(os.environ, {"BENCHMARK_ROUTE": "/v1/audio/speech",
                                          "OPENAI_ROUTES": "/v1/completions"}):
            with self.assertRaises(RuntimeError):
                handlers()

    def test_the_benchmark_route_is_logged_at_startup(self):
        with mock.patch("builtins.print") as printed:
            handlers()
        lines = " ".join(str(c.args[0]) for c in printed.call_args_list)
        self.assertIn("benchmarking: /v1/completions", lines)
        self.assertIn("BENCHMARK_ROUTE", lines)


class TestImageEdit(unittest.TestCase):
    PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20

    def b64(self, data=None):
        return base64.b64encode(data or self.PNG).decode()

    def test_single_image_becomes_one_file_part(self):
        fields = ImageEditPayload.from_json_msg(
            {"image": self.b64(), "prompt": "make it blue"}
        ).generate_payload_multipart()
        self.assertEqual(fields["image"], [("image.png", self.PNG, "image/png")])
        self.assertEqual(fields["prompt"], "make it blue")

    def test_multiple_images_become_repeated_parts(self):
        fields = ImageEditPayload.from_json_msg(
            {"image": [self.b64(), self.b64()], "filename": ["a.png", "b.jpg"],
             "prompt": "p"}
        ).generate_payload_multipart()
        self.assertEqual([f[0] for f in fields["image"]], ["a.png", "b.jpg"])
        self.assertEqual([f[2] for f in fields["image"]], ["image/png", "image/jpeg"])

    def test_mask_is_carried_when_present(self):
        fields = ImageEditPayload.from_json_msg(
            {"image": self.b64(), "mask": self.b64(), "prompt": "p"}
        ).generate_payload_multipart()
        self.assertIn("mask", fields)

    def test_mask_has_its_own_filename(self):
        fields = ImageEditPayload.from_json_msg(
            {"image": [self.b64(), self.b64()], "filename": ["a.jpg", "b.jpg"],
             "mask": self.b64(), "mask_filename": "m.png", "prompt": "p"}
        ).generate_payload_multipart()
        self.assertEqual(fields["mask"], [("m.png", self.PNG, "image/png")])

    def test_non_image_types_are_rejected(self):
        for name in ("x.svg", "x.html", "x.wav"):
            with self.subTest(name), self.assertRaises(JsonDataException):
                ImageEditPayload.from_json_msg(
                    {"image": self.b64(), "filename": name, "prompt": "p"})

    def test_model_is_filled_from_the_engine_var(self):
        payload = ImageEditPayload.from_json_msg({"image": self.b64(), "prompt": "p"})
        self.assertEqual(payload.fields["model"], "openai/whisper-large-v3")

    def test_url_is_accepted_instead_of_an_upload(self):
        fields = ImageEditPayload.from_json_msg(
            {"url": "https://x/a.png", "prompt": "p"}
        ).generate_payload_multipart()
        self.assertEqual(fields["url"], "https://x/a.png")
        self.assertNotIn("image", fields)

    def test_neither_image_nor_url_is_rejected(self):
        with self.assertRaises(JsonDataException):
            ImageEditPayload.from_json_msg({"prompt": "p"})

    def test_bad_base64_is_rejected(self):
        with self.assertRaises(JsonDataException):
            ImageEditPayload.from_json_msg({"image": "!!!", "prompt": "p"})


ONE_REQUEST = float(BENCHMARK_MAX_TOKENS)
FLOOR = BENCHMARK_MAX_TOKENS * MIN_REQUEST_MULTIPLE
CEILING = BENCHMARK_MAX_TOKENS * MAX_REQUEST_MULTIPLE


class TestImageWorkload(unittest.TestCase):
    def setUp(self):
        self.calc = handlers()["/v1/images/generations"].workload_calculator

    def test_one_reference_image_is_one_request(self):
        self.assertEqual(self.calc({"size": "1024x1024", "n": 1}), ONE_REQUEST)

    def test_scales_with_n_and_pixels(self):
        self.assertEqual(self.calc({"size": "1024x1024", "n": 2}), 2 * ONE_REQUEST)
        self.assertEqual(self.calc({"size": "512x512", "n": 4}), ONE_REQUEST)

    def test_auto_missing_or_unparseable_size_is_one_reference_image(self):
        for size in (None, "auto", "nonsense"):
            with self.subTest(size):
                self.assertEqual(self.calc({"size": size}), ONE_REQUEST)

    def test_is_clamped(self):
        self.assertEqual(self.calc({"size": "64x64"}), FLOOR)
        self.assertEqual(self.calc({"size": "8192x8192", "n": 10}), CEILING)


class TestEmbeddingsWorkload(unittest.TestCase):
    def setUp(self):
        self.calc = handlers()["/v1/embeddings"].workload_calculator

    def test_all_four_input_shapes_land_on_one_scale(self):
        """One reference request of text, however the caller shapes it: a string, a
        batch, pre-tokenised ids, or a batch of those."""
        from workers.openai.benchmark import REF_EMBED_CHARS
        chars, tokens = REF_EMBED_CHARS, REF_EMBED_CHARS // core.CHARS_PER_TOKEN
        for label, value in [
            ("string",          "x" * chars),
            ("list of strings", ["x" * (chars // 2), "x" * (chars // 2)]),
            ("token array",     list(range(tokens))),
            ("array of arrays", [list(range(tokens // 2)), list(range(tokens // 2))]),
        ]:
            with self.subTest(label):
                self.assertEqual(self.calc({"input": value}), ONE_REQUEST)

    def test_a_batch_costs_more_than_one_item(self):
        from workers.openai.benchmark import REF_EMBED_CHARS
        one = self.calc({"input": ["x" * REF_EMBED_CHARS]})
        three = self.calc({"input": ["x" * REF_EMBED_CHARS] * 3})
        self.assertEqual(three, 3 * one)

    def test_the_benchmark_payload_fits_a_short_context_encoder(self):
        """THE defect this sizing exists for, measured on BAAI/bge-small-en-v1.5: at
        2000 chars the payload tokenised to 513 against a 512 limit, vLLM refused it,
        and the worker never became ready. Tokens per character vary with the draw
        (500 chars measured 133-160 tokens over five draws), so the reference is sized
        for all-MiniLM-L6's 256 rather than the 512 of the bge/e5/gte family."""
        from workers.openai.benchmark import REF_EMBED_CHARS, embeddings_benchmark_generator
        worst_case_tokens = len(embeddings_benchmark_generator()["input"]) / 3.1
        self.assertLess(worst_case_tokens, 256 * 0.8,
                        f"{REF_EMBED_CHARS} chars can tokenise past a 256-token encoder")

    def test_an_empty_embed_chars_setting_is_the_default(self):
        """benchmark.py is imported by every OpenAI worker, so an empty value (templates
        often list every variable) must mean unset, not a crash at boot."""
        env = dict(os.environ, BENCHMARK_EMBED_CHARS="", PYTHONPATH=os.pathsep.join(sys.path))
        out = subprocess.run(
            [sys.executable, "-c",
             "from workers.openai.benchmark import REF_EMBED_CHARS; print(REF_EMBED_CHARS)"],
            env=env, capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.split()[-1], "600")

    def test_missing_or_null_input_is_the_floor(self):
        self.assertEqual(self.calc({}), FLOOR)
        self.assertEqual(self.calc({"input": None}), FLOOR)


class TestSpeechWorkload(unittest.TestCase):
    def setUp(self):
        self.calc = handlers()["/v1/audio/speech"].workload_calculator

    def test_reference_length_text_is_one_request(self):
        self.assertEqual(self.calc({"input": "x" * 500}), ONE_REQUEST)

    def test_short_or_missing_text_is_the_floor(self):
        self.assertEqual(self.calc({"input": "hi"}), FLOOR)
        self.assertEqual(self.calc({}), FLOOR)

    def test_each_clone_reference_adds_one_request(self):
        """Counted, not measured: a URL's length says nothing about the file it names,
        so a URL and an inline blob cost the same."""
        base = {"input": "x" * 500}
        for ref in ("https://x/a.wav", "file:///a.wav", "A" * 100_000):
            with self.subTest(ref[:20]):
                self.assertEqual(self.calc({**base, "ref_audio": ref}), 2 * ONE_REQUEST)
        self.assertEqual(self.calc({**base, "ref_audio": ["u1", "u2"]}), 3 * ONE_REQUEST)
        self.assertEqual(self.calc({**base, "ref_audio": "u1", "ref_audio_2": "u2"}),
                         3 * ONE_REQUEST)

    def test_a_clone_request_is_never_free(self):
        plain = self.calc({"input": "hello"})
        clone = self.calc({"input": "hello", "ref_audio": "https://x/a.wav"})
        self.assertGreater(clone, plain)


class TestUploadWorkload(unittest.TestCase):
    """ASR is counted in seconds of audio, which is what it costs and what engines bill,
    and is knowable before the request runs."""

    def test_a_reference_length_clip_is_one_request(self):
        from workers.openai.benchmark import benchmark_speech

        audio = base64.b64encode(benchmark_speech()).decode()
        payload = TranscriptionPayload.from_json_msg({"file": audio,
                                                      "filename": "a.wav"})
        self.assertAlmostEqual(payload.count_workload(), ONE_REQUEST, delta=1)

    def test_a_clip_with_no_parsable_header_falls_back_to_bytes(self):
        """The fallback must still produce a number, since refusing to price a request
        is worse than pricing it approximately."""
        audio = base64.b64encode(b"\x00" * (320 * 1000)).decode()
        payload = TranscriptionPayload.from_json_msg({"file": audio,
                                                      "filename": "a.webm"})
        self.assertGreater(payload.count_workload(), 0)


class TestUploadLimits(unittest.TestCase):
    def test_oversized_upload_is_rejected_before_decoding(self):
        with mock.patch.object(core, "MAX_UPLOAD_BYTES", 1024), \
                mock.patch.object(core.base64, "b64decode",
                                  side_effect=AssertionError("decoded")):
            with self.assertRaises(JsonDataException):
                TranscriptionPayload.from_json_msg({"file": "A" * 10_000})

    def test_data_uri_and_line_wrapped_base64_are_accepted(self):
        wrapped = base64.encodebytes(WAV * 20).decode()      # 76-column lines
        for value in (f"data:audio/wav;base64,{b64()}", wrapped):
            with self.subTest(value[:20]):
                TranscriptionPayload.from_json_msg({"file": value})

    def test_an_oversized_data_uri_prefix_is_rejected(self):
        """The comma is looked for in the first 128 characters only, so a prefix of any
        length is refused without the data being scanned or copied."""
        huge = "data:" + "A" * (2 * core.MAX_UPLOAD_BYTES) + "," + b64()
        with self.assertRaises(JsonDataException):
            TranscriptionPayload.from_json_msg({"file": huge})

    def test_a_data_uri_prefix_is_case_insensitive(self):
        """URI schemes are case-insensitive; an upper-case DATA: was refused as bad base64."""
        payload = TranscriptionPayload.from_json_msg({"file": "DATA:audio/wav;base64," + b64()})
        self.assertTrue(payload.audio.startswith(b"RIFF"))

    def test_a_long_filename_keeps_its_extension(self):
        """Truncating mid-extension would reject a legitimate upload for having none."""
        name = "a" * 140 + ".wav"
        payload = TranscriptionPayload.from_json_msg({"file": b64(), "filename": name})
        part = payload.generate_payload_multipart()["file"]
        self.assertTrue(part[0].endswith(".wav"))
        self.assertEqual(part[2], "audio/wav")

    def test_too_many_images_is_rejected(self):
        img = base64.b64encode(b"\x89PNG").decode()
        with self.assertRaises(JsonDataException):
            ImageEditPayload.from_json_msg({"image": [img] * 17, "prompt": "p"})

    def test_non_object_input_is_a_422_not_a_500(self):
        for payload in ({"input": "text"}, ["a"], "x"):
            with self.subTest(str(payload)):
                with self.assertRaises(JsonDataException):
                    TranscriptionPayload.from_json_msg(payload)
                with self.assertRaises(JsonDataException):
                    ImageEditPayload.from_json_msg(payload)


class TestTranscriptionBenchmarkPayload(unittest.TestCase):
    def test_for_test_uploads_the_speech_sample(self):
        """The benchmark must send real speech: noise or silence leaves the decoder idle
        and overstates throughput. Checked against the bundled sample, since a valid
        30 s WAV of anything else would pass every other check."""
        import wave
        from io import BytesIO
        from workers.openai import benchmark

        fields = TranscriptionPayload.for_test().generate_payload_multipart()
        name, audio, ctype = fields["file"]
        self.assertEqual((name, ctype), ("benchmark.wav", "audio/wav"))
        self.assertEqual(fields["model"], "openai/whisper-large-v3")
        with wave.open(str(benchmark._SPEECH_PATH)) as w:
            sample = w.readframes(w.getnframes())
        with wave.open(BytesIO(audio)) as w:
            self.assertEqual(w.readframes(len(sample) // 2), sample)

    def test_the_clip_is_audio_of_the_reference_size(self):
        """A shorter clip makes the benchmark weigh less than one request, so the score
        is in a different unit from every other candidate's."""
        import wave
        from io import BytesIO
        from workers.openai.benchmark import REF_AUDIO_SECONDS, benchmark_speech

        clip = benchmark_speech()
        with wave.open(BytesIO(clip)) as w:
            self.assertEqual((w.getnchannels(), w.getsampwidth(), w.getframerate()),
                             (1, 2, 16000))
            self.assertEqual(w.getnframes(), int(REF_AUDIO_SECONDS * 16000))
            frames = w.readframes(w.getnframes())
        self.assertNotEqual(frames, b"\x00" * len(frames), "clip is digital silence")

    def test_every_candidate_benchmarks_one_reference_request(self):
        """The score means the same thing whichever route wins only if every candidate's
        benchmark request weighs the same."""
        weigh = {"/v1/completions": lambda b: b.get("max_tokens", 0),
                 "/v1/chat/completions": lambda b: b.get("max_tokens", 0),
                 "/v1/embeddings": core._embeddings_workload,
                 "/v1/audio/speech": core._speech_workload,
                 "/v1/images/generations": core._image_workload,
                 "/v1/images/edits":
                     lambda _b: core.ImageEditPayload.for_test().count_workload(),
                 "/v1/audio/transcriptions":
                     lambda _b: TranscriptionPayload.for_test().count_workload(),
                 "/v1/audio/translations":
                     lambda _b: TranscriptionPayload.for_test().count_workload(),
                 "/v1/chat/completions/batch": core._chat_batch_workload,
                 "/v1/audio/speech/batch": core._speech_batch_workload,
                 "/v1/audio/generate": core._audio_generate_workload,
                 "/v1/videos/sync":
                     lambda _b: core.VideoPayload.for_test().count_workload()}
        for route, b in BENCHMARKS.items():
            with self.subTest(route):
                body = b.generator() if b.generator else None
                weight = weigh[route](body)
                self.assertGreaterEqual(weight, 0.5 * core.BENCHMARK_MAX_TOKENS)
                self.assertLessEqual(weight, 2 * core.BENCHMARK_MAX_TOKENS)


class TestEditBenchmark(unittest.TestCase):
    """Some models are edit-only. Without an edits benchmark such a deployment could
    neither benchmark on edits (BENCHMARK_ROUTE refused it at startup) nor on
    generations (which its model does not serve), so it could never become ready."""

    @staticmethod
    def _decode(png: bytes):
        import struct
        import zlib
        assert png[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
        pos, chunks = 8, {}
        while pos < len(png):
            n = struct.unpack(">I", png[pos:pos + 4])[0]
            kind, data = png[pos + 4:pos + 8], png[pos + 8:pos + 8 + n]
            crc = struct.unpack(">I", png[pos + 8 + n:pos + 12 + n])[0]
            assert crc == zlib.crc32(kind + data) & 0xFFFFFFFF, f"bad CRC on {kind}"
            chunks.setdefault(kind, b"")
            chunks[kind] += data
            pos += 12 + n
        w, h, depth, colour = struct.unpack(">IIBB", chunks[b"IHDR"][:10])
        return w, h, depth, colour, zlib.decompress(chunks[b"IDAT"])

    def test_the_input_is_a_valid_reference_sized_png(self):
        from workers.openai.benchmark import REF_IMAGE_SIDE, synthetic_png
        w, h, depth, colour, raw = self._decode(synthetic_png())
        self.assertEqual((w, h, depth, colour), (REF_IMAGE_SIDE, REF_IMAGE_SIDE, 8, 2))
        self.assertEqual(len(raw), h * (1 + w * 3), "scanlines do not match the header")
        pixels = bytes(b for i, b in enumerate(raw) if i % (1 + w * 3))
        self.assertGreater(len(set(pixels)), 200, "the image is effectively uniform")

    def test_the_input_is_re_rolled_per_call(self):
        """Engines cache processed multimodal input by content hash; a fixed image would
        measure the cache after the first request."""
        from workers.openai.benchmark import synthetic_png
        self.assertNotEqual(synthetic_png(), synthetic_png())

    def test_the_input_is_tiled_from_one_block(self):
        """Built inside the SDK's timed window, so it is tiled rather than drawn pixel by
        pixel; rows repeating every tile is what that looks like."""
        from workers.openai.benchmark import synthetic_png
        w, h, _depth, _colour, raw = self._decode(synthetic_png(side=256, tile=64))
        stride = 1 + w * 3
        rows = [raw[y * stride:(y + 1) * stride] for y in range(h)]
        self.assertEqual(rows[0], rows[64])
        self.assertNotEqual(rows[0], rows[1])

    def test_for_test_sends_an_image_part_and_the_fields(self):
        body = core.ImageEditPayload.for_test().generate_payload_multipart()
        self.assertIn("image", body)
        filename, data, ctype = body["image"][0]
        self.assertEqual(ctype, "image/png")
        self.assertEqual(data[:4], b"\x89PNG")
        self.assertTrue(body.get("prompt"))
        self.assertEqual(body.get("n"), 1)


class TestEveryServedRouteCanBeBenchmarked(unittest.TestCase):
    def test_no_served_route_lacks_a_benchmark(self):
        """A deployment can narrow OPENAI_ROUTES to any single route; one with no
        benchmark could never become ready."""
        self.assertEqual(ALL_ROUTES - set(BENCHMARKS), set())


class TestRequestBudget(unittest.TestCase):
    def test_image_uploads_over_the_request_budget_are_refused_before_decoding(self):
        img = "A" * 4096
        with mock.patch.object(core, "MAX_REQUEST_UPLOAD_BYTES", 8000), \
                mock.patch.object(core.base64, "b64decode",
                                  side_effect=AssertionError("decoded")):
            with self.assertRaises(JsonDataException):
                ImageEditPayload.from_json_msg({"image": [img] * 3, "prompt": "p"})

    def test_inline_references_count_against_the_budget(self):
        ref = "A" * 4096
        parser = handlers()["/v1/audio/speech"].request_parser
        with mock.patch.object(core, "MAX_REQUEST_UPLOAD_BYTES", 5000), \
                mock.patch.object(core.base64, "b64decode",
                                  side_effect=AssertionError("decoded")):
            with self.assertRaises(JsonDataException):
                parser({"input": "hi", "ref_audio": [ref], "ref_audio_2": ref})


class TestReferences(unittest.TestCase):
    """References accept only http(s) URLs and data: URIs, so nothing reaches the engine
    as a value it might open on the instance's own disk."""
    INLINE = base64.b64encode(b"RIFF" + b"\x00" * 200).decode()
    ALLOWED = ("https://x/a.png", "http://x/a.png", f"data:audio/wav;base64,{INLINE}",
               f"DATA:audio/wav;base64,{INLINE}")
    REFUSED = ("file:///root/.ssh/id_rsa", "file:/etc/passwd", "ftp://x/a", "gopher://x",
               "/etc/passwd", "/tmp/abcd/efgh", "../../workspace/x.wav",
               "\\\\host\\share\\a.wav",
               # Bare base64, refused outright: "/" is a base64 character, so a path padded
               # with slashes decodes, and a size check alone let these through.
               INLINE, "/" * 78 + "etc/passwd", "/" * 71 + "proc/self/environ")

    def test_image_edit_urls(self):
        for key in ("url", "url[]"):
            for value in self.ALLOWED:
                with self.subTest((key, value)):
                    ImageEditPayload.from_json_msg({key: [value], "prompt": "p"})
            for value in self.REFUSED:
                with self.subTest((key, value)), self.assertRaises(JsonDataException):
                    ImageEditPayload.from_json_msg({key: value, "prompt": "p"})

    def test_a_malformed_url_is_a_json_error(self):
        """urlparse raises on an unclosed IPv6 host; that must be a 422, not a 500."""
        parser = handlers()["/v1/audio/speech"].request_parser
        with self.assertRaises(JsonDataException):
            parser({"input": "hi", "ref_audio": "http://[::1"})
        with self.assertRaises(JsonDataException):
            ImageEditPayload.from_json_msg({"url": "http://[::1", "prompt": "p"})

    def test_an_oversized_data_uri_reference_is_refused(self):
        """A data: reference is forwarded, not decoded, so the per-file limit is checked on
        its length; otherwise only the per-request budget, over twice as large, applies."""
        big = "data:audio/wav;base64," + "A" * (4 * (core.MAX_UPLOAD_BYTES // 3 + 64))
        parser = handlers()["/v1/audio/speech"].request_parser
        with self.assertRaisesRegex(JsonDataException, "larger than"):
            parser({"input": "hi", "ref_audio": big})

    def test_an_empty_reference_is_dropped(self):
        """Empty means none: the engine is not sent an empty reference to refuse."""
        parser = handlers()["/v1/audio/speech"].request_parser
        self.assertNotIn("ref_audio", parser({"input": "hi", "ref_audio": ""}))
        payload = ImageEditPayload.from_json_msg(
            {"image": b64(), "url": "", "prompt": "p"})
        self.assertNotIn("url", payload.generate_payload_multipart())

    def test_speech_references(self):
        parser = handlers()["/v1/audio/speech"].request_parser
        for key in ("ref_audio", "ref_audio_2"):
            for value in self.ALLOWED:
                with self.subTest((key, value)):
                    parser({"input": "hi", key: value})
            for value in self.REFUSED:
                with self.subTest((key, value)), self.assertRaises(JsonDataException):
                    parser({"input": "hi", key: value})


class TestServedRoutes(unittest.TestCase):
    def test_the_default_is_what_llm_deployments_served_before(self):
        """Every instance pulls this worker from main at boot, so the default must not
        change what an existing deployment serves: completions and chat, plus the route
        it is benchmarked on."""
        with mock.patch.dict(os.environ, {"OPENAI_ROUTES": ""}):
            self.assertEqual(set(handlers()), {"/v1/completions", "/v1/chat/completions"})
        with mock.patch.dict(os.environ, {"OPENAI_ROUTES": "",
                                          "BENCHMARK_ROUTE": "/v1/audio/transcriptions"}):
            self.assertEqual(set(handlers()), {"/v1/completions", "/v1/chat/completions",
                                               "/v1/audio/transcriptions"})

    def test_every_route_is_served_when_asked(self):
        self.assertEqual(set(handlers()), ALL_ROUTES)

    def test_openai_routes_narrows(self):
        with mock.patch.dict(os.environ, {"OPENAI_ROUTES": "/v1/audio/speech",
                                          "BENCHMARK_ROUTE": "/v1/audio/speech"}):
            self.assertEqual(set(handlers()), {"/v1/audio/speech"})

    def test_unknown_route_names_are_reported(self):
        with mock.patch.dict(os.environ,
                             {"OPENAI_ROUTES": "/v1/audio/speech,/v1/embedding",
                              "BENCHMARK_ROUTE": "/v1/audio/speech"}), \
                mock.patch("builtins.print") as printed:
            handlers()
        lines = " ".join(str(c.args[0]) for c in printed.call_args_list)
        self.assertIn("unknown routes: /v1/embedding", lines)
        self.assertIn("serving: /v1/audio/speech", lines)

    def test_an_old_sdk_is_only_reported_when_an_upload_route_is_asked_for(self):
        """Until the SDK ships multipart, every LLM deployment boots on one; an ERROR
        about routes it never asked for would be noise on every one of them."""
        class OldApiPayload:
            pass

        for routes, reported in (("", False), ("/v1/completions,/v1/images/edits", True)):
            with self.subTest(routes), mock.patch.object(core, "ApiPayload", OldApiPayload), \
                    mock.patch.dict(os.environ, {"OPENAI_ROUTES": routes}), \
                    mock.patch("builtins.print") as printed:
                handlers()
            lines = " ".join(str(c.args[0]) for c in printed.call_args_list)
            self.assertEqual("cannot send multipart" in lines, reported)

    def test_an_upload_benchmark_route_on_an_old_sdk_names_the_sdk(self):
        """The route is dropped because the SDK cannot send it, so the error must say
        so; "widen OPENAI_ROUTES" would send the operator the wrong way."""
        class OldApiPayload:
            pass

        with mock.patch.object(core, "ApiPayload", OldApiPayload), \
                mock.patch.dict(os.environ, {"BENCHMARK_ROUTE": "/v1/audio/transcriptions"}), \
                self.assertRaisesRegex(RuntimeError, "multipart"):
            handlers()

    def test_upload_routes_are_dropped_on_an_sdk_without_multipart(self):
        class OldApiPayload:
            pass

        with mock.patch.object(core, "ApiPayload", OldApiPayload):
            served = handlers()
        self.assertEqual(set(served), ALL_ROUTES - set(core.UPLOAD_ROUTES))
        # derived from the table, so dropping a route from UPLOAD_ROUTES fails here
        # instead of leaving it served and 500ing on every request
        multipart_only = {r for r, h in handlers().items()
                          if h.payload_class is not None
                          and issubclass(h.payload_class, core._UploadPayload)}
        self.assertEqual(multipart_only, set(core.UPLOAD_ROUTES))
        self.assertFalse(any(h.payload_class is not None
                             and issubclass(h.payload_class, core._UploadPayload)
                             for h in served.values()))


REALISTIC_MAX_THROUGHPUT = 200.0   # tokens/s: 10 x 500-token completions in ~25s


def wait_time_with_one_in_flight(workload):
    """The SDK's own wait_time for a worker holding exactly one request."""
    from vastai.serverless.server.lib.data_types import ModelMetrics, RequestMetrics

    metrics = ModelMetrics.empty()
    metrics.max_throughput = REALISTIC_MAX_THROUGHPUT
    metrics.requests_working[1] = RequestMetrics(
        request_idx=1, reqnum=1, workload=workload, status="Started")
    return metrics.wait_time


class TestAdmissionGate(unittest.TestCase):
    """wait_time sums in-flight workload across EVERY route and divides by a
    max_throughput measured on the benchmark route alone; the SDK 429s any request that
    arrives while it exceeds max_queue_time. So a single legal request on any route
    must not be able to push it over on its own, or one upload 429s the whole worker."""

    WORST_CASES = {
        "/v1/audio/speech": {"input": "x" * 100_000,
                             "ref_audio": ["A" * 10_000] * 10,
                             "ref_audio_2": "https://x/a.wav"},
        "/v1/embeddings": {"input": ["x" * 10_000] * 1_000},
        "/v1/images/generations": {"n": 100_000, "size": "8192x8192"},
        "/v1/audio/speech/batch": {"items": [{"input": "x" * 100_000}] * 1_000,
                                   "ref_audio": "https://x/a.wav"},
        "/v1/audio/generate": {"audio_length": 1e308},
    }

    def test_no_single_request_trips_the_gate(self):
        h = handlers()
        for route, data in self.WORST_CASES.items():
            with self.subTest(route):
                wt = wait_time_with_one_in_flight(h[route].workload_calculator(data))
                self.assertLess(wt, h[route].max_queue_time)

    def test_a_large_upload_does_not_trip_the_gate(self):
        h = handlers()
        big = base64.b64encode(b"\x00" * (25 * 1024 * 1024)).decode()   # OpenAI's file limit
        for route, payload in [
            ("/v1/audio/transcriptions", TranscriptionPayload.from_json_msg({"file": big})),
            ("/v1/images/edits", ImageEditPayload.from_json_msg(
                {"image": [big] * 2, "prompt": "p", "n": 100_000, "size": "8192x8192"})),
        ]:
            with self.subTest(route):
                wt = wait_time_with_one_in_flight(payload.count_workload())
                self.assertLess(wt, h[route].max_queue_time)

    def test_a_large_video_does_not_trip_the_gate(self):
        payload = core.VideoPayload.from_json_msg(
            {"prompt": "p", "width": 8192, "height": 8192, "num_frames": 10**9,
             "num_outputs_per_prompt": 10})
        wt = wait_time_with_one_in_flight(payload.count_workload())
        self.assertLess(wt, handlers()["/v1/videos/sync"].max_queue_time)

    def test_absurd_declared_sizes_do_not_raise(self):
        calc = handlers()["/v1/images/generations"].workload_calculator
        for data in ({"size": "9" * 4000 + "x" + "9" * 4000},
                     {"n": "9" * 5000, "size": "1024x1024"},
                     {"n": -5, "size": "-10x-10"}):
            with self.subTest(str(data)[:40]):
                self.assertGreaterEqual(calc(data), FLOOR)

    def test_malformed_speech_input_does_not_raise(self):
        calc = handlers()["/v1/audio/speech"].workload_calculator
        for data in ({"input": 123}, {"input": "hi", "ref_audio": 5},
                     {"input": ["a", "b"]}, {"ref_audio": [None, 7, ""]}):
            with self.subTest(str(data)):
                self.assertGreater(calc(data), 0)

    def test_completions_units_are_unchanged(self):
        # Existing scores must not move: the benchmarked route keeps max_tokens.
        calc = handlers()["/v1/completions"].workload_calculator
        self.assertEqual(calc({"max_tokens": 500}), 500)


class TestAudioDuration(unittest.TestCase):
    """WAV is read exactly; other containers are estimated from their own byte rate."""

    def test_wav_is_exact(self):
        """At 48 kHz, so the byte-rate estimate (which assumes 16 kHz mono) would be 3x
        off: a 16 kHz clip lands on the right answer by coincidence and proves nothing."""
        self.assertAlmostEqual(core._audio_seconds(_wav(7.5, rate=48000), "a.wav"),
                               7.5, places=2)

    def test_an_unparsable_container_uses_its_own_byte_rate(self):
        """webm has no cheap header parse, so it is priced from bytes -- per container,
        which is the part that was wrong before: one global constant charged 26s of MP3
        and 349s of Opus identically."""
        raw = b"\x1aE\xdf\xa3" + b"\x00" * 80000
        ogg_guess = core._audio_seconds(raw, "a.webm")
        self.assertAlmostEqual(ogg_guess, 80004 / core.AUDIO_BYTES_PER_SECOND["webm"],
                               places=2)

    def test_a_byte_rate_exists_for_every_accepted_format(self):
        """A format the worker accepts but cannot price would fall to the default."""
        for ext in core.AUDIO_TYPES:
            with self.subTest(ext):
                self.assertIn(ext, core.AUDIO_BYTES_PER_SECOND)


class TestBenchmarkClipCost(unittest.TestCase):
    """The SDK builds each benchmark payload inside its timed window, so the clip must be
    cheap to build, and distinct per request so an engine's cache cannot answer it."""

    def test_the_clip_is_the_speech_sample_tiled(self):
        """Real speech, not noise: noise leaves the decoder idle and overstates
        throughput. Tiled from the bundled sample, which keeps it cheap to build."""
        import wave
        from io import BytesIO
        from workers.openai import benchmark

        with wave.open(str(benchmark._SPEECH_PATH)) as w:
            sample = w.readframes(w.getnframes())
        with wave.open(BytesIO(benchmark.benchmark_speech())) as w:
            frames = w.readframes(w.getnframes())
        self.assertEqual(frames[:len(sample)], sample)
        self.assertEqual(frames[len(sample):2 * len(sample)], sample)

    def test_every_clip_differs(self):
        """Engines cache processed multimodal input by content hash. Identical audio
        every request would measure that cache from the second request onward."""
        import hashlib
        from workers.openai.benchmark import benchmark_speech

        digests = {hashlib.sha256(benchmark_speech()).hexdigest() for _ in range(4)}
        self.assertEqual(len(digests), 4)


if __name__ == "__main__":
    unittest.main()


class TestBatchRoutes(unittest.TestCase):
    def test_a_chat_batch_costs_each_conversation(self):
        calc = handlers()["/v1/chat/completions/batch"].workload_calculator
        convo = [{"role": "user", "content": "hi"}]
        self.assertEqual(calc({"messages": [convo] * 4, "max_tokens": 100}), 400)
        self.assertEqual(calc({"messages": [convo] * 2, "max_tokens": 100, "n": 3}), 600)
        self.assertEqual(calc({"messages": [convo], "max_completion_tokens": 50}), 50)

    def test_a_malformed_chat_batch_does_not_raise(self):
        calc = handlers()["/v1/chat/completions/batch"].workload_calculator
        for data in ({}, {"messages": "x", "max_tokens": 5}, {"messages": [[]], "max_tokens": "a"},
                     {"messages": [[]], "max_tokens": -5, "n": "z"}):
            with self.subTest(str(data)):
                self.assertGreaterEqual(calc(data), 0)

    def test_a_speech_batch_sums_its_items_and_counts_the_batch_reference(self):
        calc = handlers()["/v1/audio/speech/batch"].workload_calculator
        item = {"input": "x" * core.REF_SPEECH_CHARS}
        self.assertEqual(calc({"items": [item] * 2}), 2 * core.BENCHMARK_MAX_TOKENS)
        # the batch reference stands in for each item's: one encoder pass per item
        self.assertEqual(calc({"items": [item] * 2, "ref_audio": "https://x/a.wav"}),
                         4 * core.BENCHMARK_MAX_TOKENS)
        self.assertEqual(calc({"items": "x"}), FLOOR)

    def test_every_speech_batch_reference_is_checked(self):
        parse = handlers()["/v1/audio/speech/batch"].request_parser
        for label, data in [
            ("batch", {"items": [{"input": "a"}], "ref_audio": "/etc/passwd"}),
            ("item", {"items": [{"input": "a", "ref_audio": "/etc/passwd"}]}),
            ("not a list", {"items": {"input": "a"}}),
            ("not objects", {"items": ["a"]}),
        ]:
            with self.subTest(label), self.assertRaises(JsonDataException):
                parse(data)
        ok = parse({"items": [{"input": "a", "ref_audio": ""}], "ref_audio": "https://x/a.wav"})
        self.assertNotIn("ref_audio", ok["items"][0])

    def test_the_batch_benchmarks_split_one_reference_request(self):
        for route in ("/v1/chat/completions/batch", "/v1/audio/speech/batch"):
            with self.subTest(route):
                body = BENCHMARKS[route].generator()
                key = "messages" if "chat" in route else "items"
                self.assertEqual(len(body[key]), 4)


class TestAudioGenerate(unittest.TestCase):
    def test_workload_is_seconds_of_audio(self):
        calc = handlers()["/v1/audio/generate"].workload_calculator
        ref = core.REF_AUDIO_GEN_SECONDS
        self.assertEqual(calc({"audio_length": ref}), core.BENCHMARK_MAX_TOKENS)
        self.assertEqual(calc({"audio_length": 2 * ref}), 2 * core.BENCHMARK_MAX_TOKENS)
        self.assertEqual(calc({}), core.BENCHMARK_MAX_TOKENS)       # the engine's own length
        for bad in ("x", float("nan"), None, -3, [1]):
            with self.subTest(bad):
                self.assertGreaterEqual(calc({"audio_length": bad}), FLOOR)

    def test_input_is_the_prompt_not_a_wrapper(self):
        parse = handlers()["/v1/audio/generate"].request_parser
        self.assertEqual(parse({"input": "rain on a tin roof"})["input"], "rain on a tin roof")


class TestVideo(unittest.TestCase):
    PNG = None

    def setUp(self):
        from workers.openai.benchmark import synthetic_png
        self.png = base64.b64encode(synthetic_png(64)).decode()

    def test_an_upload_becomes_one_file_part_and_the_rest_form_fields(self):
        p = core.VideoPayload.from_json_msg(
            {"prompt": "a cat", "input_reference": self.png,
             "input_reference_filename": "cat.png", "num_frames": 33})
        body = p.generate_payload_multipart()
        name, raw, ctype = body["input_reference"]
        self.assertEqual((name, ctype), ("cat.png", "image/png"))
        self.assertEqual(body["prompt"], "a cat")
        self.assertEqual(body["num_frames"], 33)
        self.assertEqual(body["model"], "openai/whisper-large-v3")

    def test_input_references_repeat_and_single_fields_do_not(self):
        p = core.VideoPayload.from_json_msg(
            {"prompt": "p", "input_references": [self.png, self.png],
             "input_references_filename": ["a.png", "b.png"]})
        self.assertEqual([f[0] for f in p.files["input_references"]], ["a.png", "b.png"])
        with self.assertRaises(JsonDataException):
            core.VideoPayload.from_json_msg({"prompt": "p", "source_video": [self.png] * 2})

    def test_each_field_takes_only_its_formats(self):
        with self.assertRaises(JsonDataException):
            core.VideoPayload.from_json_msg(
                {"prompt": "p", "source_audio": self.png, "source_audio_filename": "x.png"})
        p = core.VideoPayload.from_json_msg({"prompt": "p", "source_audio": b64(_wav(1))})
        self.assertEqual(p.files["source_audio"][2], "audio/wav")

    def test_references_are_sent_as_one_json_field_and_their_urls_checked(self):
        refs = [{"image_url": "https://x/a.png"}, {"file_id": "file-1"}]
        p = core.VideoPayload.from_json_msg({"prompt": "p", "image_reference": refs,
                                             "lora": {"name": "l", "path": "hub/x"}})
        self.assertEqual(json.loads(p.fields["image_reference"]), refs)
        self.assertEqual(json.loads(p.fields["lora"]), {"name": "l", "path": "hub/x"})
        for bad in ({"image_url": "/etc/passwd"}, ["not an object"], "https://x/a.png"):
            with self.subTest(str(bad)), self.assertRaises(JsonDataException):
                core.VideoPayload.from_json_msg({"prompt": "p", "image_reference": bad})

    def test_uploads_count_against_the_request_budget(self):
        big = "A" * (core.MAX_REQUEST_UPLOAD_BYTES // 3 * 4 // 4)
        with self.assertRaisesRegex(JsonDataException, "in total"):
            core.VideoPayload.from_json_msg(
                {"prompt": "p", "input_references": [big] * 5,
                 "video_reference": {"video_url": "data:video/mp4;base64," + big}})

    def test_workload_is_pixels_frames_and_outputs(self):
        ref = core.VideoPayload.for_test().count_workload()
        self.assertEqual(ref, core.BENCHMARK_MAX_TOKENS)
        w = lambda **f: core._video_workload(f)   # noqa: E731
        self.assertEqual(w(), ref)                                      # model's own size
        self.assertEqual(w(width=832, height=480, num_frames=66), 2 * ref)
        self.assertEqual(w(size="832x480", num_frames=33, num_outputs_per_prompt=2), 2 * ref)
        self.assertEqual(w(width=832, height=480, seconds="4", fps=16),
                         core._in_request_units(832 * 480 * 64, core.REF_VIDEO_PIXELS * 33))
        for bad in ({"width": "x", "height": 5}, {"num_frames": -1}, {"seconds": "a"},
                    {"size": "9" * 400 + "x1"}, {"num_outputs_per_prompt": "9" * 5000}):
            with self.subTest(str(bad)[:30]):
                self.assertGreaterEqual(core._video_workload(bad), FLOOR)
