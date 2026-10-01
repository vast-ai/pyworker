"""workers/openai: python -m unittest discover -s tests"""

import base64
import contextlib
import hashlib
import io
import json
import os
import struct
import subprocess
import sys
import unittest
import wave
import zlib
from unittest import mock

# Instances set these for real; the suite must not read them.
os.environ["MODEL_NAME"] = "openai/whisper-large-v3"
for _leak in ("OPENAI_ROUTES", "BENCHMARK_ROUTE", "BENCHMARK_SPEECH_VOICE",
              "BENCHMARK_EMBED_CHARS", "MODEL_LOG", "MODEL_HEALTH_ENDPOINT", "BACKEND",
              "VLLM_MODEL", "SGLANG_MODEL", "LLAMA_MODEL"):
    os.environ.pop(_leak, None)

from vastai.serverless.server.lib.data_types import (  # noqa: E402
    JsonDataException, ModelMetrics, RequestMetrics)

import workers.openai.core as core  # noqa: E402
from workers.openai import benchmark  # noqa: E402
from workers.openai.benchmark import BENCHMARKS  # noqa: E402
from workers.openai.core import (  # noqa: E402
    BENCHMARK_MAX_TOKENS,
    MAX_REQUEST_MULTIPLE,
    MIN_REQUEST_MULTIPLE,
    EngineDefaults,
    ImageEditPayload,
    TranscriptionPayload,
    VideoPayload,
    build_config,
    request_parser,
)

WAV = b"RIFF\x24\x00\x00\x00WAVEfmt "
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
ONE_REQUEST = float(BENCHMARK_MAX_TOKENS)
FLOOR = BENCHMARK_MAX_TOKENS * MIN_REQUEST_MULTIPLE
CEILING = BENCHMARK_MAX_TOKENS * MAX_REQUEST_MULTIPLE
CONVO = [{"role": "user", "content": "hi"}]

DEFAULTS = EngineDefaults(name="stub", model_log_file="/tmp/stub.log",
                          load_log_msgs=["loaded"], error_log_msgs=["failed"])
ALL_ROUTES = {"/v1/completions", "/v1/chat/completions", "/v1/chat/completions/batch",
              "/v1/audio/speech", "/v1/audio/speech/batch", "/v1/audio/generate",
              "/v1/embeddings", "/v1/images/generations", "/v1/images/edits",
              "/v1/audio/transcriptions", "/v1/audio/translations", "/v1/videos/sync",
              "/v1/rerank", "/v1/score"}


def b64(data=WAV):
    return base64.b64encode(data).decode()


def _wav(seconds, rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return buf.getvalue()


def handlers():
    """Every route, unless the test sets OPENAI_ROUTES itself."""
    env = {} if "OPENAI_ROUTES" in os.environ else {"OPENAI_ROUTES": ",".join(sorted(ALL_ROUTES))}
    with mock.patch.dict(os.environ, env), contextlib.redirect_stdout(io.StringIO()):
        return {h.route: h for h in build_config(DEFAULTS)["handlers"]}


def startup_log(env):
    with mock.patch.dict(os.environ, env), mock.patch("builtins.print") as printed:
        handlers()
    return " ".join(str(c.args[0]) for c in printed.call_args_list)


class OldApiPayload:
    """An SDK without multipart support."""


def wait_time_with_one_in_flight(workload):
    """The SDK's own wait_time, at 200 tokens/s: 10 x 500-token completions in ~25 s."""
    metrics = ModelMetrics.empty()
    metrics.max_throughput = 200.0
    metrics.requests_working[1] = RequestMetrics(
        request_idx=1, reqnum=1, workload=workload, status="Started")
    return metrics.wait_time


class TestUploads(unittest.TestCase):
    def test_a_bad_file_is_refused(self):
        for label, payload in [("missing", {"model": "m"}), ("not a string", {"file": 123}),
                               ("not base64", {"file": "!!!"}), ("empty", {"file": ""})]:
            with self.subTest(label), self.assertRaises(JsonDataException):
                TranscriptionPayload.from_json_msg(payload)

    def test_non_object_input_is_a_422_not_a_500(self):
        for payload in ({"input": "text"}, ["a"], "x"):
            for cls in (TranscriptionPayload, ImageEditPayload, VideoPayload):
                with self.subTest((cls.__name__, str(payload))), \
                        self.assertRaises(JsonDataException):
                    cls.from_json_msg(payload)

    def test_the_file_type_comes_from_the_extension(self):
        part = TranscriptionPayload.from_json_msg(
            {"file": b64(), "filename": "clip.mp3"}).generate_payload_multipart()["file"]
        self.assertEqual(part, ("clip.mp3", WAV, "audio/mpeg"))
        for name in ("clip.unknown", "clip.html", "clip"):
            with self.subTest(name), self.assertRaises(JsonDataException):
                TranscriptionPayload.from_json_msg({"file": b64(), "filename": name})

    def test_the_filename_is_reduced_to_a_safe_basename(self):
        for given, expected in [("../../etc/cron.d/x.wav", "x.wav"),
                                ('a"\r\nX: 1.wav', "a___X__1.wav"),
                                (".hidden.wav", "hidden.wav"),
                                (None, "audio.wav"),
                                ("a" * 140 + ".wav", "a" * 128 + ".wav")]:
            with self.subTest(given):
                part = TranscriptionPayload.from_json_msg(
                    {"file": b64(), "filename": given}).generate_payload_multipart()["file"]
                self.assertEqual(part[0], expected)

    def test_other_fields_pass_through_and_the_model_is_filled(self):
        body = TranscriptionPayload.from_json_msg(
            {"input": {"file": b64(), "language": "en"}}).generate_payload_multipart()
        self.assertEqual((body["language"], body["model"]), ("en", "openai/whisper-large-v3"))
        body = TranscriptionPayload.from_json_msg({"file": b64(), "model": "given"})
        self.assertEqual(body.fields["model"], "given")

    def test_an_oversized_upload_is_refused_before_decoding(self):
        with mock.patch.object(core, "MAX_UPLOAD_BYTES", 1024), \
                mock.patch.object(core.base64, "b64decode", side_effect=AssertionError):
            with self.assertRaises(JsonDataException):
                TranscriptionPayload.from_json_msg({"file": "A" * 10_000})

    def test_data_uris_and_line_wrapped_base64_are_accepted(self):
        for value in (f"data:audio/wav;base64,{b64()}", f"DATA:audio/wav;base64,{b64()}",
                      base64.encodebytes(WAV * 20).decode()):
            with self.subTest(value[:20]):
                TranscriptionPayload.from_json_msg({"file": value})

    def test_an_oversized_data_uri_prefix_is_refused(self):
        huge = "data:" + "A" * (2 * core.MAX_UPLOAD_BYTES) + "," + b64()
        with self.assertRaises(JsonDataException):
            TranscriptionPayload.from_json_msg({"file": huge})

    def test_uploads_over_the_request_budget_are_refused_before_decoding(self):
        ref = "A" * 4096
        speech = handlers()["/v1/audio/speech"].request_parser
        cases = [
            lambda: ImageEditPayload.from_json_msg({"image": [ref] * 3, "prompt": "p"}),
            lambda: speech({"input": "hi", "ref_audio": [ref, ref], "ref_audio_2": ref}),
            lambda: VideoPayload.from_json_msg(
                {"prompt": "p", "input_references": [ref, ref],
                 "video_reference": {"video_url": "data:video/mp4;base64," + ref}}),
        ]
        with mock.patch.object(core, "MAX_REQUEST_UPLOAD_BYTES", 8000), \
                mock.patch.object(core.base64, "b64decode", side_effect=AssertionError):
            for i, case in enumerate(cases):
                with self.subTest(i), self.assertRaisesRegex(JsonDataException, "in total"):
                    case()


class TestImageEdit(unittest.TestCase):
    def test_images_become_repeated_parts_with_their_own_names(self):
        body = ImageEditPayload.from_json_msg(
            {"image": [b64(PNG), b64(PNG)], "filename": ["a.png", "b.jpg"],
             "mask": b64(PNG), "mask_filename": "m.webp", "prompt": "p"}
        ).generate_payload_multipart()
        self.assertEqual([(f[0], f[2]) for f in body["image"]],
                         [("a.png", "image/png"), ("b.jpg", "image/jpeg")])
        self.assertEqual(body["mask"], [("m.webp", PNG, "image/webp")])
        self.assertNotIn("mask_filename", body)
        self.assertEqual(body["prompt"], "p")
        with self.assertRaises(JsonDataException):
            ImageEditPayload.from_json_msg({"image": b64(PNG), "filename": "x.wav", "prompt": "p"})
        with self.assertRaises(JsonDataException):
            ImageEditPayload.from_json_msg({"image": [b64(PNG)] * 17, "prompt": "p"})
        body = ImageEditPayload.from_json_msg(
            {"image": [b64(PNG), b64(PNG)], "filename": "a.png", "prompt": "p"}).files
        self.assertEqual([f[0] for f in body["image"]], ["a.png", "image.png"])

    def test_a_url_stands_in_for_an_upload(self):
        body = ImageEditPayload.from_json_msg(
            {"url": "https://x/a.png", "prompt": "p"}).generate_payload_multipart()
        self.assertEqual(body["url"], "https://x/a.png")
        self.assertNotIn("image", body)
        with self.assertRaises(JsonDataException):
            ImageEditPayload.from_json_msg({"prompt": "p"})


class TestVideo(unittest.TestCase):
    PNG = b64(benchmark.synthetic_png(64))

    def test_uploads_become_file_parts_and_the_rest_form_fields(self):
        body = VideoPayload.from_json_msg(
            {"prompt": "a cat", "input_reference": self.PNG,
             "input_reference_filename": "cat.png", "num_frames": 33,
             "input_references": [self.PNG, self.PNG],
             "input_references_filename": ["a.png", "b.png"]}).generate_payload_multipart()
        self.assertEqual(body["input_reference"][0::2], ("cat.png", "image/png"))
        self.assertEqual([f[0] for f in body["input_references"]], ["a.png", "b.png"])
        self.assertEqual((body["prompt"], body["num_frames"]), ("a cat", 33))
        with self.assertRaises(JsonDataException):
            VideoPayload.from_json_msg({"prompt": "p", "source_video": [self.PNG] * 2})

    def test_each_field_takes_only_its_formats(self):
        with self.assertRaises(JsonDataException):
            VideoPayload.from_json_msg(
                {"prompt": "p", "source_audio": self.PNG, "source_audio_filename": "x.png"})
        mask = b64(json.dumps({"frames": [0]}).encode())
        p = VideoPayload.from_json_msg({"prompt": "p", "source_audio": b64(_wav(1)),
                                        "video_noise_mask": mask,
                                        "audio_noise_mask": mask,
                                        "audio_noise_mask_filename": "m.json"})
        self.assertEqual(p.files["source_audio"][2], "audio/wav")
        self.assertEqual(p.files["video_noise_mask"][0::2], ("mask.json", "application/json"))
        self.assertEqual(p.files["audio_noise_mask"][0], "m.json")
        p = VideoPayload.from_json_msg({"prompt": "p", "input_references": [self.PNG] * 2,
                                        "input_references_filename": ["clip.mp4", "voice.wav"]})
        self.assertEqual([f[2] for f in p.files["input_references"]], ["video/mp4", "audio/wav"])
        with self.assertRaises(JsonDataException):
            VideoPayload.from_json_msg({"prompt": "p", "control_reference": self.PNG,
                                        "control_reference_filename": "x.wav"})

    def test_a_list_of_references_is_sent_as_one_json_field(self):
        refs = [{"image_url": "https://x/a.png"}, {"file_id": "file-1"}]
        p = VideoPayload.from_json_msg({"prompt": "p", "image_reference": refs,
                                        "video_reference": {"file_id": "file-2"}})
        self.assertEqual(json.loads(p.fields["image_reference"]), refs)
        self.assertEqual(p.fields["video_reference"], {"file_id": "file-2"})
        for bad in (["not an object"], "https://x/a.png"):
            with self.subTest(bad), self.assertRaises(JsonDataException):
                VideoPayload.from_json_msg({"prompt": "p", "image_reference": bad})


class TestReferences(unittest.TestCase):
    """Only http(s) URLs and data: URIs, so nothing names a path on the instance."""
    INLINE = b64(b"RIFF" + b"\x00" * 200)
    ALLOWED = ("https://x/a.png", "http://x/a.png", f"data:audio/wav;base64,{INLINE}",
               f"DATA:audio/wav;base64,{INLINE}")
    REFUSED = ("file:///root/.ssh/id_rsa", "file:/etc/passwd", "ftp://x/a", "/etc/passwd",
               "../../workspace/x.wav", "\\\\host\\share\\a.wav", "http://[::1",
               # bare base64 is refused: "/" is a base64 character, so a padded path decodes
               INLINE, "/" * 78 + "etc/passwd")

    def places(self):
        speech = handlers()["/v1/audio/speech"].request_parser
        batch = handlers()["/v1/audio/speech/batch"].request_parser
        return {
            "edit url": lambda v: ImageEditPayload.from_json_msg({"url": v, "prompt": "p"}),
            "edit url[]": lambda v: ImageEditPayload.from_json_msg({"url[]": [v], "prompt": "p"}),
            "speech ref_audio": lambda v: speech({"input": "hi", "ref_audio": v}),
            "speech ref_audio_2": lambda v: speech({"input": "hi", "ref_audio_2": v}),
            "speech references": lambda v: speech({"input": "hi", "references": [{"audio_path": v}]}),
            "batch ref_audio": lambda v: batch({"items": [{"input": "a"}], "ref_audio": v}),
            "batch item ref_audio": lambda v: batch({"items": [{"input": "a", "ref_audio": v}]}),
            "video reference": lambda v: VideoPayload.from_json_msg(
                {"prompt": "p", "image_reference": [{"image_url": v}]}),
        }

    def test_every_reference_field(self):
        for place, send in self.places().items():
            for value in self.ALLOWED:
                with self.subTest((place, value)):
                    send(value)
            for value in self.REFUSED:
                with self.subTest((place, value)), self.assertRaises(JsonDataException):
                    send(value)

    def test_an_oversized_data_uri_reference_is_refused(self):
        big = "data:audio/wav;base64," + "A" * (4 * (core.MAX_UPLOAD_BYTES // 3 + 64))
        with self.assertRaisesRegex(JsonDataException, "larger than"):
            handlers()["/v1/audio/speech"].request_parser({"input": "hi", "ref_audio": big})

    def test_an_empty_reference_is_dropped(self):
        speech = handlers()["/v1/audio/speech"].request_parser
        batch = handlers()["/v1/audio/speech/batch"].request_parser
        self.assertNotIn("ref_audio", speech({"input": "hi", "ref_audio": ""}))
        self.assertNotIn("ref_audio_2", speech({"input": "hi", "ref_audio_2": ""}))
        self.assertNotIn("ref_audio", batch({"items": [{"input": "a", "ref_audio": ""}]})["items"][0])
        self.assertNotIn("ref_audio", batch({"items": [{"input": "a"}], "ref_audio": ""}))
        self.assertNotIn("url", ImageEditPayload.from_json_msg(
            {"image": b64(), "url": "", "prompt": "p"}).fields)

    def test_a_refused_speech_reference_names_its_field(self):
        speech = handlers()["/v1/audio/speech"].request_parser
        for data, field in [({"ref_audio_2": "/etc/passwd"}, "ref_audio_2"),
                            ({"references": [{"audio_path": "/etc/passwd"}]}, "references")]:
            with self.subTest(field), self.assertRaisesRegex(JsonDataException, field):
                speech({"input": "hi", **data})

    def test_a_speech_batch_takes_one_reference_string_per_item(self):
        batch = handlers()["/v1/audio/speech/batch"].request_parser
        for data in ({"items": [{"input": "a", "ref_audio": ["https://x/a.wav"]}]},
                     {"items": {"input": "a"}}, {"items": ["a"]}):
            with self.subTest(str(data)), self.assertRaises(JsonDataException):
                batch(data)


class TestHandlerTable(unittest.TestCase):
    def test_input_is_a_field_on_speech_embeddings_and_audio_generation(self):
        """The shared parser unwraps a top-level `input`; these routes unwrap only a dict."""
        body = {"model": "t", "input": "Hi", "voice": "a"}
        self.assertEqual(request_parser(dict(body)), "Hi")
        for route in ("/v1/audio/speech", "/v1/embeddings", "/v1/audio/generate"):
            with self.subTest(route):
                parser = handlers()[route].request_parser
                self.assertEqual(parser(dict(body)), body)
                self.assertEqual(parser({"input": dict(body)}), body)

    def test_benchmark_route_moves_the_one_benchmark(self):
        for env, route in [({}, "/v1/completions"), ({"BENCHMARK_ROUTE": ""}, "/v1/completions"),
                           *[({"BENCHMARK_ROUTE": r}, r) for r in ALL_ROUTES]]:
            with self.subTest(env), mock.patch.dict(os.environ, env):
                self.assertEqual([r for r, h in handlers().items() if h.benchmark_config], [route])

    def test_an_unusable_benchmark_route_is_refused(self):
        for env, says in [({"BENCHMARK_ROUTE": "/v1/audio/speach"}, "not a route"),
                          ({"BENCHMARK_ROUTE": "/v1/audio/speech",
                            "OPENAI_ROUTES": "/v1/completions"}, "not served")]:
            with self.subTest(env), mock.patch.dict(os.environ, env), \
                    self.assertRaisesRegex(RuntimeError, says):
                handlers()


class TestServedRoutes(unittest.TestCase):
    def test_the_default_is_what_llm_deployments_served_before(self):
        with mock.patch.dict(os.environ, {"OPENAI_ROUTES": ""}):
            self.assertEqual(set(handlers()), {"/v1/completions", "/v1/chat/completions"})
        with mock.patch.dict(os.environ, {"OPENAI_ROUTES": "",
                                          "BENCHMARK_ROUTE": "/v1/audio/transcriptions"}):
            self.assertEqual(set(handlers()), {"/v1/completions", "/v1/chat/completions",
                                               "/v1/audio/transcriptions"})
        self.assertEqual(set(handlers()), ALL_ROUTES)

    def test_the_startup_log_names_the_routes(self):
        lines = startup_log({"OPENAI_ROUTES": "/v1/audio/speech,/v1/embedding",
                             "BENCHMARK_ROUTE": "/v1/audio/speech"})
        self.assertIn("unknown routes: /v1/embedding", lines)
        self.assertIn("serving: /v1/audio/speech", lines)
        self.assertIn("benchmarking: /v1/audio/speech", lines)

    def test_an_old_sdk_is_only_reported_when_an_upload_route_is_asked_for(self):
        for routes, reported in (("", False), ("/v1/completions,/v1/images/edits", True)):
            with self.subTest(routes), mock.patch.object(core, "ApiPayload", OldApiPayload):
                self.assertEqual("cannot send multipart" in startup_log({"OPENAI_ROUTES": routes}),
                                 reported)

    def test_an_upload_benchmark_route_on_an_old_sdk_names_the_sdk(self):
        with mock.patch.object(core, "ApiPayload", OldApiPayload), \
                mock.patch.dict(os.environ, {"BENCHMARK_ROUTE": "/v1/audio/transcriptions"}), \
                self.assertRaisesRegex(RuntimeError, "multipart"):
            handlers()

    def test_every_multipart_route_is_dropped_on_an_old_sdk(self):
        multipart = {r for r, h in handlers().items()
                     if h.payload_class and issubclass(h.payload_class, core._UploadPayload)}
        self.assertEqual(multipart, set(core.UPLOAD_ROUTES))
        with mock.patch.object(core, "ApiPayload", OldApiPayload):
            self.assertEqual(set(handlers()), ALL_ROUTES - multipart)


class TestWorkload(unittest.TestCase):
    def calc(self, route):
        return handlers()[route].workload_calculator

    def test_completions_units_are_unchanged(self):
        self.assertEqual(self.calc("/v1/completions")({"max_tokens": 500}), 500)

    def test_images(self):
        calc = self.calc("/v1/images/generations")
        self.assertEqual(calc({"size": "1024x1024", "n": 2}), 2 * ONE_REQUEST)
        self.assertEqual(calc({"size": "512x512", "n": 4}), ONE_REQUEST)
        for size in (None, "auto", "nonsense"):
            with self.subTest(size):
                self.assertEqual(calc({"size": size}), ONE_REQUEST)
        self.assertEqual(calc({"size": "64x64"}), FLOOR)
        self.assertEqual(calc({"size": "8192x8192", "n": 10}), CEILING)

    def test_embedding_inputs_of_every_shape_land_on_one_scale(self):
        chars = benchmark.REF_EMBED_CHARS
        tokens = chars // core.CHARS_PER_TOKEN
        for value in ("x" * chars, ["x" * (chars // 2)] * 2, list(range(tokens)),
                      [list(range(tokens // 2))] * 2):
            with self.subTest(str(value)[:20]):
                self.assertEqual(self.calc("/v1/embeddings")({"input": value}), ONE_REQUEST)

    def test_rerank_and_score_count_each_query_document_pair(self):
        q, d = "q" * benchmark.RERANK_QUERY_CHARS, "d" * benchmark.RERANK_DOC_CHARS
        docs = [d] * benchmark.RERANK_DOCS
        for route, data in [("/v1/rerank", {"query": q, "documents": docs}),
                            ("/v1/score", {"queries": q, "documents": docs}),
                            ("/v1/score", {"queries": q, "items": docs}),
                            ("/v1/score", {"text_1": q, "text_2": docs}),
                            ("/v1/score", {"data_1": [q] * len(docs), "data_2": docs})]:
            with self.subTest((route, list(data))):
                self.assertEqual(self.calc(route)(data), ONE_REQUEST)
        pairwise = self.calc("/v1/score")({"queries": [q, q * 10], "items": [d, d]})
        self.assertEqual(pairwise, core._in_request_units(len(q) * 11 + 2 * len(d),
                                                          benchmark.REF_RERANK_CHARS))
        image = {"type": "image_url", "image_url": {"url": "u"}}
        one_query = self.calc("/v1/rerank")({"query": [{"type": "text", "text": q}, image],
                                             "documents": docs})
        self.assertEqual(one_query, core._in_request_units(
            len(docs) * 2 * benchmark.RERANK_DOC_CHARS, benchmark.REF_RERANK_CHARS))

    def test_each_clone_reference_adds_one_request(self):
        calc, base = self.calc("/v1/audio/speech"), {"input": "x" * 500}
        self.assertEqual(calc(base), ONE_REQUEST)
        self.assertEqual(calc({**base, "ref_audio": "A" * 100_000}), 2 * ONE_REQUEST)
        self.assertEqual(calc({**base, "ref_audio": ["u1", "u2"]}), 3 * ONE_REQUEST)
        self.assertEqual(calc({**base, "ref_audio": "u1", "ref_audio_2": "u2"}), 3 * ONE_REQUEST)
        self.assertEqual(calc({**base, "references": [{"audio_path": "u1"}]}), 2 * ONE_REQUEST)

    def test_a_speech_batch_counts_the_batch_reference_for_each_item(self):
        calc, item = self.calc("/v1/audio/speech/batch"), {"input": "x" * 500}
        self.assertEqual(calc({"items": [item] * 2}), 2 * ONE_REQUEST)
        self.assertEqual(calc({"items": [item] * 2, "ref_audio": "u"}), 4 * ONE_REQUEST)

    def test_a_chat_batch_counts_each_conversation(self):
        calc = self.calc("/v1/chat/completions/batch")
        self.assertEqual(calc({"messages": [CONVO] * 4, "max_tokens": 100}), 400)
        self.assertEqual(calc({"messages": [CONVO] * 4, "max_completion_tokens": 100,
                               "max_tokens": 16}), 400)   # vLLM reads max_completion_tokens first
        self.assertEqual(calc({"messages": [CONVO] * 2}), 2 * ONE_REQUEST)

    def test_audio_generation_is_one_request_whatever_the_length(self):
        calc = self.calc("/v1/audio/generate")
        self.assertEqual(calc({"audio_length": 1}), ONE_REQUEST)
        self.assertEqual(calc({"audio_length": 47}), ONE_REQUEST)

    def test_video_counts_pixels_frames_and_outputs(self):
        w = lambda **f: core._video_workload(f)   # noqa: E731
        self.assertEqual(w(), ONE_REQUEST)
        self.assertEqual(w(width=832, height=480, num_frames=66), 2 * ONE_REQUEST)
        self.assertEqual(w(size="832x480", num_frames=33, num_outputs_per_prompt=2), 2 * ONE_REQUEST)
        self.assertEqual(w(size="1664x480", width=64, height=64), 2 * ONE_REQUEST)   # size wins
        self.assertEqual(w(seconds="2", fps="16.5"), ONE_REQUEST)
        self.assertEqual(w(seconds="2"), core._in_request_units(48, 33))    # 24 fps

    def test_transcription_counts_seconds_of_audio(self):
        self.assertAlmostEqual(core._audio_seconds(_wav(7.5, rate=48000), "a.wav"), 7.5, places=2)
        raw = b"\x1aE\xdf\xa3" + b"\x00" * 80000
        self.assertAlmostEqual(core._audio_seconds(raw, "a.webm"),
                               len(raw) / core.AUDIO_BYTES_PER_SECOND["webm"], places=2)
        self.assertAlmostEqual(core._audio_seconds(raw, "a.mp3"),
                               len(raw) / core.DEFAULT_AUDIO_BYTES_PER_SECOND, places=2)

    def test_malformed_input_is_priced_without_raising(self):
        cases = {
            "/v1/images/generations": ({"size": "9" * 4000 + "x" + "9" * 4000},
                                       {"n": "9" * 5000}, {"n": -5, "size": "-10x-10"}),
            "/v1/audio/speech": ({"input": 123}, {"ref_audio": [None, 7, ""]},
                                 {"references": ["x", 5]}, {"references": 5}),
            "/v1/embeddings": ({}, {"input": None}),
            "/v1/rerank": ({}, {"query": 5, "documents": "x"}, {"documents": [None, {}, [[1]]]}),
            "/v1/score": ({}, {"text_1": [], "text_2": []}, {"queries": ["a"] * 3, "items": ["b"] * 2}),
            "/v1/chat/completions/batch": ({}, {"messages": "x", "max_tokens": 5},
                                           {"messages": [[]], "max_tokens": "a"}),
            "/v1/audio/speech/batch": ({"items": "x"}, {"items": [5, {"input": 3}]}),
        }
        for route, datas in cases.items():
            for data in datas:
                with self.subTest((route, str(data)[:30])):
                    self.assertGreaterEqual(self.calc(route)(data), FLOOR)
        for data in ({"width": "x", "height": 5}, {"num_frames": -1}, {"seconds": "a"},
                     {"size": "9" * 400 + "x1"}, {"num_outputs_per_prompt": "9" * 5000},
                     {"width": float("inf"), "height": 0}, {"num_frames": float("nan")}):
            with self.subTest(str(data)[:30]):
                self.assertGreaterEqual(core._video_workload(data), FLOOR)

    def test_no_single_request_trips_the_admission_gate(self):
        """The SDK divides all routes' in-flight work by one route's throughput, and 429s
        everything while that exceeds max_queue_time."""
        big = b64(b"\x00" * core.MAX_UPLOAD_BYTES)
        h = handlers()
        worst = {
            "/v1/audio/speech": {"input": "x" * 100_000, "ref_audio": ["A" * 10_000] * 10},
            "/v1/embeddings": {"input": ["x" * 10_000] * 1_000},
            "/v1/rerank": {"query": "x" * 10_000, "documents": ["x" * 10_000] * 1_000},
            "/v1/images/generations": {"n": 100_000, "size": "8192x8192"},
            "/v1/audio/speech/batch": {"items": [{"input": "x" * 100_000}] * 1_000},
            "/v1/chat/completions/batch": {"messages": [CONVO] * 256, "max_tokens": 2048},
        }
        for route, data in worst.items():
            with self.subTest(route):
                wt = wait_time_with_one_in_flight(h[route].workload_calculator(data))
                self.assertLess(wt, h[route].max_queue_time)
        for route, payload in [
            ("/v1/audio/transcriptions", TranscriptionPayload.from_json_msg({"file": big})),
            ("/v1/images/edits", ImageEditPayload.from_json_msg(
                {"image": [big] * 2, "prompt": "p", "n": 100_000, "size": "8192x8192"})),
            ("/v1/videos/sync", VideoPayload.from_json_msg(
                {"prompt": "p", "size": "8192x8192", "num_frames": 10**9,
                 "num_outputs_per_prompt": 10})),
        ]:
            with self.subTest(route):
                wt = wait_time_with_one_in_flight(payload.count_workload())
                self.assertLess(wt, h[route].max_queue_time)


class TestBenchmarks(unittest.TestCase):
    def test_every_benchmark_request_weighs_one_reference_request(self):
        """Otherwise the score means something different depending on the route."""
        weigh = {"/v1/completions": lambda b: b["max_tokens"],
                 "/v1/chat/completions": lambda b: b["max_tokens"],
                 "/v1/chat/completions/batch": core._chat_batch_workload,
                 "/v1/embeddings": core._embeddings_workload,
                 "/v1/rerank": core._rerank_workload,
                 "/v1/score": core._score_workload,
                 "/v1/audio/speech": core._speech_workload,
                 "/v1/audio/speech/batch": core._speech_batch_workload,
                 "/v1/audio/generate": core._audio_generate_workload,
                 "/v1/images/generations": core._image_workload,
                 "/v1/images/edits": lambda _: ImageEditPayload.for_test().count_workload(),
                 "/v1/audio/transcriptions": lambda _: TranscriptionPayload.for_test().count_workload(),
                 "/v1/audio/translations": lambda _: TranscriptionPayload.for_test().count_workload(),
                 "/v1/videos/sync": lambda _: VideoPayload.for_test().count_workload()}
        for route, b in BENCHMARKS.items():
            with self.subTest(route):
                weight = weigh[route](b.generator() if b.generator else None)
                self.assertGreaterEqual(weight, 0.5 * ONE_REQUEST)
                self.assertLessEqual(weight, 2 * ONE_REQUEST)

    def test_the_embeddings_benchmark_fits_a_256_token_encoder(self):
        """At 2000 chars it tokenised past bge-small's 512 and the worker never became ready."""
        worst_case_tokens = len(benchmark.embeddings_benchmark_generator()["input"]) / 3.1
        self.assertLess(worst_case_tokens, 256 * 0.8)

    def test_a_bad_embed_chars_setting_falls_back_to_the_default(self):
        """Imported by every OpenAI worker: a bad value must not stop the boot."""
        for value in ("", "abc", "0", "-5"):
            with self.subTest(value):
                env = dict(os.environ, BENCHMARK_EMBED_CHARS=value, PYTHONPATH=os.pathsep.join(sys.path))
                out = subprocess.run(
                    [sys.executable, "-c",
                     "from workers.openai.benchmark import REF_EMBED_CHARS; print(REF_EMBED_CHARS)"],
                    env=env, capture_output=True, text=True)
                self.assertEqual(out.returncode, 0, out.stderr)
                self.assertEqual(out.stdout.split()[-1], "600")
                self.assertEqual("WARNING" in out.stdout, bool(value))

    def test_the_transcription_benchmark_sends_the_speech_sample_tiled(self):
        """Noise leaves the decoder idle and overstates throughput."""
        name, audio, ctype = TranscriptionPayload.for_test().generate_payload_multipart()["file"]
        self.assertEqual((name, ctype), ("benchmark.wav", "audio/wav"))
        with wave.open(str(benchmark._SPEECH_PATH)) as w:
            sample = w.readframes(w.getnframes())
        with wave.open(io.BytesIO(audio)) as w:
            self.assertEqual((w.getnchannels(), w.getsampwidth(), w.getframerate()), (1, 2, 16000))
            frames = w.readframes(w.getnframes())
        self.assertEqual(frames[:2 * len(sample)], sample * 2)

    def test_benchmark_inputs_differ_per_request(self):
        """Engines cache multimodal input by content hash."""
        clips = {hashlib.sha256(benchmark.benchmark_speech()).hexdigest() for _ in range(4)}
        self.assertEqual(len(clips), 4)
        self.assertNotEqual(benchmark.synthetic_png(), benchmark.synthetic_png())

    def test_the_edit_benchmark_image_is_a_valid_reference_sized_png(self):
        png = benchmark.synthetic_png()
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        pos, chunks = 8, {}
        while pos < len(png):
            n = struct.unpack(">I", png[pos:pos + 4])[0]
            kind, data = png[pos + 4:pos + 8], png[pos + 8:pos + 8 + n]
            self.assertEqual(struct.unpack(">I", png[pos + 8 + n:pos + 12 + n])[0],
                             zlib.crc32(kind + data) & 0xFFFFFFFF, kind)
            chunks[kind] = chunks.get(kind, b"") + data
            pos += 12 + n
        w, h, depth, colour = struct.unpack(">IIBB", chunks[b"IHDR"][:10])
        side = benchmark.REF_IMAGE_SIDE
        self.assertEqual((w, h, depth, colour), (side, side, 8, 2))
        raw = zlib.decompress(chunks[b"IDAT"])
        self.assertEqual(len(raw), h * (1 + w * 3))
        pixels = bytes(b for i, b in enumerate(raw) if i % (1 + w * 3))
        self.assertGreater(len(set(pixels)), 200, "the image is effectively uniform")


if __name__ == "__main__":
    unittest.main()
