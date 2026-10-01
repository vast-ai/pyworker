# benchmark_speech.wav

The speech clip the transcription and translation benchmarks send, tiled to 30 s.

- Text, written for this purpose: "A quick update on the project. We finished the first
  round of testing on Monday, and the results look good, so we plan to ship next week."
- Synthesised with openbmb/VoxCPM2 (Apache-2.0) through vLLM-Omni's `/v1/audio/speech`,
  `voice="default"`, `response_format="wav"`, `speed=0.7`, then converted with
  `ffmpeg -ar 16000 -ac 1 -sample_fmt s16`.
- 10.74 s, 16 kHz mono 16-bit PCM, 343,850 bytes.

Checked on faster-whisper large-v2: transcribed word for word, and it costs about 1.3x
a real recording of the same length (it is denser speech), where synthetic noise costs
a small fraction. vLLM whisper-large-v3 also transcribes it word for word.
