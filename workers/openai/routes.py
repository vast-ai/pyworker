"""The routes the OpenAI-compatible workers serve, named once for the worker, its
benchmark and the client. No imports, so the client can use it without the worker's
dependencies."""

COMPLETIONS = "/v1/completions"
CHAT = "/v1/chat/completions"
CHAT_BATCH = "/v1/chat/completions/batch"
EMBEDDINGS = "/v1/embeddings"
RERANK = "/v1/rerank"
SCORE = "/v1/score"
SPEECH = "/v1/audio/speech"
SPEECH_BATCH = "/v1/audio/speech/batch"
AUDIO_GENERATE = "/v1/audio/generate"
TRANSCRIPTIONS = "/v1/audio/transcriptions"
TRANSLATIONS = "/v1/audio/translations"
IMAGES = "/v1/images/generations"
IMAGE_EDITS = "/v1/images/edits"
VIDEO = "/v1/videos/sync"

# Multipart: these carry file uploads.
UPLOAD_ROUTES = (IMAGE_EDITS, VIDEO, TRANSCRIPTIONS, TRANSLATIONS)
