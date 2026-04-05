## 1.0.0

- Initial release.
- `OllamaClient` — HTTP client for Ollama REST API (status, list, show, pull, create, delete, chat, stream).
- `OllamaSetup` — Automated setup: install detection, server start, model provisioning with progress streaming.
- `OllamaChatService` — OpenAI-compatible `/v1/chat/completions` client with conversation history.
- `OllamaModel` — Data class with name, size, family, parameters, quantization.
- `OllamaPullProgress` — Streaming progress for model downloads.
- Pure Dart — no Flutter dependency, works in CLI, desktop, and mobile.
