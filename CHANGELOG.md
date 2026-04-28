## 1.1.0

- **Hardware profiler** — cross-platform RAM/CPU/GPU detection (`HardwareProfiler.detect()`), with native (FFI), web (navigator + WebGL), and stub backends.
- **Model advisor** — recommends Ollama models based on detected hardware tier.
- **Model optimizer** — picks optimal `num_gpu` / `num_thread` parameters per host.
- **Thinking trace support** — `OllamaClient.chatWithThinking()` exposes the reasoning trace for Qwen3, DeepSeek-R1, QwQ, gpt-oss, and other reasoning models. Honours both the new Ollama `message.thinking` field and inline `<think>…</think>` / `Thought:` patterns.
- **`ThinkingParser`** — pure-Dart utility (zero deps) to split reasoning from content; handles closed tags, dangling open tags (streaming), and bare `Thought: / Answer:` formatting.
- **`OllamaChatResult`** — new return type with `thinking` + `content` fields.
- **Web typing fixes** — added explicit `<Object?>` type args to `js_util.getProperty` and `callMethod` so the package compiles cleanly under Dart 3.10+ strict typing.
- Tests added: `ollama_client_test.dart`, `thinking_parser_test.dart`.

## 1.0.0

- Initial release.
- `OllamaClient` — HTTP client for Ollama REST API (status, list, show, pull, create, delete, chat, stream).
- `OllamaSetup` — Automated setup: install detection, server start, model provisioning with progress streaming.
- `OllamaChatService` — OpenAI-compatible `/v1/chat/completions` client with conversation history.
- `OllamaModel` — Data class with name, size, family, parameters, quantization.
- `OllamaPullProgress` — Streaming progress for model downloads.
- Pure Dart — no Flutter dependency, works in CLI, desktop, and mobile.
