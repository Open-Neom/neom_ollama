# neom_ollama

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Ollama integration for Dart and Flutter. Model discovery, pull/delete, health checks, setup automation, and OpenAI-compatible chat service for local inference.

Part of the [Open Neom](https://github.com/Open-Neom) ecosystem.

## Features

- **OllamaClient** — Full Ollama REST API: status, list/show/pull/create/delete models, chat, streaming.
- **OllamaSetup** — Automated provisioning: detect install, start server, pull models with progress.
- **OllamaChatService** — OpenAI-compatible `/v1/chat/completions` client with conversation history.
- **Pure Dart** — No Flutter dependency. Works in CLI, desktop, mobile, and server.

## Usage

```dart
import 'package:neom_ollama/neom_ollama.dart';

// Check if Ollama is running
final client = OllamaClient();
final status = await client.checkStatus(); // OllamaStatus.running

// List local models
final models = await client.listModels();
for (final m in models) {
  print('${m.displayName} — ${m.sizeLabel}');
}

// Single-turn chat
final response = await client.chat('qwen2.5:3b', 'What is Dart?');
print(response);

// Streaming chat
await for (final chunk in client.chatStream('qwen2.5:3b', 'Explain async/await')) {
  stdout.write(chunk);
}

// Pull a model with progress
await for (final p in client.pullModel('llama3.1:8b')) {
  print('${p.status} ${((p.progress ?? 0) * 100).toStringAsFixed(0)}%');
}

// Create a custom model from Modelfile
await client.createModel('my-model', 'FROM qwen2.5:3b\nSYSTEM "You are helpful."');

// OpenAI-compatible chat service
final chat = OllamaChatService(model: 'qwen2.5:3b');
chat.setSystemInstruction('You are a coding assistant.');
final reply = await chat.sendMessage('Write a hello world in Dart');
print(reply);
```

### Automated Setup

```dart
final setup = OllamaSetup();

// Check status
final status = await setup.checkStatus(requiredModel: 'qwen2.5:3b');

// Full automated setup with progress
await setup.runFullSetup(
  modelName: 'qwen2.5:3b',
  onProgress: (step, message, progress) {
    print('[$step] $message (${(progress * 100).toStringAsFixed(0)}%)');
  },
);
```

## Requirements

- [Ollama](https://ollama.com) installed and running on `localhost:11434`
- Dart SDK ≥ 3.8.0

## License

Apache 2.0 — see [LICENSE](LICENSE).
