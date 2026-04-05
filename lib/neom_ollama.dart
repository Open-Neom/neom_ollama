/// Ollama integration for Open Neom.
///
/// Provides model discovery, pull/delete, health checks, setup automation,
/// and OpenAI-compatible chat service for local inference.
///
/// ```dart
/// import 'package:neom_ollama/neom_ollama.dart';
///
/// final client = OllamaClient();
/// final models = await client.listModels();
/// final response = await client.chat('qwen2.5:3b', 'Hello');
/// ```
library neom_ollama;

export 'src/ollama_client.dart';
export 'src/ollama_setup.dart';
export 'src/ollama_chat_service.dart';
