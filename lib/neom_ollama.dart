/// Ollama integration for Open Neom.
///
/// Provides model discovery, pull/delete, health checks, setup automation,
/// hardware-aware model recommendation, runtime optimization,
/// and OpenAI-compatible chat service for local inference.
///
/// ```dart
/// import 'package:neom_ollama/neom_ollama.dart';
///
/// final client = OllamaClient();
/// final models = await client.listModels();
/// final response = await client.chat('qwen2.5:3b', 'Hello');
///
/// // Hardware-aware optimization
/// final optimizer = ModelOptimizer();
/// final hw = await optimizer.detectHardware();
/// final rec = ModelAdvisor().recommend(hw, category: 'coding');
/// await optimizer.preloadModel(rec.recommended.ollamaTag);
/// ```
library neom_ollama;

export 'src/ollama_client.dart';
export 'src/ollama_setup.dart';
export 'src/ollama_chat_service.dart';
export 'src/thinking_parser.dart';

// Hardware & Optimization
export 'src/hardware_profiler.dart';
export 'src/model_advisor.dart';
export 'src/model_optimizer.dart';
