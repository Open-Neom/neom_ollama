import 'dart:async';
import 'dart:convert';

import 'package:http/http.dart' as http;
import 'package:logger/logger.dart';

import 'thinking_parser.dart';

// ═══════════════════════════════════════════
// Status & Models
// ═══════════════════════════════════════════

enum OllamaStatus { unknown, checking, running, notRunning, error }

class OllamaModel {
  final String name;
  final String? digest;
  final int? sizeBytes;
  final DateTime? modifiedAt;
  final String? family;
  final String? parameterSize;
  final String? quantizationLevel;

  const OllamaModel({
    required this.name,
    this.digest,
    this.sizeBytes,
    this.modifiedAt,
    this.family,
    this.parameterSize,
    this.quantizationLevel,
  });

  factory OllamaModel.fromJson(Map<String, dynamic> json) {
    final details = json['details'] as Map<String, dynamic>? ?? {};
    return OllamaModel(
      name: json['name'] as String? ?? json['model'] as String? ?? '',
      digest: json['digest'] as String?,
      sizeBytes: json['size'] as int?,
      modifiedAt: json['modified_at'] != null
          ? DateTime.tryParse(json['modified_at'] as String)
          : null,
      family: details['family'] as String?,
      parameterSize: details['parameter_size'] as String?,
      quantizationLevel: details['quantization_level'] as String?,
    );
  }

  String get sizeLabel {
    if (sizeBytes == null) return '';
    final gb = sizeBytes! / (1024 * 1024 * 1024);
    if (gb >= 1) return '${gb.toStringAsFixed(1)} GB';
    final mb = sizeBytes! / (1024 * 1024);
    return '${mb.toStringAsFixed(0)} MB';
  }

  String get displayName {
    if (name.endsWith(':latest')) return name.replaceAll(':latest', '');
    return name;
  }
}

class OllamaPullProgress {
  final String status;
  final int? total;
  final int? completed;
  final String? digest;

  const OllamaPullProgress({
    required this.status,
    this.total,
    this.completed,
    this.digest,
  });

  double? get progress {
    if (total == null || total == 0 || completed == null) return null;
    return completed! / total!;
  }

  bool get isDone => status == 'success';
  bool get isError => status.startsWith('error');
}

// ═══════════════════════════════════════════
// Client — core HTTP operations
// ═══════════════════════════════════════════

/// Low-level Ollama HTTP client. Stateless, pure Dart, no Flutter dependency.
///
/// Supports the full Ollama REST API:
/// - Status check, model listing, model info
/// - Pull (download) with streaming progress
/// - Create model from Modelfile
/// - Delete model
/// - Chat (native + OpenAI-compatible)
class OllamaClient {
  final String host;
  final Duration timeout;

  static final Logger _logger = Logger(
    printer: PrettyPrinter(
      methodCount: 0,
      errorMethodCount: 5,
      lineLength: 80,
      colors: true,
      printEmojis: true,
      dateTimeFormat: DateTimeFormat.none,
    ),
  );

  OllamaClient({
    this.host = 'http://localhost:11434',
    this.timeout = const Duration(seconds: 5),
  });

  /// Check if Ollama server is reachable.
  Future<OllamaStatus> checkStatus() async {
    try {
      final resp = await http.get(Uri.parse(host)).timeout(timeout);
      if (resp.statusCode == 200) {
        _logger.d('[OllamaClient] checkStatus: Server is running at $host');
        return OllamaStatus.running;
      }
      _logger.w('[OllamaClient] checkStatus: Server returned code ${resp.statusCode} at $host');
      return OllamaStatus.notRunning;
    } catch (e) {
      _logger.w('[OllamaClient] checkStatus: Server unreachable at $host. Error: $e');
      return OllamaStatus.notRunning;
    }
  }

  /// Whether Ollama is running (convenience).
  Future<bool> get isRunning async => await checkStatus() == OllamaStatus.running;

  /// List all locally available models.
  Future<List<OllamaModel>> listModels() async {
    try {
      _logger.d('[OllamaClient] Listing local models from host: $host');
      final resp = await http.get(Uri.parse('$host/api/tags')).timeout(timeout);
      if (resp.statusCode != 200) {
        _logger.w('[OllamaClient] listModels: Server returned code ${resp.statusCode}');
        return [];
      }

      final body = jsonDecode(resp.body);
      final models = body['models'] as List? ?? [];
      _logger.d('[OllamaClient] listModels: Found ${models.length} models locally');
      return models
          .map((m) => OllamaModel.fromJson(m as Map<String, dynamic>))
          .toList()
        ..sort((a, b) => a.name.compareTo(b.name));
    } catch (e) {
      _logger.e('[OllamaClient] listModels: Failed to query local models. Error: $e');
      return [];
    }
  }

  /// Check if a specific model is available locally.
  Future<bool> hasModel(String name) async {
    final list = await listModels();
    return list.any((m) => m.name == name || m.displayName == name);
  }

  /// Get detailed info about a model.
  Future<OllamaModel?> showModel(String name) async {
    try {
      _logger.d('[OllamaClient] showModel: Querying metadata for $name');
      final resp = await http.post(
        Uri.parse('$host/api/show'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'name': name}),
      ).timeout(timeout);
      if (resp.statusCode != 200) {
        _logger.w('[OllamaClient] showModel ($name): Server returned code ${resp.statusCode}');
        return null;
      }

      final body = jsonDecode(resp.body) as Map<String, dynamic>;
      return OllamaModel.fromJson({'name': name, ...body});
    } catch (e) {
      _logger.e('[OllamaClient] showModel ($name): Failed to fetch info. Error: $e');
      return null;
    }
  }

  /// Pull (download) a model. Streams progress updates.
  Stream<OllamaPullProgress> pullModel(String name) async* {
    try {
      _logger.i('[OllamaClient] ⏬ Initiating model download: $name');
      final request = http.Request('POST', Uri.parse('$host/api/pull'));
      request.headers['Content-Type'] = 'application/json';
      request.body = jsonEncode({'name': name, 'stream': true});

      final client = http.Client();
      final response = await client.send(request);

      await for (final chunk in response.stream.transform(utf8.decoder)) {
        for (final line in chunk.split('\n')) {
          if (line.trim().isEmpty) continue;
          try {
            final json = jsonDecode(line) as Map<String, dynamic>;
            final progress = OllamaPullProgress(
              status: json['status'] as String? ?? '',
              total: json['total'] as int?,
              completed: json['completed'] as int?,
              digest: json['digest'] as String?,
            );
            _logger.t('[OllamaClient] pullModel ($name) progress: status="${progress.status}" total=${progress.total} completed=${progress.completed}');
            yield progress;
          } catch (_) {}
        }
      }
      client.close();
      _logger.i('[OllamaClient] ✅ Completed pull request stream for model: $name');
    } catch (e) {
      _logger.e('[OllamaClient] ❌ Error pulling model $name. Detail: $e');
      yield OllamaPullProgress(status: 'error: $e');
    }
  }

  /// Create a model from a Modelfile string.
  Future<bool> createModel(String name, String modelfile) async {
    try {
      _logger.i('[OllamaClient] 🛠️ Creating custom model "$name" from Modelfile...');
      final resp = await http.post(
        Uri.parse('$host/api/create'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'name': name, 'modelfile': modelfile, 'stream': false}),
      ).timeout(const Duration(seconds: 120));
      
      final success = resp.statusCode == 200;
      if (success) {
        _logger.i('[OllamaClient] ✅ Custom model "$name" created successfully');
      } else {
        _logger.e('[OllamaClient] ❌ Failed to create model "$name". Status: ${resp.statusCode}. Response: ${resp.body}');
      }
      return success;
    } catch (e) {
      _logger.e('[OllamaClient] ❌ Exception during model creation for "$name". Error: $e');
      return false;
    }
  }

  /// Delete a local model.
  Future<bool> deleteModel(String name) async {
    try {
      _logger.w('[OllamaClient] ❌ Deleting local model: $name');
      final resp = await http.delete(
        Uri.parse('$host/api/delete'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'name': name}),
      );
      final success = resp.statusCode == 200;
      if (success) {
        _logger.w('[OllamaClient] ✅ Model "$name" deleted successfully');
      } else {
        _logger.e('[OllamaClient] ❌ Failed to delete model "$name". Status: ${resp.statusCode}');
      }
      return success;
    } catch (e) {
      _logger.e('[OllamaClient] ❌ Exception during model deletion for "$name". Error: $e');
      return false;
    }
  }

  /// Single-turn chat (native Ollama API). Returns response text with any
  /// `<think>…</think>` reasoning block stripped (user-visible content only).
  ///
  /// For full access to the reasoning trace, use [chatWithThinking].
  Future<String> chat(String model, String prompt, {String? system}) async {
    final result = await chatWithThinking(model, prompt, system: system);
    return result.content;
  }

  /// Single-turn chat that also exposes the reasoning trace.
  ///
  /// Ollama's native API (as of the `fix: enable thinking support for the
  /// ollama api` patch on main) can return a `message.thinking` field when
  /// the backing model supports it (Qwen3, DeepSeek-R1, QwQ, gpt-oss…).
  /// When the server omits that field but the model still emits inline
  /// `<think>…</think>` tags, we fall back to parsing the content via
  /// [ThinkingParser].
  ///
  /// Pass [enableThinking] = false to request a non-reasoning response
  /// (sent as `"think": false` in the Ollama request body).
  Future<OllamaChatResult> chatWithThinking(
    String model,
    String prompt, {
    String? system,
    bool enableThinking = true,
  }) async {
    final messages = <Map<String, String>>[];
    if (system != null) messages.add({'role': 'system', 'content': system});
    messages.add({'role': 'user', 'content': prompt});

    _logger.d('[OllamaClient] 🚀 Sending chat request for model: $model');
    final sw = Stopwatch()..start();

    try {
      final resp = await http.post(
        Uri.parse('$host/api/chat'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'model': model,
          'messages': messages,
          'stream': false,
          if (!enableThinking) 'think': false,
        }),
      ).timeout(const Duration(seconds: 60));

      sw.stop();

      if (resp.statusCode != 200) {
        _logger.e('[OllamaClient] ❌ Chat request failed. Status: ${resp.statusCode}. Response: ${resp.body}');
        throw Exception('Ollama error: ${resp.statusCode}');
      }

      final body = jsonDecode(resp.body) as Map<String, dynamic>;
      final message = body['message'] as Map<String, dynamic>? ?? const {};
      final rawContent = message['content'] as String? ?? '';
      final serverThinking = message['thinking'] as String? ?? '';

      _logger.d('[OllamaClient] Chat request completed in ${sw.elapsedMilliseconds}ms. Content length: ${rawContent.length} chars.');

      // Prefer the structured thinking field (new Ollama API). Fall back to
      // inline-tag parsing for older servers or models that still emit tags.
      if (serverThinking.isNotEmpty) {
        return OllamaChatResult(
          thinking: serverThinking,
          content: rawContent,
        );
      }
      final split = ThinkingParser.split(rawContent);
      return OllamaChatResult(
        thinking: split.thinking,
        content: split.content,
      );
    } catch (e) {
      _logger.e('[OllamaClient] ❌ Chat request failed after ${sw.elapsedMilliseconds}ms. Error: $e');
      rethrow;
    }
  }

  /// Streaming chat (native Ollama API). Yields text chunks.
  Stream<String> chatStream(String model, String prompt, {String? system}) async* {
    final messages = <Map<String, String>>[];
    if (system != null) messages.add({'role': 'system', 'content': system});
    messages.add({'role': 'user', 'content': prompt});

    _logger.d('[OllamaClient] 🚀 Initiating chat response stream for model: $model');

    try {
      final request = http.Request('POST', Uri.parse('$host/api/chat'));
      request.headers['Content-Type'] = 'application/json';
      request.body = jsonEncode({'model': model, 'messages': messages, 'stream': true});

      final response = await http.Client().send(request);

      await for (final chunk in response.stream.transform(utf8.decoder)) {
        for (final line in chunk.split('\n')) {
          if (line.trim().isEmpty) continue;
          try {
            final json = jsonDecode(line) as Map<String, dynamic>;
            final content = json['message']?['content'] as String?;
            if (content != null && content.isNotEmpty) {
              yield content;
            }
          } catch (_) {}
        }
      }
      _logger.d('[OllamaClient] ✅ Chat response stream ended for model: $model');
    } catch (e) {
      _logger.e('[OllamaClient] ❌ Error during chat response stream. Detail: $e');
      rethrow;
    }
  }

  /// Load a model into RAM (keep-alive indefinitely).
  Future<bool> loadModel(String name) async {
    _logger.i('[OllamaClient] 🚀 Loading model into memory: $name');
    final sw = Stopwatch()..start();
    try {
      final resp = await http.post(
        Uri.parse('$host/api/generate'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'model': name,
          'prompt': '',
          'keep_alive': -1,
        }),
      ).timeout(const Duration(seconds: 120));
      
      sw.stop();
      final success = resp.statusCode == 200;
      if (success) {
        _logger.i('[OllamaClient] ✅ Model "$name" successfully loaded in memory in ${sw.elapsedMilliseconds}ms');
      } else {
        _logger.e('[OllamaClient] ❌ Failed to load model "$name" in memory. Status: ${resp.statusCode}');
      }
      return success;
    } catch (e) {
      sw.stop();
      _logger.e('[OllamaClient] ❌ Exception loading model "$name" in memory after ${sw.elapsedMilliseconds}ms. Error: $e');
      return false;
    }
  }

  /// Unload a model from RAM.
  Future<bool> unloadModel(String name) async {
    _logger.w('[OllamaClient] 🗑️ Unloading model from memory: $name');
    try {
      final resp = await http.post(
        Uri.parse('$host/api/generate'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'model': name,
          'keep_alive': 0,
        }),
      ).timeout(const Duration(seconds: 10));
      
      final success = resp.statusCode == 200;
      if (success) {
        _logger.w('[OllamaClient] ✅ Model "$name" successfully unloaded from memory');
      } else {
        _logger.e('[OllamaClient] ❌ Failed to unload model "$name" from memory. Status: ${resp.statusCode}');
      }
      return success;
    } catch (e) {
      _logger.e('[OllamaClient] ❌ Exception unloading model "$name" from memory. Error: $e');
      return false;
    }
  }

  /// List models currently loaded in memory/RAM.
  Future<List<String>> listRunningModels() async {
    try {
      _logger.t('[OllamaClient] Querying currently running models in memory via /api/ps...');
      final resp = await http.get(Uri.parse('$host/api/ps')).timeout(timeout);
      if (resp.statusCode != 200) {
        _logger.w('[OllamaClient] listRunningModels: Server returned code ${resp.statusCode}');
        return [];
      }
      final body = jsonDecode(resp.body);
      final models = body['models'] as List? ?? [];
      final names = models.map((m) => (m['name'] as String? ?? '')).toList();
      _logger.t('[OllamaClient] Currently running models in memory: $names');
      return names;
    } catch (e) {
      _logger.e('[OllamaClient] Failed to fetch running models. Error: $e');
      return [];
    }
  }

  /// OpenAI-compatible base URL for providers that use /v1/chat/completions.
  String get openAiBaseUrl => '$host/v1';
}

/// Result of [OllamaClient.chatWithThinking].
///
/// * [thinking] — the reasoning trace (empty when the model didn't think).
/// * [content] — the user-visible answer, always free of thinking markers.
class OllamaChatResult {
  final String thinking;
  final String content;

  const OllamaChatResult({
    required this.thinking,
    required this.content,
  });

  bool get hasThinking => thinking.isNotEmpty;

  @override
  String toString() =>
      'OllamaChatResult(thinking: ${thinking.length} chars, content: ${content.length} chars)';
}
