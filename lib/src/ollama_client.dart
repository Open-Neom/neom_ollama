import 'dart:async';
import 'dart:convert';

import 'package:http/http.dart' as http;

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

  OllamaClient({
    this.host = 'http://localhost:11434',
    this.timeout = const Duration(seconds: 5),
  });

  /// Check if Ollama server is reachable.
  Future<OllamaStatus> checkStatus() async {
    try {
      final resp = await http.get(Uri.parse(host)).timeout(timeout);
      if (resp.statusCode == 200) return OllamaStatus.running;
      return OllamaStatus.notRunning;
    } catch (_) {
      return OllamaStatus.notRunning;
    }
  }

  /// Whether Ollama is running (convenience).
  Future<bool> get isRunning async => await checkStatus() == OllamaStatus.running;

  /// List all locally available models.
  Future<List<OllamaModel>> listModels() async {
    try {
      final resp = await http.get(Uri.parse('$host/api/tags')).timeout(timeout);
      if (resp.statusCode != 200) return [];

      final body = jsonDecode(resp.body);
      final models = body['models'] as List? ?? [];
      return models
          .map((m) => OllamaModel.fromJson(m as Map<String, dynamic>))
          .toList()
        ..sort((a, b) => a.name.compareTo(b.name));
    } catch (_) {
      return [];
    }
  }

  /// Check if a specific model is installed.
  Future<bool> hasModel(String name) async {
    final models = await listModels();
    return models.any((m) => m.name.startsWith(name));
  }

  /// Get detailed info about a model.
  Future<OllamaModel?> showModel(String name) async {
    try {
      final resp = await http.post(
        Uri.parse('$host/api/show'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'name': name}),
      ).timeout(timeout);
      if (resp.statusCode != 200) return null;

      final body = jsonDecode(resp.body) as Map<String, dynamic>;
      return OllamaModel.fromJson({'name': name, ...body});
    } catch (_) {
      return null;
    }
  }

  /// Pull (download) a model. Streams progress updates.
  Stream<OllamaPullProgress> pullModel(String name) async* {
    try {
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
            yield OllamaPullProgress(
              status: json['status'] as String? ?? '',
              total: json['total'] as int?,
              completed: json['completed'] as int?,
              digest: json['digest'] as String?,
            );
          } catch (_) {}
        }
      }
      client.close();
    } catch (e) {
      yield OllamaPullProgress(status: 'error: $e');
    }
  }

  /// Create a model from a Modelfile string.
  Future<bool> createModel(String name, String modelfile) async {
    try {
      final resp = await http.post(
        Uri.parse('$host/api/create'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'name': name, 'modelfile': modelfile, 'stream': false}),
      ).timeout(const Duration(seconds: 120));
      return resp.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  /// Delete a local model.
  Future<bool> deleteModel(String name) async {
    try {
      final resp = await http.delete(
        Uri.parse('$host/api/delete'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'name': name}),
      );
      return resp.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  /// Single-turn chat (native Ollama API). Returns response text.
  Future<String> chat(String model, String prompt, {String? system}) async {
    final messages = <Map<String, String>>[];
    if (system != null) messages.add({'role': 'system', 'content': system});
    messages.add({'role': 'user', 'content': prompt});

    final resp = await http.post(
      Uri.parse('$host/api/chat'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'model': model, 'messages': messages, 'stream': false}),
    ).timeout(const Duration(seconds: 60));

    if (resp.statusCode != 200) throw Exception('Ollama error: ${resp.statusCode}');
    final body = jsonDecode(resp.body);
    return body['message']?['content'] as String? ?? '';
  }

  /// Streaming chat (native Ollama API). Yields text chunks.
  Stream<String> chatStream(String model, String prompt, {String? system}) async* {
    final messages = <Map<String, String>>[];
    if (system != null) messages.add({'role': 'system', 'content': system});
    messages.add({'role': 'user', 'content': prompt});

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
          if (content != null && content.isNotEmpty) yield content;
        } catch (_) {}
      }
    }
  }

  /// OpenAI-compatible base URL for providers that use /v1/chat/completions.
  String get openAiBaseUrl => '$host/v1';
}
