import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

import 'ollama_client.dart';
import 'thinking_parser.dart';

/// OpenAI-compatible chat service for Ollama.
///
/// Speaks the /v1/chat/completions endpoint — same format as OpenAI, DeepSeek,
/// Groq, etc. This allows any OpenAI-compatible consumer to use Ollama.
class OllamaChatService {
  static const int defaultMaxHistoryMessages = 40;
  static const int maxStreamLineChars = 1024 * 1024;
  static const int maxErrorSnippetChars = 512;
  static const Set<String> _prototypePollutionKeys = {
    '__proto__',
    'constructor',
    'prototype',
  };

  static String _truncateUtf16Safe(String input, int maxChars) {
    if (maxChars <= 0 || input.isEmpty) return '';
    if (input.length <= maxChars) return input;
    var end = maxChars;
    final prev = input.codeUnitAt(end - 1);
    if (prev >= 0xD800 && prev <= 0xDBFF) {
      end -= 1;
    }
    return end <= 0 ? '' : input.substring(0, end);
  }

  static Map<String, dynamic> _sanitizeHistoryEntry(Map<String, dynamic> raw) {
    final out = <String, dynamic>{};
    for (final entry in raw.entries) {
      if (_prototypePollutionKeys.contains(entry.key)) continue;
      out[entry.key] = entry.value;
    }
    return out;
  }

  final String baseUrl;
  final String model;
  final double temperature;
  final int maxTokens;
  final int maxHistoryMessages;
  final http.Client? _injectedClient;

  final List<Map<String, dynamic>> _conversationMessages = [];

  OllamaChatService({
    String baseUrl = 'http://localhost:11434/v1',
    required this.model,
    double temperature = 0.7,
    int maxTokens = 1024,
    int maxHistoryMessages = defaultMaxHistoryMessages,
    http.Client? httpClient,
  }) : baseUrl = OllamaClient.normalizeEndpointUrl(
         baseUrl,
         defaultUrl: 'http://localhost:11434/v1',
       ),
       temperature = temperature.isFinite ? temperature.clamp(0.0, 2.0) : 0.7,
       maxTokens = maxTokens.isFinite ? maxTokens.clamp(1, 32768) : 1024,
       maxHistoryMessages = maxHistoryMessages.isFinite
           ? maxHistoryMessages.clamp(2, 500)
           : defaultMaxHistoryMessages,
       _injectedClient = httpClient;

  /// Add a system instruction (call before first message).
  void setSystemInstruction(String instruction) {
    _conversationMessages.removeWhere((m) => m['role'] == 'system');
    if (instruction.isNotEmpty) {
      _conversationMessages.insert(0, {
        'role': 'system',
        'content': instruction,
      });
    }
  }

  void _trimHistory() {
    if (_conversationMessages.length <= maxHistoryMessages) return;
    final hasSystem =
        _conversationMessages.isNotEmpty &&
        _conversationMessages.first['role'] == 'system';
    final keepTail = (maxHistoryMessages - (hasSystem ? 1 : 0)).clamp(2, 1000);
    final tailStart = _conversationMessages.length - keepTail;
    final preserved = <Map<String, dynamic>>[
      if (hasSystem) _conversationMessages.first,
      ..._conversationMessages.sublist(tailStart < (hasSystem ? 1 : 0) ? (hasSystem ? 1 : 0) : tailStart),
    ];
    _conversationMessages
      ..clear()
      ..addAll(preserved);
  }

  /// Send a message and get full response.
  ///
  /// Does not leave an orphaned user turn in [_conversationMessages] if the
  /// HTTP call fails or times out.
  Future<String> sendMessage(String message) async {
    if (!OllamaClient.isValidModelName(model)) {
      throw ArgumentError.value(model, 'model', 'Invalid Ollama model name');
    }
    final userEntry = <String, dynamic>{'role': 'user', 'content': message};
    final requestMessages = <Map<String, dynamic>>[
      ..._conversationMessages,
      userEntry,
    ];

    http.Client? ownedClient;
    late final int statusCode;
    late final String respBody;
    try {
      final client = _injectedClient ?? (ownedClient = http.Client());
      final req = http.Request(
        'POST',
        Uri.parse('$baseUrl/chat/completions'),
      )
        ..followRedirects = false
        ..headers['Content-Type'] = 'application/json'
        ..body = jsonEncode({
          'model': model,
          'messages': requestMessages,
          'temperature': temperature,
          'max_tokens': maxTokens,
          'stream': false,
          'keep_alive': 1800,
        });
      final streamed = await client
          .send(req)
          .timeout(const Duration(seconds: 60));
      statusCode = streamed.statusCode;
      if (statusCode >= 300 && statusCode < 400) {
        await streamed.stream.drain<void>().catchError((_) {});
        respBody = 'Redirect blocked';
      } else {
        final declaredLen = streamed.contentLength;
        if (declaredLen != null &&
            (declaredLen < 0 || declaredLen > OllamaClient.maxResponseBytes)) {
          await streamed.stream.drain<void>().catchError((_) {});
          throw const FormatException('Ollama chat completion exceeded 8 MiB limit');
        }
        final builder = BytesBuilder(copy: false);
        await for (final chunk in streamed.stream.timeout(const Duration(seconds: 60))) {
          if (builder.length + chunk.length > OllamaClient.maxResponseBytes) {
            throw const FormatException('Ollama chat completion exceeded 8 MiB limit');
          }
          builder.add(chunk);
        }
        respBody = utf8.decode(builder.takeBytes(), allowMalformed: true);
      }
    } finally {
      ownedClient?.close();
    }

    if (statusCode != 200) {
      final snippet = respBody.length > maxErrorSnippetChars
          ? '${_truncateUtf16Safe(respBody, maxErrorSnippetChars)}...'
          : respBody;
      throw Exception('Ollama error $statusCode: $snippet');
    }

    final json = jsonDecode(respBody);
    if (json is! Map) {
      throw const FormatException('Invalid Ollama chat completion payload');
    }
    final firstChoice = (json['choices'] as List?)?.firstOrNull;
    final msgMap = firstChoice is Map ? firstChoice['message'] : null;
    final rawText = msgMap is Map ? (msgMap['content']?.toString() ?? '') : '';
    final text = ThinkingParser.stripThinking(rawText);

    _conversationMessages.add(userEntry);
    if (text.isNotEmpty) {
      _conversationMessages.add({'role': 'assistant', 'content': text});
    }
    _trimHistory();
    return text;
  }

  /// Stream a response. Yields text chunks and always closes transient HTTP
  /// clients even on cancellation or exception.
  Stream<String> sendMessageStream(String message) async* {
    if (!OllamaClient.isValidModelName(model)) {
      throw ArgumentError.value(model, 'model', 'Invalid Ollama model name');
    }
    final userEntry = <String, dynamic>{'role': 'user', 'content': message};
    final requestMessages = <Map<String, dynamic>>[
      ..._conversationMessages,
      userEntry,
    ];

    http.Client? ownedClient;
    try {
      final request = http.Request(
        'POST',
        Uri.parse('$baseUrl/chat/completions'),
      )..followRedirects = false;
      request.headers['Content-Type'] = 'application/json';
      request.body = jsonEncode({
        'model': model,
        'messages': requestMessages,
        'temperature': temperature,
        'max_tokens': maxTokens,
        'stream': true,
        'keep_alive': 1800,
      });

      final client = _injectedClient ?? (ownedClient = http.Client());
      final response = await client
          .send(request)
          .timeout(const Duration(seconds: 60));

      if (response.statusCode != 200) {
        final bytes = await response.stream
            .timeout(const Duration(seconds: 15))
            .take(maxErrorSnippetChars * 4)
            .expand((c) => c)
            .take(maxErrorSnippetChars * 4)
            .toList();
        var body = utf8.decode(bytes, allowMalformed: true);
        if (body.length > maxErrorSnippetChars) {
          body = '${_truncateUtf16Safe(body, maxErrorSnippetChars)}...';
        }
        yield 'Error: Ollama ${response.statusCode}: $body';
        return;
      }

      final buffer = StringBuffer();
      var lineBuf = '';

      String? extractDelta(String rawLine) {
        final line = rawLine.trim();
        if (line.isEmpty || !line.startsWith('data:')) {
          return null;
        }
        final payload = line.substring(5).trim();
        if (payload.isEmpty || payload == '[DONE]') return null;
        try {
          final json = jsonDecode(payload);
          if (json is! Map) return null;
          final firstChoice = (json['choices'] as List?)?.firstOrNull;
          final delta = firstChoice is Map ? firstChoice['delta'] : null;
          final content = delta is Map ? delta['content']?.toString() : null;
          return (content != null && content.isNotEmpty) ? content : null;
        } catch (_) {
          return null;
        }
      }

      await for (final chunk in response.stream
          .timeout(const Duration(seconds: 60))
          .transform(const Utf8Decoder(allowMalformed: true))) {
        lineBuf += chunk;
        if (lineBuf.length > maxStreamLineChars && !lineBuf.contains('\n')) {
          throw const FormatException('Ollama SSE line exceeded cap');
        }
        while (true) {
          final idx = lineBuf.indexOf('\n');
          if (idx < 0) break;
          final line = lineBuf.substring(0, idx);
          lineBuf = lineBuf.substring(idx + 1);
          final content = extractDelta(line);
          if (content != null) {
            buffer.write(content);
            yield content;
          }
        }
      }
      final tail = extractDelta(lineBuf);
      if (tail != null) {
        buffer.write(tail);
        yield tail;
      }

      final rawFull = buffer.toString();
      final cleanFull = ThinkingParser.stripThinking(rawFull);
      if (rawFull.isNotEmpty) {
        _conversationMessages.add(userEntry);
        if (cleanFull.isNotEmpty) {
          _conversationMessages.add({'role': 'assistant', 'content': cleanFull});
        }
        _trimHistory();
      }
    } finally {
      ownedClient?.close();
    }
  }

  /// Clear conversation history (keep system instruction).
  void clearHistory() {
    _conversationMessages.removeWhere((m) => m['role'] != 'system');
  }

  /// Get current conversation messages (unmodifiable snapshot).
  List<Map<String, dynamic>> get conversationMessages =>
      List.unmodifiable(_conversationMessages);

  /// Rebuilds conversation history from list of messages.
  void rebuildHistory(List<Map<String, dynamic>> messages) {
    _conversationMessages
        ..clear()
        ..addAll(messages.map(_sanitizeHistoryEntry));
    _trimHistory();
  }
}
