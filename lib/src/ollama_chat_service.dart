import 'dart:convert';

import 'package:http/http.dart' as http;

/// OpenAI-compatible chat service for Ollama.
///
/// Speaks the /v1/chat/completions endpoint — same format as OpenAI, DeepSeek,
/// Groq, etc. This allows any OpenAI-compatible consumer to use Ollama.
class OllamaChatService {
  final String baseUrl;
  final String model;
  final double temperature;
  final int maxTokens;

  final List<Map<String, dynamic>> _conversationMessages = [];

  OllamaChatService({
    this.baseUrl = 'http://localhost:11434/v1',
    required this.model,
    this.temperature = 0.7,
    this.maxTokens = 1024,
  });

  /// Add a system instruction (call before first message).
  void setSystemInstruction(String instruction) {
    _conversationMessages.removeWhere((m) => m['role'] == 'system');
    if (instruction.isNotEmpty) {
      _conversationMessages.insert(0, {'role': 'system', 'content': instruction});
    }
  }

  /// Send a message and get full response.
  Future<String> sendMessage(String message) async {
    _conversationMessages.add({'role': 'user', 'content': message});

    final resp = await http.post(
      Uri.parse('$baseUrl/chat/completions'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'model': model,
        'messages': _conversationMessages,
        'temperature': temperature,
        'max_tokens': maxTokens,
        'stream': false,
        'keep_alive': 1800, // Keep model in RAM for 30 minutes
      }),
    ).timeout(const Duration(seconds: 60));

    if (resp.statusCode != 200) {
      throw Exception('Ollama error ${resp.statusCode}: ${resp.body}');
    }

    final json = jsonDecode(resp.body) as Map<String, dynamic>;
    final content = (json['choices'] as List?)?.firstOrNull;
    final text = (content?['message']?['content'] as String?) ?? '';

    if (text.isNotEmpty) {
      _conversationMessages.add({'role': 'assistant', 'content': text});
    }
    return text;
  }

  /// Stream a response. Yields text chunks.
  Stream<String> sendMessageStream(String message) async* {
    _conversationMessages.add({'role': 'user', 'content': message});

    final request = http.Request('POST', Uri.parse('$baseUrl/chat/completions'));
    request.headers['Content-Type'] = 'application/json';
    request.body = jsonEncode({
      'model': model,
      'messages': _conversationMessages,
      'temperature': temperature,
      'max_tokens': maxTokens,
      'stream': true,
      'keep_alive': 1800, // Keep model in RAM for 30 minutes
    });

    final response = await http.Client().send(request)
        .timeout(const Duration(seconds: 60));

    if (response.statusCode != 200) {
      final body = await response.stream.bytesToString();
      yield 'Error: Ollama ${response.statusCode}: $body';
      return;
    }

    final buffer = StringBuffer();
    String lineBuf = '';

    await for (final chunk in response.stream.transform(utf8.decoder)) {
      lineBuf += chunk;
      while (lineBuf.contains('\n')) {
        final idx = lineBuf.indexOf('\n');
        final line = lineBuf.substring(0, idx).trim();
        lineBuf = lineBuf.substring(idx + 1);

        if (line.isEmpty || line == 'data: [DONE]' || !line.startsWith('data: ')) continue;
        try {
          final json = jsonDecode(line.substring(6)) as Map<String, dynamic>;
          final delta = (json['choices'] as List?)?.firstOrNull?['delta'] as Map<String, dynamic>?;
          final content = delta?['content'] as String?;
          if (content != null && content.isNotEmpty) {
            buffer.write(content);
            yield content;
          }
        } catch (_) {}
      }
    }

    final fullText = buffer.toString();
    if (fullText.isNotEmpty) {
      _conversationMessages.add({'role': 'assistant', 'content': fullText});
    }
  }

  /// Clear conversation history (keep system instruction).
  void clearHistory() {
    _conversationMessages.removeWhere((m) => m['role'] != 'system');
  }

  /// Get current conversation messages.
  List<Map<String, dynamic>> get conversationMessages => _conversationMessages;

  /// Rebuilds conversation history from list of messages.
  void rebuildHistory(List<Map<String, dynamic>> messages) {
    _conversationMessages.clear();
    _conversationMessages.addAll(messages);
  }
}
