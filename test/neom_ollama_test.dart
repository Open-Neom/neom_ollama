import 'package:neom_ollama/neom_ollama.dart';
import 'package:test/test.dart';

void main() {
  group('OllamaClient', () {
    test('default host is localhost:11434', () {
      final client = OllamaClient();
      expect(client.host, 'http://localhost:11434');
      expect(client.openAiBaseUrl, 'http://localhost:11434/v1');
    });

    test('custom host is respected', () {
      final client = OllamaClient(host: 'http://192.168.1.50:11434');
      expect(client.host, 'http://192.168.1.50:11434');
      expect(client.openAiBaseUrl, 'http://192.168.1.50:11434/v1');
    });
  });

  group('OllamaModel', () {
    test('fromJson parses correctly', () {
      final model = OllamaModel.fromJson({
        'name': 'qwen2.5:3b',
        'size': 1900000000,
        'details': {
          'family': 'qwen2',
          'parameter_size': '3B',
          'quantization_level': 'Q4_K_M',
        },
      });

      expect(model.name, 'qwen2.5:3b');
      expect(model.displayName, 'qwen2.5:3b');
      expect(model.sizeLabel, '1.8 GB');
      expect(model.family, 'qwen2');
      expect(model.parameterSize, '3B');
      expect(model.quantizationLevel, 'Q4_K_M');
    });

    test('displayName strips :latest', () {
      final model = OllamaModel(name: 'llama3:latest');
      expect(model.displayName, 'llama3');
    });
  });

  group('OllamaPullProgress', () {
    test('progress calculates correctly', () {
      final p = OllamaPullProgress(status: 'downloading', total: 1000, completed: 500);
      expect(p.progress, 0.5);
      expect(p.isDone, false);
      expect(p.isError, false);
    });

    test('isDone on success', () {
      final p = OllamaPullProgress(status: 'success');
      expect(p.isDone, true);
    });

    test('isError on error', () {
      final p = OllamaPullProgress(status: 'error: timeout');
      expect(p.isError, true);
    });
  });

  group('OllamaChatService', () {
    test('constructs with defaults', () {
      final chat = OllamaChatService(model: 'test');
      expect(chat.model, 'test');
      expect(chat.baseUrl, 'http://localhost:11434/v1');
    });
  });

  group('OllamaSetup', () {
    test('downloadUrl returns platform URL', () {
      // Just verify it doesn't throw
      expect(OllamaSetup.downloadUrl, isNotEmpty);
    });
  });
}
