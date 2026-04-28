import 'package:neom_ollama/src/ollama_client.dart';
import 'package:test/test.dart';

void main() {
  group('OllamaModel.fromJson', () {
    test('parses basic model', () {
      final m = OllamaModel.fromJson({
        'name': 'llama3:8b',
        'digest': 'abc',
        'size': 4 * 1024 * 1024 * 1024, // 4 GB
      });
      expect(m.name, 'llama3:8b');
      expect(m.digest, 'abc');
      expect(m.sizeBytes, 4 * 1024 * 1024 * 1024);
    });

    test('falls back to "model" key if "name" missing', () {
      final m = OllamaModel.fromJson({'model': 'mistral'});
      expect(m.name, 'mistral');
    });

    test('empty json yields empty name, no crash', () {
      final m = OllamaModel.fromJson({});
      expect(m.name, '');
      expect(m.digest, isNull);
      expect(m.sizeBytes, isNull);
    });

    test('parses nested details map', () {
      final m = OllamaModel.fromJson({
        'name': 'x',
        'details': {
          'family': 'llama',
          'parameter_size': '8B',
          'quantization_level': 'Q4_0',
        },
      });
      expect(m.family, 'llama');
      expect(m.parameterSize, '8B');
      expect(m.quantizationLevel, 'Q4_0');
    });

    test('parses ISO8601 modified_at', () {
      final m = OllamaModel.fromJson({
        'name': 'x',
        'modified_at': '2025-06-15T10:00:00Z',
      });
      expect(m.modifiedAt, isNotNull);
      expect(m.modifiedAt!.year, 2025);
    });

    test('malformed modified_at becomes null (does not throw)', () {
      final m = OllamaModel.fromJson({
        'name': 'x',
        'modified_at': 'not-a-date',
      });
      expect(m.modifiedAt, isNull);
    });
  });

  group('OllamaModel.sizeLabel', () {
    test('bytes >= 1GB → GB', () {
      final m = OllamaModel(name: 'x', sizeBytes: (1.5 * 1024 * 1024 * 1024).toInt());
      expect(m.sizeLabel, '1.5 GB');
    });

    test('bytes < 1GB → MB', () {
      final m = OllamaModel(name: 'x', sizeBytes: 500 * 1024 * 1024);
      expect(m.sizeLabel, '500 MB');
    });

    test('exactly 1GB → 1.0 GB', () {
      final m = OllamaModel(name: 'x', sizeBytes: 1024 * 1024 * 1024);
      expect(m.sizeLabel, '1.0 GB');
    });

    test('null size → empty label', () {
      const m = OllamaModel(name: 'x');
      expect(m.sizeLabel, '');
    });

    test('zero size → 0 MB', () {
      const m = OllamaModel(name: 'x', sizeBytes: 0);
      expect(m.sizeLabel, '0 MB');
    });
  });

  group('OllamaModel.displayName', () {
    test('strips :latest suffix', () {
      const m = OllamaModel(name: 'llama3:latest');
      expect(m.displayName, 'llama3');
    });

    test('does not touch name without :latest', () {
      const m = OllamaModel(name: 'llama3:8b');
      expect(m.displayName, 'llama3:8b');
    });

    test('handles empty name', () {
      const m = OllamaModel(name: '');
      expect(m.displayName, '');
    });
  });

  group('OllamaPullProgress', () {
    test('progress fraction = completed/total', () {
      const p = OllamaPullProgress(
        status: 'downloading',
        total: 1000,
        completed: 250,
      );
      expect(p.progress, 0.25);
    });

    test('null total → null progress', () {
      const p = OllamaPullProgress(status: 'downloading', completed: 50);
      expect(p.progress, isNull);
    });

    test('zero total → null progress (no div-by-zero)', () {
      const p = OllamaPullProgress(
        status: 'downloading',
        total: 0,
        completed: 0,
      );
      expect(p.progress, isNull);
    });

    test('null completed → null progress', () {
      const p = OllamaPullProgress(status: 'downloading', total: 100);
      expect(p.progress, isNull);
    });

    test('isDone when status == "success"', () {
      expect(const OllamaPullProgress(status: 'success').isDone, isTrue);
      expect(const OllamaPullProgress(status: 'downloading').isDone, isFalse);
    });

    test('isError when status starts with "error"', () {
      expect(const OllamaPullProgress(status: 'error: x').isError, isTrue);
      expect(const OllamaPullProgress(status: 'errors').isError, isTrue);
      expect(const OllamaPullProgress(status: 'success').isError, isFalse);
    });
  });

  group('OllamaClient openAiBaseUrl', () {
    test('appends /v1 to host', () {
      final c = OllamaClient(host: 'http://localhost:11434');
      expect(c.openAiBaseUrl, 'http://localhost:11434/v1');
    });

    test('custom host is honored', () {
      final c = OllamaClient(host: 'http://example.com:1234');
      expect(c.openAiBaseUrl, 'http://example.com:1234/v1');
    });
  });
}
