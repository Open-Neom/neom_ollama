import 'dart:io';

import 'package:neom_ollama/neom_ollama.dart';
import 'package:neom_ollama/src/hardware_profiler_native.dart';
import 'package:neom_ollama/src/ollama_setup_io.dart';
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

    test('validateOllamaExecutablePath follows symlinks and verifies executable regular file', () {
      final tempDir = Directory.systemTemp.createTempSync('ollama_symlink_test_');
      try {
        final realBin = File('${tempDir.path}/ollama_real');
        realBin.writeAsStringSync('#!/bin/sh\necho ok\n');
        if (!Platform.isWindows) {
          Process.runSync('chmod', ['+x', realBin.path]);
        }

        final symlink = Link('${tempDir.path}/ollama_link')..createSync(realBin.path);
        final resolved = validateOllamaExecutablePath(symlink.path);
        expect(resolved, isNotNull);
        expect(resolved, realBin.resolveSymbolicLinksSync());

        // Broken symlink -> null
        final brokenLink = Link('${tempDir.path}/broken_link')
          ..createSync('${tempDir.path}/does_not_exist');
        expect(validateOllamaExecutablePath(brokenLink.path), isNull);

        // Directory symlink -> null
        final subDir = Directory('${tempDir.path}/subdir')..createSync();
        final dirLink = Link('${tempDir.path}/dir_link')..createSync(subDir.path);
        expect(validateOllamaExecutablePath(dirLink.path), isNull);

        // Non-executable file on POSIX -> null
        if (!Platform.isWindows) {
          final nonExec = File('${tempDir.path}/non_exec')..writeAsStringSync('data');
          Process.runSync('chmod', ['644', nonExec.path]);
          expect(validateOllamaExecutablePath(nonExec.path), isNull);
        }
      } finally {
        if (tempDir.existsSync()) {
          tempDir.deleteSync(recursive: true);
        }
      }
    });
  });

  group('HardwareProfiler /proc/meminfo parser', () {
    test('parses Android/Linux MemTotal and MemAvailable accurately', () {
      const meminfo = '''
MemTotal:        6144000 kB
MemFree:          512000 kB
MemAvailable:    2621440 kB
Buffers:          128000 kB
Cached:          1024000 kB
''';
      final parsed = parseProcMeminfo(meminfo);
      expect(parsed, isNotNull);
      expect(parsed!.totalGb, 6);
      expect(parsed.availableGb, 3);
    });

    test('falls back to MemFree + Buffers + Cached when MemAvailable is absent', () {
      const legacyAndroidMeminfo = '''
MemTotal:        4194304 kB
MemFree:          524288 kB
Buffers:          524288 kB
Cached:          1048576 kB
''';
      final parsed = parseProcMeminfo(legacyAndroidMeminfo);
      expect(parsed, isNotNull);
      expect(parsed!.totalGb, 4);
      expect(parsed.availableGb, 2);
    });
  });
}
