import 'dart:io';

import 'ollama_client.dart';

/// Automated Ollama setup — install check, server start, model provisioning.
///
/// Used by onboarding flows to ensure Ollama + a model are ready.
class OllamaSetup {
  final OllamaClient client;

  OllamaSetup({OllamaClient? client})
      : client = client ?? OllamaClient();

  /// Check if the `ollama` binary is installed on this machine.
  Future<bool> isInstalled() async {
    try {
      final result = await Process.run('which', ['ollama']);
      if (result.exitCode == 0) return true;

      final paths = Platform.isMacOS
          ? ['/usr/local/bin/ollama', '/opt/homebrew/bin/ollama']
          : Platform.isWindows
              ? ['C:\\Users\\${Platform.environment['USERNAME']}\\AppData\\Local\\Programs\\Ollama\\ollama.exe']
              : ['/usr/local/bin/ollama', '/usr/bin/ollama'];

      for (final path in paths) {
        if (await File(path).exists()) return true;
      }
      return false;
    } catch (_) {
      return false;
    }
  }

  /// Start the Ollama server process. Waits up to 10s for it to respond.
  Future<bool> startServer() async {
    try {
      await Process.start('ollama', ['serve'], mode: ProcessStartMode.detached);

      for (int i = 0; i < 20; i++) {
        await Future<void>.delayed(const Duration(milliseconds: 500));
        if (await client.isRunning) return true;
      }
      return false;
    } catch (_) {
      return false;
    }
  }

  /// Full status check.
  Future<OllamaSetupStatus> checkStatus({String? requiredModel}) async {
    if (!await isInstalled()) return OllamaSetupStatus.notInstalled;
    if (!await client.isRunning) return OllamaSetupStatus.notRunning;
    if (requiredModel != null && !await client.hasModel(requiredModel)) {
      return OllamaSetupStatus.modelMissing;
    }
    return OllamaSetupStatus.ready;
  }

  /// Full automated setup: start server → pull model → optional create custom.
  ///
  /// [onProgress] reports step, message, and progress (0.0–1.0).
  Future<bool> runFullSetup({
    required String modelName,
    String? modelfile,
    void Function(OllamaSetupStep step, String message, double progress)? onProgress,
  }) async {
    // 1. Check Ollama
    onProgress?.call(OllamaSetupStep.checkingOllama, 'Checking Ollama...', 0.0);
    if (!await isInstalled()) {
      onProgress?.call(OllamaSetupStep.installRequired, 'Ollama not installed. Download at ollama.com', 0.0);
      return false;
    }

    if (!await client.isRunning) {
      onProgress?.call(OllamaSetupStep.startingOllama, 'Starting Ollama...', 0.1);
      if (!await startServer()) {
        onProgress?.call(OllamaSetupStep.error, 'Failed to start Ollama', 0.1);
        return false;
      }
    }
    onProgress?.call(OllamaSetupStep.ollamaReady, 'Ollama running', 0.2);

    // 2. Pull model if not present
    if (!await client.hasModel(modelName)) {
      onProgress?.call(OllamaSetupStep.pullingModel, 'Downloading $modelName...', 0.25);

      await for (final p in client.pullModel(modelName)) {
        if (p.isError) {
          onProgress?.call(OllamaSetupStep.error, 'Download error: ${p.status}', 0.25);
          return false;
        }
        final prog = p.progress ?? 0.0;
        onProgress?.call(OllamaSetupStep.pullingModel, p.status, 0.25 + (prog * 0.5));
      }
    }

    // 3. Create custom model if modelfile provided
    if (modelfile != null) {
      onProgress?.call(OllamaSetupStep.creatingModel, 'Creating custom model...', 0.8);
      if (!await client.createModel(modelName, modelfile)) {
        onProgress?.call(OllamaSetupStep.error, 'Failed to create model', 0.8);
        return false;
      }
    }

    onProgress?.call(OllamaSetupStep.ready, 'Ready', 1.0);
    return true;
  }

  /// Download URL by platform.
  static String get downloadUrl {
    if (Platform.isMacOS) return 'https://ollama.com/download/mac';
    if (Platform.isWindows) return 'https://ollama.com/download/windows';
    return 'https://ollama.com/download/linux';
  }
}

enum OllamaSetupStatus { notInstalled, notRunning, modelMissing, ready }

enum OllamaSetupStep {
  checkingOllama, installRequired, startingOllama, ollamaReady,
  pullingModel, creatingModel, ready, error,
}
