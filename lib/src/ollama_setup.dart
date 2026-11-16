import 'ollama_client.dart';
import 'ollama_setup_stub.dart'
    if (dart.library.io) 'ollama_setup_io.dart'
    as platform;

/// Automated Ollama setup — install check, server start, model provisioning.
///
/// Used by onboarding flows to ensure Ollama + a model are ready.
/// Safe to import on Web / Wasm (`isInstalled` and `startServer` return `false`).
class OllamaSetup {
  final OllamaClient client;

  OllamaSetup({OllamaClient? client}) : client = client ?? OllamaClient();

  /// Check if the `ollama` binary is installed on this machine.
  Future<bool> isInstalled() => platform.isOllamaInstalled();

  /// Start the Ollama server process. Waits up to 10s for it to respond.
  Future<bool> startServer() => platform.startOllamaServer(client);

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
    void Function(OllamaSetupStep step, String message, double progress)?
    onProgress,
  }) async {
    // 1. Check Ollama
    onProgress?.call(OllamaSetupStep.checkingOllama, 'Checking Ollama...', 0.0);
    if (!await isInstalled()) {
      onProgress?.call(
        OllamaSetupStep.installRequired,
        'Ollama not installed. Download at ollama.com',
        0.0,
      );
      return false;
    }

    if (!await client.isRunning) {
      onProgress?.call(
        OllamaSetupStep.startingOllama,
        'Starting Ollama...',
        0.1,
      );
      if (!await startServer()) {
        onProgress?.call(OllamaSetupStep.error, 'Failed to start Ollama', 0.1);
        return false;
      }
    }
    onProgress?.call(OllamaSetupStep.ollamaReady, 'Ollama running', 0.2);

    // 2. Pull model if not present
    if (!await client.hasModel(modelName)) {
      onProgress?.call(
        OllamaSetupStep.pullingModel,
        'Downloading $modelName...',
        0.25,
      );

      await for (final p in client.pullModel(modelName)) {
        if (p.isError) {
          onProgress?.call(
            OllamaSetupStep.error,
            'Download error: ${p.status}',
            0.25,
          );
          return false;
        }
        final prog = p.progress ?? 0.0;
        onProgress?.call(
          OllamaSetupStep.pullingModel,
          p.status,
          0.25 + (prog * 0.5),
        );
      }
    }

    // 3. Create custom model if modelfile provided
    if (modelfile != null) {
      onProgress?.call(
        OllamaSetupStep.creatingModel,
        'Creating custom model...',
        0.8,
      );
      if (!await client.createModel(modelName, modelfile)) {
        onProgress?.call(OllamaSetupStep.error, 'Failed to create model', 0.8);
        return false;
      }
    }

    onProgress?.call(OllamaSetupStep.ready, 'Ready', 1.0);
    return true;
  }

  /// Download URL by platform.
  static String get downloadUrl => platform.ollamaDownloadUrl;
}

enum OllamaSetupStatus { notInstalled, notRunning, modelMissing, ready }

enum OllamaSetupStep {
  checkingOllama,
  installRequired,
  startingOllama,
  ollamaReady,
  pullingModel,
  creatingModel,
  ready,
  error,
}
