// Runtime model optimization for Ollama.
// Handles model preloading, unloading, memory monitoring,
// and inference profiling — inspired by AirLLM patterns.

import 'dart:async';

import 'hardware_profiler.dart';
import 'ollama_client.dart';
import 'thinking_parser.dart';

// ═══════════════════════════════════════════
// Inference Stats
// ═══════════════════════════════════════════

class InferenceStats {
  final Duration loadDuration;
  final Duration promptEvalDuration;
  final Duration totalDuration;
  final int promptTokens;
  final int completionTokens;
  final double tokensPerSecond;

  const InferenceStats({
    required this.loadDuration,
    required this.promptEvalDuration,
    required this.totalDuration,
    required this.promptTokens,
    required this.completionTokens,
    required this.tokensPerSecond,
  });

  static int _finiteNonNegInt(Object? raw, {int fallback = 0}) {
    if (raw is num && raw.isFinite && raw >= 0) {
      return raw is int ? raw : raw.toInt();
    }
    return fallback;
  }

  factory InferenceStats.fromOllamaResponse(Map<Object?, Object?> json) {
    final loadNs = _finiteNonNegInt(json['load_duration']);
    final promptNs = _finiteNonNegInt(json['prompt_eval_duration']);
    final totalNs = _finiteNonNegInt(json['total_duration']);
    final promptTokens = _finiteNonNegInt(json['prompt_eval_count']);
    final completionTokens = _finiteNonNegInt(json['eval_count']);
    final evalNs = _finiteNonNegInt(json['eval_duration'], fallback: 1);

    return InferenceStats(
      loadDuration: Duration(microseconds: loadNs ~/ 1000),
      promptEvalDuration: Duration(microseconds: promptNs ~/ 1000),
      totalDuration: Duration(microseconds: totalNs ~/ 1000),
      promptTokens: promptTokens,
      completionTokens: completionTokens,
      tokensPerSecond: evalNs > 0
          ? (completionTokens / (evalNs / 1e9))
          : 0,
    );
  }

  Map<String, dynamic> toJson() => {
    'load_ms': loadDuration.inMilliseconds,
    'prompt_eval_ms': promptEvalDuration.inMilliseconds,
    'total_ms': totalDuration.inMilliseconds,
    'prompt_tokens': promptTokens,
    'completion_tokens': completionTokens,
    'tokens_per_second': tokensPerSecond.toStringAsFixed(1),
  };

  @override
  String toString() =>
      '${tokensPerSecond.toStringAsFixed(1)} tok/s, '
      'load: ${loadDuration.inMilliseconds}ms, '
      'total: ${totalDuration.inMilliseconds}ms, '
      '$completionTokens tokens';
}

// ═══════════════════════════════════════════
// Model Load State
// ═══════════════════════════════════════════

enum ModelLoadState { unloaded, loading, loaded, error }

class LoadedModelInfo {
  final String name;
  final ModelLoadState state;
  final DateTime? loadedAt;
  final Duration? loadDuration;

  const LoadedModelInfo({
    required this.name,
    required this.state,
    this.loadedAt,
    this.loadDuration,
  });

  /// Duration since the model was loaded.
  Duration? get idleTime =>
      loadedAt != null ? DateTime.now().difference(loadedAt!) : null;
}

// ═══════════════════════════════════════════
// Optimizer
// ═══════════════════════════════════════════

class ModelOptimizer {
  final OllamaClient _client;
  final HardwareProfiler _profiler;
  final Duration idleUnloadTimeout;

  LoadedModelInfo? _currentModel;
  Timer? _idleTimer;
  HardwareProfile? _cachedProfile;

  ModelOptimizer({
    OllamaClient? client,
    HardwareProfiler? profiler,
    this.idleUnloadTimeout = const Duration(minutes: 10),
  })  : _client = client ?? OllamaClient(),
        _profiler = profiler ?? const HardwareProfiler();

  /// Currently loaded model info.
  LoadedModelInfo? get currentModel => _currentModel;

  /// Cached hardware profile. Call [detectHardware] first.
  HardwareProfile? get hardwareProfile => _cachedProfile;

  /// Detects and caches hardware profile.
  Future<HardwareProfile> detectHardware() async {
    _cachedProfile = await _profiler.detect();
    return _cachedProfile!;
  }

  /// Preloads a model into Ollama's memory by sending a dummy request.
  /// This avoids cold-start latency on the first real user prompt.
  Future<LoadedModelInfo> preloadModel(String modelName) async {
    _currentModel = LoadedModelInfo(
      name: modelName,
      state: ModelLoadState.loading,
    );

    try {
      final sw = Stopwatch()..start();

      // Send a minimal prompt to force model load
      await _client.chat(modelName, 'hi', system: 'Reply with OK only.');

      sw.stop();
      _currentModel = LoadedModelInfo(
        name: modelName,
        state: ModelLoadState.loaded,
        loadedAt: DateTime.now(),
        loadDuration: sw.elapsed,
      );

      _startIdleTimer(modelName);
      return _currentModel!;
    } catch (e) {
      _currentModel = LoadedModelInfo(
        name: modelName,
        state: ModelLoadState.error,
      );
      return _currentModel!;
    }
  }

  /// Unloads the model from Ollama's memory to free RAM/VRAM.
  /// Uses the keep_alive=0 trick via [_client] (`followRedirects = false`).
  Future<void> unloadModel(String modelName) async {
    _cancelIdleTimer();
    if (!OllamaClient.isValidModelName(modelName)) return;
    await _client.unloadModel(modelName);
    _currentModel = LoadedModelInfo(
      name: modelName,
      state: ModelLoadState.unloaded,
    );
  }

  /// Switches models: unloads current, preloads new.
  Future<LoadedModelInfo> switchModel(String newModelName) async {
    if (_currentModel != null &&
        _currentModel!.name != newModelName &&
        _currentModel!.state == ModelLoadState.loaded) {
      await unloadModel(_currentModel!.name);
    }
    return preloadModel(newModelName);
  }

  /// Resets the idle timer. Call this after each user interaction.
  void resetIdleTimer() {
    if (_currentModel?.state != ModelLoadState.loaded) return;
    _startIdleTimer(_currentModel!.name);
  }

  /// Performs a chat and returns both the response and inference stats.
  Future<({String response, InferenceStats stats})> chatWithStats(
    String model,
    String prompt, {
    String? system,
  }) async {
    resetIdleTimer();

    final decoded = await _client.chatRaw(model, prompt, system: system);
    final rawMessage = decoded['message'];
    final rawContent = rawMessage is Map
        ? (rawMessage['content']?.toString() ?? '')
        : '';
    final content = ThinkingParser.stripThinking(rawContent);
    final stats = InferenceStats.fromOllamaResponse(decoded);

    return (response: content, stats: stats);
  }

  /// Checks available resources and warns if running low.
  Future<ResourceWarning?> checkResources() async {
    final hw = _cachedProfile ?? await detectHardware();

    if (hw.availableRamGB < 2) {
      return ResourceWarning(
        level: WarningLevel.critical,
        message: 'Very low available RAM (${hw.availableRamGB} GB). '
            'Model performance will be severely degraded.',
        suggestion: 'Close other applications or use a smaller model.',
      );
    }

    if (hw.availableRamGB < 4) {
      return ResourceWarning(
        level: WarningLevel.warning,
        message: 'Low available RAM (${hw.availableRamGB} GB). '
            'Consider using a smaller quantization.',
        suggestion: 'Try Q4_K_M quantization for better performance.',
      );
    }

    if (hw.availableDiskGB < 5) {
      return ResourceWarning(
        level: WarningLevel.warning,
        message: 'Low disk space (${hw.availableDiskGB} GB). '
            'May not be able to download new models.',
        suggestion: 'Free disk space or remove unused models.',
      );
    }

    return null;
  }

  /// Lists models currently loaded in Ollama's memory.
  Future<List<String>> getLoadedModels() => _client.listRunningModels();

  /// Unloads all models from Ollama memory.
  Future<void> unloadAll() async {
    final loaded = await getLoadedModels();
    for (final name in loaded) {
      await unloadModel(name);
    }
  }

  void _startIdleTimer(String modelName) {
    _cancelIdleTimer();
    _idleTimer = Timer(idleUnloadTimeout, () {
      unloadModel(modelName);
    });
  }

  void _cancelIdleTimer() {
    _idleTimer?.cancel();
    _idleTimer = null;
  }

  /// Call when disposing to clean up timers.
  void dispose() {
    _cancelIdleTimer();
  }
}

// ═══════════════════════════════════════════
// Resource Warning
// ═══════════════════════════════════════════

enum WarningLevel { info, warning, critical }

class ResourceWarning {
  final WarningLevel level;
  final String message;
  final String suggestion;

  const ResourceWarning({
    required this.level,
    required this.message,
    required this.suggestion,
  });
}
