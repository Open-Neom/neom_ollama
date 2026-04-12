// Model recommendation engine based on hardware profile.
// Suggests the optimal quantization and model variant
// that fits the user's hardware constraints.

import 'hardware_profiler.dart';

// ═══════════════════════════════════════════
// Quantization
// ═══════════════════════════════════════════

enum Quantization {
  q4KM('Q4_K_M', 0.28, 'Smallest, fastest. ~5% quality loss.'),
  q4KS('Q4_K_S', 0.26, 'Slightly smaller than Q4_K_M.'),
  q5KM('Q5_K_M', 0.35, 'Good balance. ~3% quality loss.'),
  q5KS('Q5_K_S', 0.33, 'Slightly smaller than Q5_K_M.'),
  q6K('Q6_K', 0.50, 'High quality. ~1% quality loss.'),
  q8('Q8_0', 0.56, 'Near-lossless. Needs more RAM.'),
  fp16('FP16', 1.0, 'Full precision. Best quality, most RAM.');

  const Quantization(this.label, this.sizeRatio, this.description);

  /// User-facing label (e.g. "Q4_K_M").
  final String label;

  /// Ratio vs FP16 size (Q4_K_M ≈ 28% of FP16).
  final double sizeRatio;

  final String description;

  /// Estimated disk/RAM size in GB for a model with [paramsBillions] parameters.
  double estimateSizeGB(double paramsBillions) {
    // FP16 ≈ 2 bytes per param → paramsBillions * 2 GB base
    return paramsBillions * 2 * sizeRatio;
  }
}

// ═══════════════════════════════════════════
// Model Variant
// ═══════════════════════════════════════════

class ModelVariant {
  final String name;
  final String tag;
  final double paramsBillions;
  final double diskSizeGB;
  final Quantization quantization;

  const ModelVariant({
    required this.name,
    required this.tag,
    required this.paramsBillions,
    required this.diskSizeGB,
    required this.quantization,
  });

  /// Full Ollama model tag (e.g. "qwen2.5-coder:7b-q4_K_M").
  String get ollamaTag => '$name:$tag';
}

// ═══════════════════════════════════════════
// Recommendation Result
// ═══════════════════════════════════════════

class ModelRecommendation {
  final ModelVariant recommended;
  final ModelVariant? upgrade;
  final Quantization bestQuantization;
  final int estimatedTokensPerSecond;
  final String reason;
  final List<ModelVariant> alternatives;
  final bool fitsInMemory;

  const ModelRecommendation({
    required this.recommended,
    this.upgrade,
    required this.bestQuantization,
    required this.estimatedTokensPerSecond,
    required this.reason,
    this.alternatives = const [],
    required this.fitsInMemory,
  });
}

// ═══════════════════════════════════════════
// Known Models Registry
// ═══════════════════════════════════════════

class _KnownModel {
  final String name;
  final double params;
  final List<String> tags;
  final String category; // coding, general, reasoning

  const _KnownModel(this.name, this.params, this.tags, this.category);
}

const _knownModels = <_KnownModel>[
  // Coding
  _KnownModel('qwen2.5-coder', 1.5, ['1.5b'], 'coding'),
  _KnownModel('qwen2.5-coder', 7, ['7b'], 'coding'),
  _KnownModel('qwen2.5-coder', 14, ['14b'], 'coding'),
  _KnownModel('qwen2.5-coder', 32, ['32b'], 'coding'),
  _KnownModel('deepseek-coder-v2', 16, ['16b'], 'coding'),

  // General
  _KnownModel('llama3.2', 3, ['3b'], 'general'),
  _KnownModel('llama3.1', 8, ['8b'], 'general'),
  _KnownModel('llama3.1', 70, ['70b'], 'general'),
  _KnownModel('gemma2', 2, ['2b'], 'general'),
  _KnownModel('gemma2', 9, ['9b'], 'general'),
  _KnownModel('gemma2', 27, ['27b'], 'general'),
  _KnownModel('qwen2.5', 3, ['3b'], 'general'),
  _KnownModel('qwen2.5', 7, ['7b'], 'general'),
  _KnownModel('qwen2.5', 14, ['14b'], 'general'),
  _KnownModel('qwen2.5', 32, ['32b'], 'general'),
  _KnownModel('phi3.5', 3.8, ['3.8b'], 'general'),

  // Reasoning
  _KnownModel('qwen3', 8, ['8b'], 'reasoning'),
  _KnownModel('deepseek-r1', 7, ['7b'], 'reasoning'),
  _KnownModel('deepseek-r1', 14, ['14b'], 'reasoning'),
];

// ═══════════════════════════════════════════
// Advisor
// ═══════════════════════════════════════════

class ModelAdvisor {
  const ModelAdvisor();

  /// Returns the best quantization for a model given hardware constraints.
  Quantization bestQuantization(double paramsBillions, HardwareProfile hw) {
    final maxGB = hw.maxModelSizeGB;

    // Try from highest quality down
    for (final q in [
      Quantization.fp16,
      Quantization.q8,
      Quantization.q6K,
      Quantization.q5KM,
      Quantization.q4KM,
      Quantization.q4KS,
    ]) {
      if (q.estimateSizeGB(paramsBillions) <= maxGB) return q;
    }

    return Quantization.q4KS; // Smallest possible
  }

  /// Recommends the best model for the given hardware and category.
  ModelRecommendation recommend(
    HardwareProfile hw, {
    String category = 'general',
  }) {
    final maxGB = hw.maxModelSizeGB;
    final candidates = _knownModels
        .where((m) => m.category == category)
        .toList()
      ..sort((a, b) => b.params.compareTo(a.params)); // Largest first

    // Find the largest model that fits with at least Q4_K_M
    _KnownModel? best;
    Quantization? bestQ;
    for (final model in candidates) {
      final q = bestQuantization(model.params, hw);
      final size = q.estimateSizeGB(model.params);
      if (size <= maxGB) {
        best = model;
        bestQ = q;
        break;
      }
    }

    // Fallback to smallest model
    best ??= candidates.last;
    bestQ ??= Quantization.q4KS;

    final diskSize = bestQ.estimateSizeGB(best.params);
    final recommended = ModelVariant(
      name: best.name,
      tag: best.tags.first,
      paramsBillions: best.params,
      diskSizeGB: diskSize,
      quantization: bestQ,
    );

    // Find upgrade option (next size up)
    ModelVariant? upgrade;
    final currentIdx = candidates.indexOf(best);
    if (currentIdx > 0) {
      final up = candidates[currentIdx - 1];
      final upQ = bestQuantization(up.params, hw);
      upgrade = ModelVariant(
        name: up.name,
        tag: up.tags.first,
        paramsBillions: up.params,
        diskSizeGB: upQ.estimateSizeGB(up.params),
        quantization: upQ,
      );
    }

    // Build alternatives (other models that fit)
    final alts = <ModelVariant>[];
    for (final model in candidates) {
      if (model == best) continue;
      final q = bestQuantization(model.params, hw);
      final size = q.estimateSizeGB(model.params);
      if (size <= maxGB) {
        alts.add(ModelVariant(
          name: model.name,
          tag: model.tags.first,
          paramsBillions: model.params,
          diskSizeGB: size,
          quantization: q,
        ));
      }
      if (alts.length >= 3) break;
    }

    final tokPerSec = hw.estimateTokensPerSecond(best.params);

    return ModelRecommendation(
      recommended: recommended,
      upgrade: upgrade,
      bestQuantization: bestQ,
      estimatedTokensPerSecond: tokPerSec,
      reason: _buildReason(hw, best, bestQ),
      alternatives: alts,
      fitsInMemory: diskSize <= maxGB,
    );
  }

  /// Checks if a specific model (by param count) can run on this hardware.
  ModelRecommendation checkModel(
    HardwareProfile hw,
    String modelName,
    double paramsBillions,
  ) {
    final q = bestQuantization(paramsBillions, hw);
    final size = q.estimateSizeGB(paramsBillions);
    final fits = size <= hw.maxModelSizeGB;
    final tokPerSec = hw.estimateTokensPerSecond(paramsBillions);

    final variant = ModelVariant(
      name: modelName,
      tag: '${paramsBillions.toStringAsFixed(0)}b',
      paramsBillions: paramsBillions,
      diskSizeGB: size,
      quantization: q,
    );

    return ModelRecommendation(
      recommended: variant,
      bestQuantization: q,
      estimatedTokensPerSecond: tokPerSec,
      reason: fits
          ? 'Model fits with ${q.label} quantization (~${size.toStringAsFixed(1)} GB).'
          : 'Model too large. Needs ${size.toStringAsFixed(1)} GB, you have ${hw.maxModelSizeGB.toStringAsFixed(1)} GB available.',
      fitsInMemory: fits,
    );
  }

  String _buildReason(HardwareProfile hw, _KnownModel model, Quantization q) {
    final size = q.estimateSizeGB(model.params);
    final buf = StringBuffer();
    buf.write('${model.name} ${model.params}B with ${q.label}');
    buf.write(' (~${size.toStringAsFixed(1)} GB)');
    buf.write(' fits in your ${hw.totalRamGB} GB ');
    buf.write(hw.isMac ? 'unified memory' : 'system');
    buf.write(' (${hw.tier.name} tier).');
    return buf.toString();
  }
}
