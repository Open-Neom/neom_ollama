// Hardware detection and profiling for local model optimization.
// Detects RAM, CPU, GPU (Metal/CUDA), and disk space to inform
// model selection and runtime optimization.
//
// On native: precise detection via system commands.
// On web: uses navigator.deviceMemory, hardwareConcurrency, and
// WebGL renderer string. Values are marked as estimated so the
// UI can show manual override controls prominently.

import 'hardware_profiler_stub.dart'
    if (dart.library.io) 'hardware_profiler_native.dart'
    if (dart.library.html) 'hardware_profiler_web.dart'
    as platform;

// ═══════════════════════════════════════════
// Enums
// ═══════════════════════════════════════════

enum GpuBackend { metal, cuda, rocm, webgl, none }

enum HardwareTier {
  ultra,    // 32+ GB RAM, dedicated GPU or M3/M4
  high,     // 24+ GB RAM or 8+ GB VRAM
  medium,   // 16+ GB RAM
  low,      // 8-15 GB RAM
  minimal,  // < 8 GB RAM
}

// ═══════════════════════════════════════════
// Hardware Profile
// ═══════════════════════════════════════════

class HardwareProfile {
  final int totalRamGB;
  final int availableRamGB;
  final String cpuName;
  final int cpuCores;
  final GpuBackend gpuBackend;
  final int gpuVramGB;
  final int availableDiskGB;
  final bool isMac;
  final bool isWeb;
  final String? macChip;       // M1, M2, M3, M4
  final String? macTier;       // base, pro, max, ultra
  final String? gpuRenderer;   // WebGL renderer string
  final HardwareTier tier;

  /// True when values are estimated (web) vs measured (native).
  /// When true, the UI should show manual override controls prominently.
  final bool isEstimated;

  const HardwareProfile({
    required this.totalRamGB,
    required this.availableRamGB,
    required this.cpuName,
    required this.cpuCores,
    required this.gpuBackend,
    required this.gpuVramGB,
    required this.availableDiskGB,
    required this.isMac,
    this.isWeb = false,
    this.macChip,
    this.macTier,
    this.gpuRenderer,
    required this.tier,
    this.isEstimated = false,
  });

  /// Creates a copy with manual overrides applied.
  HardwareProfile withOverrides({
    int? ramGB,
    bool? hasGpu,
    int? vramGB,
    GpuBackend? gpu,
  }) {
    final newRam = ramGB ?? totalRamGB;
    final newGpu = gpu ?? (hasGpu == true ? GpuBackend.cuda : gpuBackend);
    final newVram = vramGB ?? gpuVramGB;

    return HardwareProfile(
      totalRamGB: newRam,
      availableRamGB: (newRam * 0.6).round(),
      cpuName: cpuName,
      cpuCores: cpuCores,
      gpuBackend: hasGpu == false ? GpuBackend.none : newGpu,
      gpuVramGB: hasGpu == false ? 0 : newVram,
      availableDiskGB: availableDiskGB,
      isMac: isMac,
      isWeb: isWeb,
      macChip: macChip,
      macTier: macTier,
      gpuRenderer: gpuRenderer,
      tier: classifyTier(newRam, newGpu, newVram, macChip, macTier),
      isEstimated: isEstimated,
    );
  }

  /// Maximum model size in GB that can run comfortably.
  double get maxModelSizeGB {
    if (isMac) return totalRamGB * 0.65;
    if (gpuBackend != GpuBackend.none && gpuBackend != GpuBackend.webgl && gpuVramGB > 0) {
      return gpuVramGB * 0.85;
    }
    return totalRamGB * 0.45;
  }

  /// Whether the hardware can run a model of given disk size.
  bool canRunModel(double modelSizeGB) => modelSizeGB <= maxModelSizeGB;

  /// Estimated tokens/second for a model of given parameter count (billions).
  int estimateTokensPerSecond(double paramsBillions) {
    int baseSpeed;
    if (paramsBillions <= 2) baseSpeed = 45;
    else if (paramsBillions <= 4) baseSpeed = 35;
    else if (paramsBillions <= 9) baseSpeed = 22;
    else if (paramsBillions <= 14) baseSpeed = 15;
    else baseSpeed = 8;

    final multiplier = switch (tier) {
      HardwareTier.ultra   => 1.4,
      HardwareTier.high    => 1.2,
      HardwareTier.medium  => 1.0,
      HardwareTier.low     => 0.6,
      HardwareTier.minimal => 0.3,
    };

    return (baseSpeed * multiplier).round().clamp(1, 100);
  }

  Map<String, dynamic> toJson() => {
    'total_ram_gb': totalRamGB,
    'available_ram_gb': availableRamGB,
    'cpu_name': cpuName,
    'cpu_cores': cpuCores,
    'gpu_backend': gpuBackend.name,
    'gpu_vram_gb': gpuVramGB,
    'gpu_renderer': gpuRenderer,
    'available_disk_gb': availableDiskGB,
    'is_mac': isMac,
    'is_web': isWeb,
    'is_estimated': isEstimated,
    'mac_chip': macChip,
    'mac_tier': macTier,
    'tier': tier.name,
    'max_model_size_gb': maxModelSizeGB,
  };

  @override
  String toString() =>
      'HardwareProfile($cpuName, ${totalRamGB}GB RAM, '
      '${gpuBackend.name}${gpuVramGB > 0 ? " ${gpuVramGB}GB" : ""}, '
      'tier: ${tier.name}${isEstimated ? " [estimated]" : ""})';

  /// Classifies hardware tier. Public so platform files can use it.
  static HardwareTier classifyTier(
    int ram, GpuBackend gpu, int vram, String? macChip, String? macTier,
  ) {
    if (macChip != null) {
      final isNew = macChip == 'M3' || macChip == 'M4';
      if (ram >= 32 || macTier == 'max' || macTier == 'ultra') return HardwareTier.ultra;
      if (ram >= 24 || (isNew && ram >= 16 && macTier == 'pro')) return HardwareTier.high;
      if (ram >= 16) return HardwareTier.medium;
      if (ram >= 8) return HardwareTier.low;
      return HardwareTier.minimal;
    }
    if (gpu != GpuBackend.none && gpu != GpuBackend.webgl && vram > 0) {
      if (vram >= 16 && ram >= 32) return HardwareTier.ultra;
      if (vram >= 8 && ram >= 16) return HardwareTier.high;
      if (vram >= 6) return HardwareTier.medium;
      return HardwareTier.low;
    }
    if (ram >= 32) return HardwareTier.medium;
    if (ram >= 16) return HardwareTier.low;
    return HardwareTier.minimal;
  }
}

// ═══════════════════════════════════════════
// Profiler
// ═══════════════════════════════════════════

class HardwareProfiler {
  const HardwareProfiler();

  /// Detects hardware specs. Works on all platforms (native + web).
  /// On web, values are estimated from browser APIs and [isEstimated] is true.
  Future<HardwareProfile> detect() async {
    return platform.detectHardware();
  }
}
