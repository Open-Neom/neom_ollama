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

export 'hardware_profile_model.dart';
import 'hardware_profile_model.dart';

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
