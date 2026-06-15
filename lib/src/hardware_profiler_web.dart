// Web hardware detection using browser APIs.
//
// Available signals:
// - navigator.deviceMemory → RAM (capped at 8, rounded to power of 2)
// - navigator.hardwareConcurrency → logical CPU cores
// - WebGL UNMASKED_RENDERER_WEBGL → GPU name (no VRAM info)
// - navigator.platform / userAgentData → OS detection
//
// All values are marked as isEstimated=true so the UI shows
// manual override controls prominently.

// ignore: avoid_web_libraries_in_flutter
import 'dart:html' as html;
// ignore: avoid_web_libraries_in_flutter
import 'dart:js_util' as js_util;

import 'hardware_profile_model.dart';

Future<HardwareProfile> detectHardware() async {
  int totalRam = 8;
  int cpuCores = 4;
  String cpuName = 'Unknown (Web)';
  String? gpuRenderer;
  GpuBackend gpu = GpuBackend.none;
  bool isMac = false;

  try {
    // RAM — navigator.deviceMemory (Chrome, Edge, Opera)
    // Returns: 0.25, 0.5, 1, 2, 4, 8 (capped at 8 GB, rounded down)
    final deviceMemory =
        js_util.getProperty<Object?>(html.window.navigator, 'deviceMemory');
    if (deviceMemory != null) {
      totalRam = (deviceMemory as num).round().clamp(1, 128);
    }

    // CPU cores — navigator.hardwareConcurrency (all browsers)
    cpuCores = html.window.navigator.hardwareConcurrency ?? 4;

    // OS detection
    final platform = html.window.navigator.platform?.toLowerCase() ?? '';
    final userAgent = html.window.navigator.userAgent.toLowerCase();
    isMac = platform.contains('mac') || userAgent.contains('macintosh');

    if (isMac) {
      cpuName = 'Apple (Web — exact chip unknown)';
    } else if (platform.contains('win')) {
      cpuName = 'Windows PC (Web)';
    } else if (platform.contains('linux')) {
      cpuName = 'Linux (Web)';
    }

    // GPU — WebGL renderer string
    final canvas = html.CanvasElement();
    final gl = canvas.getContext('webgl2') ?? canvas.getContext('webgl');
    if (gl != null) {
      final debugExt = js_util.callMethod<Object?>(
          gl, 'getExtension', ['WEBGL_debug_renderer_info']);
      if (debugExt != null) {
        // 0x9246 = UNMASKED_RENDERER_WEBGL
        final renderer = js_util.callMethod(gl, 'getParameter', [0x9246]) as String?;
        if (renderer != null && renderer.isNotEmpty) {
          gpuRenderer = renderer;
          gpu = GpuBackend.webgl;

          // Infer Mac chip from renderer if possible
          final lower = renderer.toLowerCase();
          if (lower.contains('apple m4')) cpuName = 'Apple M4 (via WebGL)';
          else if (lower.contains('apple m3')) cpuName = 'Apple M3 (via WebGL)';
          else if (lower.contains('apple m2')) cpuName = 'Apple M2 (via WebGL)';
          else if (lower.contains('apple m1')) cpuName = 'Apple M1 (via WebGL)';
          else if (lower.contains('apple')) cpuName = 'Apple GPU (via WebGL)';
        }
      }
    }
  } catch (_) {}

  // On web, navigator.deviceMemory caps at 8GB.
  // If user has 8+ cores it's likely a better machine, bump estimate.
  if (totalRam == 8 && cpuCores >= 8) {
    totalRam = 16; // Better estimate for modern machines
  }

  final tier = HardwareProfile.classifyTier(
    totalRam, gpu, 0, null, null,
  );

  return HardwareProfile(
    totalRamGB: totalRam,
    availableRamGB: (totalRam * 0.5).round(),
    cpuName: cpuName,
    cpuCores: cpuCores,
    gpuBackend: gpu,
    gpuVramGB: 0, // Not available via WebGL
    availableDiskGB: 0, // Not available in web
    isMac: isMac,
    isWeb: true,
    gpuRenderer: gpuRenderer,
    tier: tier,
    isEstimated: true,
  );
}
