// Web / Wasm hardware detection using modern `dart:js_interop` APIs.
//
// Available signals:
// - navigator.deviceMemory → RAM (capped at 8, rounded to power of 2)
// - navigator.hardwareConcurrency → logical CPU cores
// - WebGL UNMASKED_RENDERER_WEBGL → GPU name (no VRAM info)
// - navigator.platform / userAgent → OS detection
//
// All values are marked as isEstimated=true so the UI shows
// manual override controls prominently.

import 'dart:js_interop';
import 'dart:js_interop_unsafe';

import 'hardware_profile_model.dart';

Future<HardwareProfile> detectHardware() async {
  int totalRam = 8;
  int cpuCores = 4;
  String cpuName = 'Unknown (Web)';
  String? gpuRenderer;
  GpuBackend gpu = GpuBackend.none;
  bool isMac = false;

  try {
    final nav = globalContext['navigator'];
    if (nav != null && nav.isA<JSObject>()) {
      final navObj = nav as JSObject;

      // RAM — navigator.deviceMemory (Chrome, Edge, Opera)
      final devMem = navObj['deviceMemory'];
      if (devMem != null && devMem.isA<JSNumber>()) {
        final gb = (devMem as JSNumber).toDartDouble;
        if (gb.isFinite && gb > 0) {
          totalRam = gb.round().clamp(1, 128);
        }
      }

      // CPU cores — navigator.hardwareConcurrency
      final concurrency = navObj['hardwareConcurrency'];
      if (concurrency != null && concurrency.isA<JSNumber>()) {
        final cores = (concurrency as JSNumber).toDartDouble;
        if (cores.isFinite && cores >= 1) {
          cpuCores = cores.round().clamp(1, 256);
        }
      }

      // OS detection
      final platformJs = navObj['platform'];
      final uaJs = navObj['userAgent'];
      final platform = platformJs != null && platformJs.isA<JSString>()
          ? (platformJs as JSString).toDart.toLowerCase()
          : '';
      final userAgent = uaJs != null && uaJs.isA<JSString>()
          ? (uaJs as JSString).toDart.toLowerCase()
          : '';
      isMac = platform.contains('mac') || userAgent.contains('macintosh');

      if (isMac) {
        cpuName = 'Apple (Web — exact chip unknown)';
      } else if (platform.contains('win')) {
        cpuName = 'Windows PC (Web)';
      } else if (platform.contains('linux')) {
        cpuName = 'Linux (Web)';
      }
    }

    // GPU — WebGL renderer string via document.createElement('canvas')
    final doc = globalContext['document'];
    if (doc != null && doc.isA<JSObject>()) {
      final docObj = doc as JSObject;
      final canvasAny = docObj.callMethod<JSAny?>('createElement'.toJS, 'canvas'.toJS);
      if (canvasAny != null && canvasAny.isA<JSObject>()) {
        final canvas = canvasAny as JSObject;
        final glAny =
            canvas.callMethod<JSAny?>('getContext'.toJS, 'webgl2'.toJS) ??
            canvas.callMethod<JSAny?>('getContext'.toJS, 'webgl'.toJS);
        if (glAny != null && glAny.isA<JSObject>()) {
          final gl = glAny as JSObject;
          try {
            final debugExt = gl.callMethod<JSAny?>(
              'getExtension'.toJS,
              'WEBGL_debug_renderer_info'.toJS,
            );
            if (debugExt != null) {
              // 0x9246 = UNMASKED_RENDERER_WEBGL
              final rendererAny = gl.callMethod<JSAny?>(
                'getParameter'.toJS,
                0x9246.toJS,
              );
              if (rendererAny != null && rendererAny.isA<JSString>()) {
                var renderer = (rendererAny as JSString)
                    .toDart
                    .replaceAll(
                      RegExp(
                        r'[\x00-\x1F\x7F-\x9F\u202A-\u202E\u2066-\u2069\u061C]',
                      ),
                      ' ',
                    )
                    .trim();
                if (renderer.length > 256) {
                  renderer = renderer.substring(0, 256).trim();
                }
                if (renderer.isNotEmpty) {
                  gpuRenderer = renderer;
                  gpu = GpuBackend.webgl;

                  final lower = renderer.toLowerCase();
                  if (lower.contains('apple m4')) {
                    cpuName = 'Apple M4 (via WebGL)';
                  } else if (lower.contains('apple m3')) {
                    cpuName = 'Apple M3 (via WebGL)';
                  } else if (lower.contains('apple m2')) {
                    cpuName = 'Apple M2 (via WebGL)';
                  } else if (lower.contains('apple m1')) {
                    cpuName = 'Apple M1 (via WebGL)';
                  } else if (lower.contains('apple')) {
                    cpuName = 'Apple GPU (via WebGL)';
                  }
                }
              }
            }
          } finally {
            // Explicitly release temporary WebGL context so repeated hardware
            // probes never exhaust the browser's active WebGL context limit (~16).
            try {
              final loseExt = gl.callMethod<JSAny?>(
                'getExtension'.toJS,
                'WEBGL_lose_context'.toJS,
              );
              if (loseExt != null && loseExt.isA<JSObject>()) {
                (loseExt as JSObject).callMethod<JSAny?>('loseContext'.toJS);
              }
            } catch (_) {}
          }
        }
      }
    }
  } catch (_) {}

  if (totalRam == 8 && cpuCores >= 8) {
    totalRam = 16;
  }

  final tier = HardwareProfile.classifyTier(totalRam, gpu, 0, null, null);

  return HardwareProfile(
    totalRamGB: totalRam,
    availableRamGB: (totalRam * 0.5).round(),
    cpuName: cpuName,
    cpuCores: cpuCores,
    gpuBackend: gpu,
    gpuVramGB: 0,
    availableDiskGB: 0,
    isMac: isMac,
    isWeb: true,
    gpuRenderer: gpuRenderer,
    tier: tier,
    isEstimated: true,
  );
}
