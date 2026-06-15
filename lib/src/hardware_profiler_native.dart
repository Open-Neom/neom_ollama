// Native (macOS, Linux, Windows) hardware detection.

import 'dart:io';

import 'hardware_profile_model.dart';

Future<HardwareProfile> detectHardware() async {
  int totalRam = 16, availableRam = 8;
  String cpuName = 'Unknown';
  int cpuCores = 4;
  GpuBackend gpu = GpuBackend.none;
  int vram = 0;
  int disk = 50;
  bool isMac = Platform.isMacOS;
  String? macChip;
  String? macTier;

  try {
    if (Platform.isMacOS) {
      final results = await Future.wait([
        _run('sysctl', ['-n', 'hw.memsize']),
        _run('sysctl', ['-n', 'machdep.cpu.brand_string']),
        _run('sysctl', ['-n', 'hw.ncpu']),
        _run('df', ['-g', '/']),
        _run('vm_stat', []),
      ]);

      // RAM total
      final memBytes = int.tryParse(results[0].trim()) ?? 0;
      if (memBytes > 0) totalRam = (memBytes / (1024 * 1024 * 1024)).round();

      // CPU
      cpuName = results[1].trim();
      cpuCores = int.tryParse(results[2].trim()) ?? 4;

      // Disk
      final dfLines = results[3].trim().split('\n');
      if (dfLines.length > 1) {
        final parts = dfLines[1].split(RegExp(r'\s+'));
        if (parts.length >= 4) disk = int.tryParse(parts[3]) ?? 50;
      }

      // Available RAM from vm_stat
      availableRam = _parseVmStat(results[4], totalRam);

      // Mac chip detection
      final lower = cpuName.toLowerCase();
      if (lower.contains('m4')) macChip = 'M4';
      else if (lower.contains('m3')) macChip = 'M3';
      else if (lower.contains('m2')) macChip = 'M2';
      else if (lower.contains('m1')) macChip = 'M1';

      if (lower.contains('ultra')) macTier = 'ultra';
      else if (lower.contains('max')) macTier = 'max';
      else if (lower.contains('pro')) macTier = 'pro';
      else macTier = 'base';

      // Apple Silicon always has Metal
      if (macChip != null) gpu = GpuBackend.metal;
      vram = totalRam; // Unified memory

    } else if (Platform.isLinux) {
      isMac = false;
      final results = await Future.wait([
        _run('grep', ['MemTotal', '/proc/meminfo']),
        _run('grep', ['MemAvailable', '/proc/meminfo']),
        _run('nproc', []),
        _run('df', ['-BG', '/']),
      ]);

      final memMatch = RegExp(r'(\d+)').firstMatch(results[0]);
      if (memMatch != null) totalRam = (int.parse(memMatch.group(1)!) / (1024 * 1024)).round();

      final availMatch = RegExp(r'(\d+)').firstMatch(results[1]);
      if (availMatch != null) availableRam = (int.parse(availMatch.group(1)!) / (1024 * 1024)).round();

      cpuCores = int.tryParse(results[2].trim()) ?? 4;

      // Detect NVIDIA GPU
      final nvidiaSmi = await _run('nvidia-smi', ['--query-gpu=memory.total', '--format=csv,noheader,nounits']);
      if (nvidiaSmi.isNotEmpty && !nvidiaSmi.contains('not found')) {
        gpu = GpuBackend.cuda;
        vram = (int.tryParse(nvidiaSmi.trim()) ?? 0) ~/ 1024;
      }

      // Detect AMD GPU
      if (gpu == GpuBackend.none) {
        final rocm = await _run('rocm-smi', ['--showmeminfo', 'vram']);
        if (rocm.isNotEmpty && !rocm.contains('not found')) {
          gpu = GpuBackend.rocm;
        }
      }

      cpuName = await _run('grep', ['-m1', 'model name', '/proc/cpuinfo']);
      cpuName = cpuName.replaceAll(RegExp(r'model name\s*:\s*'), '').trim();

    } else if (Platform.isWindows) {
      isMac = false;
      final memResult = await _run('wmic', ['OS', 'get', 'TotalVisibleMemorySize', '/VALUE']);
      final memMatch = RegExp(r'(\d+)').firstMatch(memResult);
      if (memMatch != null) totalRam = (int.parse(memMatch.group(1)!) / (1024 * 1024)).round();
      availableRam = (totalRam * 0.6).round();
    }
  } catch (_) {}

  final tier = HardwareProfile.classifyTier(totalRam, gpu, vram, macChip, macTier);

  return HardwareProfile(
    totalRamGB: totalRam,
    availableRamGB: availableRam,
    cpuName: cpuName,
    cpuCores: cpuCores,
    gpuBackend: gpu,
    gpuVramGB: vram,
    availableDiskGB: disk,
    isMac: isMac,
    isWeb: false,
    macChip: macChip,
    macTier: macTier,
    tier: tier,
    isEstimated: false,
  );
}

int _parseVmStat(String vmstatOutput, int totalRamGB) {
  int freePages = 0;
  int inactivePages = 0;
  const pageSize = 16384; // Apple Silicon page size

  for (final line in vmstatOutput.split('\n')) {
    if (line.contains('Pages free')) {
      final match = RegExp(r'(\d+)').firstMatch(line);
      if (match != null) freePages = int.parse(match.group(1)!);
    } else if (line.contains('Pages inactive')) {
      final match = RegExp(r'(\d+)').firstMatch(line);
      if (match != null) inactivePages = int.parse(match.group(1)!);
    }
  }

  final freeGB = ((freePages + inactivePages) * pageSize) / (1024 * 1024 * 1024);
  return freeGB.round().clamp(1, totalRamGB);
}

Future<String> _run(String cmd, List<String> args) async {
  try {
    final result = await Process.run(cmd, args);
    return result.exitCode == 0 ? result.stdout.toString() : '';
  } catch (_) {
    return '';
  }
}
