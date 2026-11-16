// Native (macOS, Linux, Windows) hardware detection.

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'hardware_profile_model.dart';

Future<HardwareProfile> detectHardware() async {
  final bool isMobile = Platform.isAndroid || Platform.isIOS;
  // Conservative fallback: never assume 16 GB total / 8 GB free on unknown mobile devices.
  int totalRam = isMobile ? 4 : 8;
  int availableRam = isMobile ? 2 : 4;
  bool isEstimated = true;
  String cpuName = 'Unknown';
  int cpuCores = Platform.numberOfProcessors.clamp(1, 512);
  GpuBackend gpu = GpuBackend.none;
  int vram = 0;
  int disk = isMobile ? 10 : 50;
  bool isMac = Platform.isMacOS;
  String? macChip;
  String? macTier;

  try {
    if (Platform.isMacOS) {
      final results = await Future.wait([
        _run('sysctl', const ['-n', 'hw.memsize']),
        _run('sysctl', const ['-n', 'machdep.cpu.brand_string']),
        _run('sysctl', const ['-n', 'hw.ncpu']),
        _run('df', const ['-g', '/']),
        _run('vm_stat', const []),
      ]);

      final memBytes = int.tryParse(results[0].trim()) ?? 0;
      if (memBytes > 0) {
        totalRam = (memBytes / (1024 * 1024 * 1024)).round().clamp(1, 2048);
        isEstimated = false;
      }

      if (results[1].trim().isNotEmpty) cpuName = results[1].trim();
      cpuCores = int.tryParse(results[2].trim()) ?? cpuCores;

      final dfLines = results[3].trim().split('\n');
      if (dfLines.length > 1) {
        final parts = dfLines[1].split(RegExp(r'\s+'));
        if (parts.length >= 4) disk = int.tryParse(parts[3]) ?? 50;
      }

      availableRam = parseVmStatAvailableGb(results[4], totalRam);

      final lower = cpuName.toLowerCase();
      if (lower.contains('m4')) {
        macChip = 'M4';
      } else if (lower.contains('m3')) {
        macChip = 'M3';
      } else if (lower.contains('m2')) {
        macChip = 'M2';
      } else if (lower.contains('m1')) {
        macChip = 'M1';
      }

      if (lower.contains('ultra')) {
        macTier = 'ultra';
      } else if (lower.contains('max')) {
        macTier = 'max';
      } else if (lower.contains('pro')) {
        macTier = 'pro';
      } else {
        macTier = 'base';
      }

      if (macChip != null) {
        gpu = GpuBackend.metal;
        vram = totalRam; // Unified memory
      }
    } else if (Platform.isLinux || Platform.isAndroid) {
      isMac = false;

      // Read /proc/meminfo directly in Dart so both Linux and sandboxed Android succeed
      // without requiring external grep binaries.
      String meminfoText = '';
      try {
        final meminfoFile = File('/proc/meminfo');
        if (meminfoFile.existsSync()) {
          meminfoText = await meminfoFile.readAsString();
        }
      } catch (_) {}

      final parsedMem = parseProcMeminfo(meminfoText);
      if (parsedMem != null) {
        totalRam = parsedMem.totalGb;
        availableRam = parsedMem.availableGb;
        isEstimated = false;
      } else if (Platform.isLinux) {
        final results = await Future.wait([
          _run('grep', const ['MemTotal', '/proc/meminfo']),
          _run('grep', const ['MemAvailable', '/proc/meminfo']),
        ]);
        final combined = '${results[0]}\n${results[1]}';
        final fallbackMem = parseProcMeminfo(combined);
        if (fallbackMem != null) {
          totalRam = fallbackMem.totalGb;
          availableRam = fallbackMem.availableGb;
          isEstimated = false;
        }
      }

      try {
        final cpuinfoFile = File('/proc/cpuinfo');
        if (cpuinfoFile.existsSync()) {
          final cpuinfo = await cpuinfoFile.readAsString();
          final cpuMatch = RegExp(
            r'^(?:model name|Hardware|Processor)\s*:\s*(.+)$',
            multiLine: true,
          ).firstMatch(cpuinfo);
          if (cpuMatch != null && cpuMatch.group(1)!.trim().isNotEmpty) {
            cpuName = cpuMatch.group(1)!.trim();
          }
        }
      } catch (_) {}

      if (Platform.isLinux) {
        final results = await Future.wait([
          _run('nproc', const []),
          _run('df', const ['-BG', '/']),
        ]);
        cpuCores = int.tryParse(results[0].trim()) ?? cpuCores;

        final dfLines = results[1].trim().split('\n');
        if (dfLines.length > 1) {
          final parts = dfLines[1].split(RegExp(r'\s+'));
          if (parts.length >= 4) {
            final rawAvail = parts[3].replaceAll(RegExp(r'[^0-9]'), '');
            disk = int.tryParse(rawAvail) ?? disk;
          }
        }

        final nvidiaSmi = await _run(
          'nvidia-smi',
          const ['--query-gpu=memory.total', '--format=csv,noheader,nounits'],
        );
        if (nvidiaSmi.isNotEmpty && !nvidiaSmi.contains('not found')) {
          final firstLine = nvidiaSmi.trim().split('\n').first.trim();
          final parsedMib = int.tryParse(firstLine) ?? 0;
          if (parsedMib > 0) {
            gpu = GpuBackend.cuda;
            vram = parsedMib ~/ 1024;
          }
        }

        if (gpu == GpuBackend.none) {
          final rocm = await _run('rocm-smi', const ['--showmeminfo', 'vram']);
          if (rocm.isNotEmpty && !rocm.contains('not found')) {
            gpu = GpuBackend.rocm;
          }
        }
      }
    } else if (Platform.isIOS) {
      isMac = false;
      cpuName = 'Apple Mobile';
      gpu = GpuBackend.metal;
      final rawMem = await _run('sysctl', const ['-n', 'hw.memsize']);
      final memBytes = int.tryParse(rawMem.trim()) ?? 0;
      if (memBytes > 0) {
        totalRam = (memBytes / (1024 * 1024 * 1024)).round().clamp(1, 64);
        availableRam = (totalRam * 0.5).round().clamp(1, totalRam);
        isEstimated = false;
      }
      vram = totalRam;
    } else if (Platform.isWindows) {
      isMac = false;
      // Windows 11 24H2+ removed `wmic`; query CIM via PowerShell first, then wmic fallback.
      final psMem = await _run('powershell', const [
        '-NoProfile',
        '-NonInteractive',
        '-Command',
        r'$os = Get-CimInstance Win32_OperatingSystem; "$($os.TotalVisibleMemorySize)|$($os.FreePhysicalMemory)"',
      ]);
      final psParts = psMem.trim().split('|');
      if (psParts.length == 2) {
        final totalKb = int.tryParse(psParts[0].trim()) ?? 0;
        final freeKb = int.tryParse(psParts[1].trim()) ?? 0;
        if (totalKb > 0) {
          totalRam = (totalKb / (1024 * 1024)).round().clamp(1, 2048);
          availableRam = freeKb > 0
              ? (freeKb / (1024 * 1024)).round().clamp(1, totalRam)
              : (totalRam * 0.6).round();
          isEstimated = false;
        }
      } else {
        final memResult = await _run(
          'wmic',
          const ['OS', 'get', 'TotalVisibleMemorySize', '/VALUE'],
        );
        final memMatch = RegExp(r'(\d+)').firstMatch(memResult);
        if (memMatch != null) {
          totalRam = (int.parse(memMatch.group(1)!) / (1024 * 1024)).round();
          availableRam = (totalRam * 0.6).round();
          isEstimated = false;
        }
      }

      final procId = Platform.environment['PROCESSOR_IDENTIFIER']?.trim() ?? '';
      if (procId.isNotEmpty) cpuName = procId;

      final nvidiaSmi = await _run(
        'nvidia-smi',
        const ['--query-gpu=memory.total', '--format=csv,noheader,nounits'],
      );
      if (nvidiaSmi.isNotEmpty) {
        final firstLine = nvidiaSmi.trim().split(RegExp(r'[\r\n]+')).first.trim();
        final parsedMib = int.tryParse(firstLine) ?? 0;
        if (parsedMib > 0) {
          gpu = GpuBackend.cuda;
          vram = parsedMib ~/ 1024;
        }
      }
    }
  } catch (_) {}

  final tier = HardwareProfile.classifyTier(
    totalRam,
    gpu,
    vram,
    macChip,
    macTier,
  );

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
    isEstimated: isEstimated,
  );
}

/// Parses Linux/Android `/proc/meminfo` kB lines into rounded total/available GB.
({int totalGb, int availableGb})? parseProcMeminfo(String content) {
  if (content.trim().isEmpty) return null;

  int? totalKb;
  int? availKb;
  int freeKb = 0;
  int buffersKb = 0;
  int cachedKb = 0;

  for (final line in content.split('\n')) {
    final match = RegExp(r'^([A-Za-z_()]+):\s*(\d+)').firstMatch(line.trim());
    if (match == null) continue;
    final key = match.group(1)!;
    final val = int.tryParse(match.group(2)!);
    if (val == null) continue;

    switch (key) {
      case 'MemTotal':
        totalKb = val;
      case 'MemAvailable':
        availKb = val;
      case 'MemFree':
        freeKb = val;
      case 'Buffers':
        buffersKb = val;
      case 'Cached':
        cachedKb = val;
    }
  }

  if (totalKb == null || totalKb <= 0) return null;
  final totalGb = (totalKb / (1024 * 1024)).round().clamp(1, 2048);

  final effectiveAvailKb = availKb ?? (freeKb + buffersKb + cachedKb);
  final availableGb = effectiveAvailKb > 0
      ? (effectiveAvailKb / (1024 * 1024)).round().clamp(1, totalGb)
      : (totalGb * 0.5).round().clamp(1, totalGb);

  return (totalGb: totalGb, availableGb: availableGb);
}

int parseVmStatAvailableGb(String vmstatOutput, int totalRamGB) {
  int freePages = 0;
  int inactivePages = 0;
  int speculativePages = 0;
  int pageSize = 16384;

  final headerMatch = RegExp(r'page size of (\d+) bytes').firstMatch(vmstatOutput);
  if (headerMatch != null) {
    pageSize = int.tryParse(headerMatch.group(1)!) ?? 16384;
  }

  for (final line in vmstatOutput.split('\n')) {
    if (line.contains('Pages free')) {
      final match = RegExp(r'(\d+)').firstMatch(line);
      if (match != null) freePages = int.parse(match.group(1)!);
    } else if (line.contains('Pages inactive')) {
      final match = RegExp(r'(\d+)').firstMatch(line);
      if (match != null) inactivePages = int.parse(match.group(1)!);
    } else if (line.contains('Pages speculative')) {
      final match = RegExp(r'(\d+)').firstMatch(line);
      if (match != null) speculativePages = int.parse(match.group(1)!);
    }
  }

  final totalAvailPages = freePages + inactivePages + speculativePages;
  if (totalAvailPages <= 0) {
    return (totalRamGB * 0.5).round().clamp(1, totalRamGB);
  }
  final freeGB = (totalAvailPages * pageSize) / (1024 * 1024 * 1024);
  return freeGB.round().clamp(1, totalRamGB);
}

const int _maxHardwareProbeBytes = 256 * 1024;

Map<String, String> _sanitizeProbeEnv() {
  final out = <String, String>{};
  for (final entry in Platform.environment.entries) {
    final upper = entry.key.toUpperCase();
    if (upper.startsWith('DYLD_') ||
        upper.startsWith('LD_') ||
        upper.startsWith('BASH_FUNC_') ||
        upper == 'BASH_ENV' ||
        upper == 'ENV' ||
        upper == 'PROMPT_COMMAND' ||
        upper == 'PSMODULEPATH' ||
        entry.key.contains('\x00') ||
        entry.value.contains('\x00')) {
      continue;
    }
    out[entry.key] = entry.value;
  }
  return out;
}

Future<String> _run(String cmd, List<String> args) async {
  Process? proc;
  try {
    proc = await Process.start(
      cmd,
      args,
      environment: _sanitizeProbeEnv(),
      includeParentEnvironment: false,
      runInShell: false,
    );
    final stdoutFuture = () async {
      final buf = StringBuffer();
      var chars = 0;
      await for (final chunk
          in proc!.stdout.transform(const Utf8Decoder(allowMalformed: true))) {
        if (chars < _maxHardwareProbeBytes) {
          final rem = _maxHardwareProbeBytes - chars;
          final slice = chunk.length > rem ? chunk.substring(0, rem) : chunk;
          buf.write(slice);
          chars += slice.length;
        }
      }
      return buf.toString();
    }();
    proc.stderr.drain<void>();
    final exitCode = await proc.exitCode.timeout(
      const Duration(seconds: 5),
      onTimeout: () {
        proc?.kill();
        return -1;
      },
    );
    if (exitCode != 0) return '';
    return await stdoutFuture.timeout(
      const Duration(seconds: 1),
      onTimeout: () => '',
    );
  } catch (_) {
    proc?.kill();
    return '';
  }
}
