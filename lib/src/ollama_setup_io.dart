import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'ollama_client.dart';

const Set<String> _blockedOllamaEnvKeys = {
  'LD_PRELOAD',
  'LD_LIBRARY_PATH',
  'LD_AUDIT',
  'DYLD_INSERT_LIBRARIES',
  'DYLD_LIBRARY_PATH',
  'DYLD_FRAMEWORK_PATH',
  'BASH_ENV',
  'ENV',
  'ZDOTDIR',
  'SHELLOPTS',
  'BASHOPTS',
  'CDPATH',
  'GLOBIGNORE',
  'IFS',
  'PS4',
  'PROMPT_COMMAND',
  'NODE_OPTIONS',
  'PYTHONPATH',
  'PYTHONSTARTUP',
  'RUBYOPT',
  'PERL5OPT',
  'JAVA_TOOL_OPTIONS',
  '_JAVA_OPTIONS',
  'DOTNET_STARTUP_HOOKS',
  'OPENSSL_CONF',
  'OPENSSL_ENGINES',
  'OLLAMA_ORIGINS',
  'GIT_SSH_COMMAND',
  'ELECTRON_RUN_AS_NODE',
};

bool _isUnsafeOllamaModelsDir(String raw) {
  final trimmed = raw.trim();
  if (trimmed.isEmpty ||
      trimmed.startsWith('-') ||
      trimmed.startsWith('//') ||
      trimmed.startsWith(r'\\') ||
      trimmed.startsWith(r'\??\')) {
    return true;
  }
  if (RegExp(r'[\x00-\x1F\x7F-\x9F\u202A-\u202E\u2066-\u2069\u061C]')
      .hasMatch(raw)) {
    return true;
  }
  final isWinDrive = RegExp(r'^[A-Za-z]:[\\/]').hasMatch(trimmed);
  final afterDrive = isWinDrive ? trimmed.substring(2) : trimmed;
  if (afterDrive.contains(':')) return true;
  final normalized = trimmed.replaceAll(r'\', '/');
  if (normalized.split('/').any((s) => s == '..')) return true;
  final lower = normalized.toLowerCase();
  const blockedDirs = [
    '/.ssh',
    '/.gnupg',
    '/.aws',
    '/.kube',
    '/etc',
    '/private/etc',
  ];
  for (final marker in blockedDirs) {
    if (lower == marker ||
        lower.endsWith(marker) ||
        lower.contains('$marker/')) {
      return true;
    }
  }
  return false;
}

/// Builds a scrubbed environment for launching or probing `ollama`, ensuring
/// `OLLAMA_HOST` cannot be hijacked via parent environment to bind `0.0.0.0`
/// or cloud IMDS targets, and `OLLAMA_MODELS` cannot point to UNC shares.
Map<String, String> sanitizeOllamaSubprocessEnv({
  Map<String, String>? baseEnv,
  String? clientHost,
}) {
  final source = baseEnv ?? Platform.environment;
  final out = <String, String>{};
  for (final entry in source.entries) {
    final key = entry.key;
    if (key.isEmpty || key.contains('=')) continue;
    final upper = key.toUpperCase();
    if (_blockedOllamaEnvKeys.contains(upper) ||
        upper.startsWith('DYLD_') ||
        upper.startsWith('LD_') ||
        upper.startsWith('BASH_FUNC_') ||
        key.contains('\x00') ||
        entry.value.contains('\x00')) {
      continue;
    }
    out[key] = entry.value;
  }

  final modelsDir = out['OLLAMA_MODELS'];
  if (modelsDir != null && _isUnsafeOllamaModelsDir(modelsDir)) {
    out.remove('OLLAMA_MODELS');
  }

  // Force loopback bind if OLLAMA_HOST is missing, wildcard, or IMDS/invalid.
  final existingHost = out['OLLAMA_HOST']?.trim() ?? '';
  final candidateUrl = existingHost.contains('://')
      ? existingHost
      : 'http://$existingHost';
  final normalizedExisting = existingHost.isEmpty
      ? ''
      : OllamaClient.normalizeEndpointUrl(candidateUrl, defaultUrl: '');
  if (normalizedExisting.isEmpty ||
      existingHost.startsWith('0.0.0.0') ||
      existingHost.contains('://0.0.0.0') ||
      existingHost.startsWith('[::]') ||
      existingHost.startsWith('::')) {
    final parsed = clientHost != null ? Uri.tryParse(clientHost) : null;
    final port = (parsed != null &&
            parsed.hasPort &&
            parsed.port >= 1 &&
            parsed.port <= 65535)
        ? parsed.port
        : 11434;
    out['OLLAMA_HOST'] = '127.0.0.1:$port';
  }
  return out;
}

/// Validates that [candidatePath] exists (following symlinks such as Homebrew
/// `/opt/homebrew/bin/ollama` or `/usr/local/bin/ollama`), resolves to a
/// regular executable file, and returns its canonical resolved path.
String? validateOllamaExecutablePath(String candidatePath) {
  final trimmed = candidatePath.trim();
  if (trimmed.isEmpty ||
      trimmed.contains('\x00') ||
      trimmed.startsWith('//') ||
      trimmed.startsWith(r'\\')) {
    return null;
  }

  try {
    final followedType = FileSystemEntity.typeSync(trimmed, followLinks: true);
    if (followedType != FileSystemEntityType.file) {
      return null;
    }

    final resolvedPath = File(trimmed).resolveSymbolicLinksSync();
    if (FileSystemEntity.typeSync(resolvedPath, followLinks: false) !=
        FileSystemEntityType.file) {
      return null;
    }

    final stat = File(resolvedPath).statSync();
    if (stat.size <= 0) return null;

    if (!Platform.isWindows) {
      // Require at least one POSIX execute bit (0111 octal = 0x49).
      if ((stat.mode & 0x49) == 0) {
        return null;
      }
    }
    return resolvedPath;
  } catch (_) {
    return null;
  }
}

/// Resolves the canonical `ollama` executable path via `which`/`where` or
/// standard platform installation locations (including Homebrew symlinks).
Future<String?> resolveOllamaExecutable() async {
  try {
    final locator = Platform.isWindows ? 'where' : 'which';
    final proc = await Process.start(
      locator,
      const ['ollama'],
      environment: sanitizeOllamaSubprocessEnv(),
      includeParentEnvironment: false,
      runInShell: false,
    );
    final stdoutFuture = proc.stdout
        .transform(const Utf8Decoder(allowMalformed: true))
        .join();
    proc.stderr.drain<void>();
    final exitCode = await proc.exitCode.timeout(
      const Duration(seconds: 5),
      onTimeout: () {
        proc.kill();
        return -1;
      },
    );
    if (exitCode == 0) {
      final rawOut = await stdoutFuture.timeout(
        const Duration(seconds: 1),
        onTimeout: () => '',
      );
      for (final line in rawOut.split(RegExp(r'[\r\n]+'))) {
        final resolved = validateOllamaExecutablePath(line);
        if (resolved != null) return resolved;
      }
    }
  } catch (_) {
    // Fall back to known installation paths below.
  }

  try {
    final localAppData = Platform.environment['LOCALAPPDATA'] ?? '';
    final userProfile = Platform.environment['USERPROFILE'] ?? '';
    final userName = Platform.environment['USERNAME'] ?? '';

    final paths = <String>[
      if (Platform.isMacOS) ...[
        '/opt/homebrew/bin/ollama',
        '/usr/local/bin/ollama',
        '/Applications/Ollama.app/Contents/Resources/ollama',
      ] else if (Platform.isWindows) ...[
        if (localAppData.isNotEmpty && !localAppData.startsWith(r'\\'))
          '$localAppData\\Programs\\Ollama\\ollama.exe',
        if (userProfile.isNotEmpty && !userProfile.startsWith(r'\\'))
          '$userProfile\\AppData\\Local\\Programs\\Ollama\\ollama.exe',
        if (userName.isNotEmpty &&
            !userName.contains(r'\') &&
            !userName.contains('/'))
          'C:\\Users\\$userName\\AppData\\Local\\Programs\\Ollama\\ollama.exe',
      ] else ...[
        '/usr/local/bin/ollama',
        '/usr/bin/ollama',
      ],
    ];

    for (final path in paths) {
      final resolved = validateOllamaExecutablePath(path);
      if (resolved != null) return resolved;
    }
  } catch (_) {}

  return null;
}

Future<bool> isOllamaInstalled() async {
  return (await resolveOllamaExecutable()) != null;
}

Future<bool> startOllamaServer(OllamaClient client) async {
  try {
    final executable = await resolveOllamaExecutable();
    if (executable == null) return false;

    await Process.start(
      executable,
      const ['serve'],
      environment: sanitizeOllamaSubprocessEnv(clientHost: client.host),
      includeParentEnvironment: false,
      runInShell: false,
      mode: ProcessStartMode.detached,
    );

    for (var i = 0; i < 20; i++) {
      await Future<void>.delayed(const Duration(milliseconds: 500));
      if (await client.isRunning) return true;
    }
    return false;
  } catch (_) {
    return false;
  }
}

String get ollamaDownloadUrl {
  if (Platform.isMacOS) return 'https://ollama.com/download/mac';
  if (Platform.isWindows) return 'https://ollama.com/download/windows';
  return 'https://ollama.com/download/linux';
}
