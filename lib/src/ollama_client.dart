import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;
import 'package:logger/logger.dart';

import 'thinking_parser.dart';

// ═══════════════════════════════════════════
// Status & Models
// ═══════════════════════════════════════════

enum OllamaStatus { unknown, checking, running, notRunning, error }

class OllamaModel {
  final String name;
  final String? digest;
  final int? sizeBytes;
  final DateTime? modifiedAt;
  final String? family;
  final String? parameterSize;
  final String? quantizationLevel;

  /// Model architecture maximum context length from `/api/show` `model_info`
  /// (e.g. `qwen2.context_length`, `gemma3.context_length`).
  final int? contextLength;

  /// Explicit runtime context window configured in Modelfile `parameters`
  /// (`num_ctx <N>`). Takes precedence over [contextLength] when set.
  final int? numCtx;

  const OllamaModel({
    required this.name,
    this.digest,
    this.sizeBytes,
    this.modifiedAt,
    this.family,
    this.parameterSize,
    this.quantizationLevel,
    this.contextLength,
    this.numCtx,
  });

  /// Effective runtime context window reported by Ollama (`num_ctx` first,
  /// then architecture `context_length`), avoiding hardcoded 128k assumptions.
  int? get effectiveContextLength => numCtx ?? contextLength;

  static int? _parsePositiveContextInt(Object? raw) {
    final parsed = _parseFiniteInt(raw);
    if (parsed == null || parsed < 256 || parsed > 4 * 1024 * 1024) {
      return null;
    }
    return parsed;
  }

  static int? _extractNumCtxFromParameters(Object? rawParams) {
    if (rawParams == null) return null;
    final text = rawParams.toString();
    final match = RegExp(
      r'^\s*num_ctx\s+(\d+)\b',
      multiLine: true,
      caseSensitive: false,
    ).firstMatch(text);
    if (match == null) return null;
    return _parsePositiveContextInt(match.group(1));
  }

  static int? _extractContextLengthFromModelInfo(Object? rawModelInfo) {
    if (rawModelInfo is! Map) return null;
    for (final entry in rawModelInfo.entries) {
      final key = entry.key?.toString().toLowerCase() ?? '';
      if (key == 'context_length' || key.endsWith('.context_length')) {
        final parsed = _parsePositiveContextInt(entry.value);
        if (parsed != null) return parsed;
      }
    }
    return null;
  }

  static int? _parseFiniteInt(Object? raw) {
    if (raw == null) return null;
    if (raw is num) {
      if (!raw.isFinite) return null;
      return raw is int ? raw : raw.toInt();
    }
    if (raw is String) {
      final t = raw.trim();
      final asInt = int.tryParse(t);
      if (asInt != null) return asInt;
      final asDouble = double.tryParse(t);
      if (asDouble != null && asDouble.isFinite) return asDouble.toInt();
    }
    return null;
  }

  static String _cleanModelField(Object? raw, {int maxLen = 256}) {
    if (raw == null) return '';
    final clean =
        raw.toString().replaceAll(OllamaClient._controlOrBidiRe, '').trim();
    if (clean.length <= maxLen) return clean;
    var end = maxLen;
    final prev = clean.codeUnitAt(end - 1);
    if (prev >= 0xD800 && prev <= 0xDBFF) end -= 1;
    return end <= 0 ? '' : clean.substring(0, end);
  }

  static String? _cleanOptionalField(Object? raw, {int maxLen = 256}) {
    if (raw == null) return null;
    final clean = _cleanModelField(raw, maxLen: maxLen);
    return clean.isEmpty ? null : clean;
  }

  factory OllamaModel.fromJson(Map<Object?, Object?> json) {
    final rawDetails = json['details'];
    final details = rawDetails is Map
        ? Map<String, dynamic>.from(rawDetails)
        : const <String, dynamic>{};
    final modRaw = json['modified_at'];
    return OllamaModel(
      name: _cleanModelField(json['name'] ?? json['model'] ?? ''),
      digest: _cleanOptionalField(json['digest']),
      sizeBytes: _parseFiniteInt(json['size']),
      modifiedAt: modRaw is String ? DateTime.tryParse(modRaw) : null,
      family: _cleanOptionalField(details['family'], maxLen: 128),
      parameterSize:
          _cleanOptionalField(details['parameter_size'], maxLen: 64),
      quantizationLevel:
          _cleanOptionalField(details['quantization_level'], maxLen: 64),
      contextLength: _extractContextLengthFromModelInfo(json['model_info']) ??
          _parsePositiveContextInt(json['context_length']),
      numCtx: _parsePositiveContextInt(json['num_ctx']) ??
          _extractNumCtxFromParameters(json['parameters']),
    );
  }

  String get sizeLabel {
    if (sizeBytes == null || !sizeBytes!.isFinite || sizeBytes! < 0) return '';
    final gb = sizeBytes! / (1024 * 1024 * 1024);
    if (gb >= 1) return '${gb.toStringAsFixed(1)} GB';
    final mb = sizeBytes! / (1024 * 1024);
    return '${mb.toStringAsFixed(0)} MB';
  }

  String get displayName {
    if (name.endsWith(':latest')) {
      return name.substring(0, name.length - ':latest'.length);
    }
    return name;
  }
}

class OllamaPullProgress {
  final String status;
  final int? total;
  final int? completed;
  final String? digest;

  const OllamaPullProgress({
    required this.status,
    this.total,
    this.completed,
    this.digest,
  });

  double? get progress {
    if (total == null ||
        !total!.isFinite ||
        total! <= 0 ||
        completed == null ||
        !completed!.isFinite) {
      return null;
    }
    return (completed! / total!).clamp(0.0, 1.0);
  }

  bool get isDone => status == 'success';
  bool get isError => status.startsWith('error');
}

// ═══════════════════════════════════════════
// Client — core HTTP operations
// ═══════════════════════════════════════════

/// Low-level Ollama HTTP client. Stateless, pure Dart, no Flutter dependency.
class OllamaClient {
  static const int maxStreamLineChars = 1024 * 1024;
  static const int maxResponseBytes = 8 * 1024 * 1024;

  final String host;
  final Duration timeout;
  final http.Client? _injectedClient;

  static final Logger _logger = Logger(
    printer: PrettyPrinter(
      methodCount: 0,
      errorMethodCount: 5,
      lineLength: 80,
      colors: true,
      printEmojis: true,
      dateTimeFormat: DateTimeFormat.none,
    ),
  );

  static final RegExp _controlOrBidiRe = RegExp(
    r'[\x00-\x1F\x7F-\x9F\u200B-\u200F\u2028-\u202E\u2060-\u2064\u2066-\u2069\u061C\uFEFF]',
  );
  static final RegExp _encodedTraversalOrBidiRe = RegExp(
    r'%(?:00|2e|2f|5c|25|c2%[89][0-9a-f]|d8%9c|e2%80%(?:8[b-f]|a[8-e])|e2%81%a[0-9]|ef%bb%bf)',
    caseSensitive: false,
  );
  static final RegExp _idnDotHomoglyphRe = RegExp(r'[\u3002\uFF0E\uFF61]');
  static const Set<String> _blockedPrototypeNames = {
    '__proto__',
    'constructor',
    'prototype',
  };

  static bool _isForbiddenMetadataOrWildcardHost(String rawHost) {
    var h = rawHost.trim().toLowerCase();
    if (h.contains('%') || h.contains('..') || _idnDotHomoglyphRe.hasMatch(h)) {
      return true;
    }
    if (h.startsWith('[') && h.endsWith(']')) {
      h = h.substring(1, h.length - 1);
    }
    while (h.endsWith('.')) {
      h = h.substring(0, h.length - 1);
    }
    if (h.isEmpty ||
        h == 'null' ||
        h == '0.0.0.0' ||
        h == '::' ||
        RegExp(r'^0+(?::0+)*$').hasMatch(h) ||
        h == '169.254.169.254' ||
        h == '100.100.100.200' ||
        h == 'metadata' ||
        h == 'metadata.goog' ||
        h.endsWith('.metadata.goog') ||
        h == 'metadata.google.internal' ||
        h.endsWith('.metadata.google.internal') ||
        h == 'metadata.azure.com' ||
        h.endsWith('.metadata.azure.com') ||
        h == 'metadata.azure.internal' ||
        h == 'instance-data' ||
        h == 'fd00:ec2::254' ||
        h.endsWith('.internal') ||
        h.endsWith('.invalid') ||
        h.endsWith('.onion') ||
        h.startsWith('169.254.') ||
        h.startsWith('::ffff:169.254.') ||
        h.startsWith('::ffff:a9fe:') ||
        h.startsWith('::ffff:6464:64c8') ||
        h.startsWith('0:0:0:0:0:ffff:169.254.') ||
        h.startsWith('0:0:0:0:0:ffff:a9fe:') ||
        h.startsWith('0:0:0:0:0:ffff:6464:64c8') ||
        h.startsWith('64:ff9b:') ||
        h.startsWith('2002:') ||
        h.startsWith('2001:0:') ||
        h.startsWith('2001:0000:') ||
        h.startsWith('fd00:ec2:')) {
      return true;
    }
    if (!h.contains(':')) {
      final octets = _tryParseIpv4Octets(h);
      if (octets != null) {
        if (octets.length != 4) return true;
        final a = octets[0];
        final b = octets[1];
        final c = octets[2];
        if (a == 0 || a >= 224) return true;
        if (a == 169 && b == 254) return true;
        if (a == 100 && b == 100 && c == 100 && octets[3] == 200) {
          return true;
        }
        if (a == 192 && b == 0 && (c == 0 || c == 2)) return true;
        if (a == 198 && (b == 18 || b == 19)) return true;
        if (a == 198 && b == 51 && c == 100) return true;
        if (a == 203 && b == 0 && c == 113) return true;
      }
    }
    return false;
  }

  static List<int>? _tryParseIpv4Octets(String host) {
    if (RegExp(r'^\d+$').hasMatch(host)) {
      final v = int.tryParse(host);
      if (v != null && v >= 0 && v <= 0xFFFFFFFF) {
        return <int>[
          (v ~/ 0x1000000) & 0xFF,
          (v ~/ 0x10000) & 0xFF,
          (v ~/ 0x100) & 0xFF,
          v & 0xFF,
        ];
      }
      return const <int>[-1];
    }
    if (RegExp(r'^0x[0-9a-f]+$').hasMatch(host)) {
      final v = int.tryParse(host.substring(2), radix: 16);
      if (v != null && v >= 0 && v <= 0xFFFFFFFF) {
        return <int>[
          (v ~/ 0x1000000) & 0xFF,
          (v ~/ 0x10000) & 0xFF,
          (v ~/ 0x100) & 0xFF,
          v & 0xFF,
        ];
      }
      return const <int>[-1];
    }
    final parts = host.split('.');
    if (parts.length < 2 || parts.length > 4) return null;
    final nums = <int>[];
    for (final p in parts) {
      if (p.isEmpty) return null;
      int? parsed;
      if (p.startsWith('0x')) {
        parsed = int.tryParse(p.substring(2), radix: 16);
      } else if (p.length > 1 &&
          p.startsWith('0') &&
          RegExp(r'^[0-7]+$').hasMatch(p)) {
        parsed = int.tryParse(p, radix: 8);
      } else if (RegExp(r'^\d+$').hasMatch(p)) {
        parsed = int.tryParse(p);
      }
      if (parsed == null || parsed < 0) return null;
      nums.add(parsed);
    }
    if (nums.length == 4) {
      if (nums.any((n) => n > 255)) return const <int>[-1];
      return nums;
    }
    if (nums.length == 3) {
      if (nums[0] > 255 || nums[1] > 255 || nums[2] > 0xFFFF) {
        return const <int>[-1];
      }
      return <int>[nums[0], nums[1], (nums[2] ~/ 0x100) & 0xFF, nums[2] & 0xFF];
    }
    if (nums.length == 2) {
      if (nums[0] > 255 || nums[1] > 0xFFFFFF) return const <int>[-1];
      return <int>[
        nums[0],
        (nums[1] ~/ 0x10000) & 0xFF,
        (nums[1] ~/ 0x100) & 0xFF,
        nums[1] & 0xFF,
      ];
    }
    return null;
  }

  /// Validates an Ollama model identifier, rejecting path traversal (`..`),
  /// URL/query characters, control/BiDi characters, and excessively long names.
  static bool isValidModelName(String name) {
    if (_controlOrBidiRe.hasMatch(name)) return false;
    final trimmed = name.trim();
    if (trimmed.isEmpty ||
        trimmed.length > 256 ||
        trimmed == '.' ||
        (trimmed.startsWith('__') && trimmed.endsWith('__')) ||
        _blockedPrototypeNames.contains(trimmed.toLowerCase())) {
      return false;
    }
    if (trimmed.split('/').where((s) => s.isNotEmpty).any(
          (s) =>
              _blockedPrototypeNames.contains(s.toLowerCase()) ||
              (s.startsWith('__') && s.endsWith('__') && s.length >= 4),
        )) {
      return false;
    }
    if (trimmed.contains('..') ||
        trimmed.contains(r'\') ||
        trimmed.contains('?') ||
        trimmed.contains('#') ||
        trimmed.contains('@') ||
        trimmed.contains('%')) {
      return false;
    }
    if (trimmed.startsWith('/') || trimmed.startsWith('-')) return false;
    return true;
  }

  /// Normalizes and validates an HTTP(S) Ollama endpoint URL, stripping any
  /// embedded credentials (`user:pass@`), blocking cloud metadata IMDS /
  /// `0.0.0.0` SSRF targets, and stripping trailing slashes.
  static String normalizeEndpointUrl(
    String raw, {
    String defaultUrl = 'http://localhost:11434',
  }) {
    if (_controlOrBidiRe.hasMatch(raw) || _idnDotHomoglyphRe.hasMatch(raw)) {
      return defaultUrl;
    }
    final trimmed = raw.trim();
    if (trimmed.isEmpty || trimmed.contains(r'\')) {
      return defaultUrl;
    }
    final lowerTrimmed = trimmed.toLowerCase();
    if (_encodedTraversalOrBidiRe.hasMatch(lowerTrimmed) ||
        RegExp(r'(?:^|/)(\.\.)(?:/|$)').hasMatch(trimmed)) {
      return defaultUrl;
    }
    final uri = Uri.tryParse(trimmed);
    if (uri == null ||
        !uri.hasAuthority ||
        uri.host.isEmpty ||
        (uri.scheme != 'http' && uri.scheme != 'https') ||
        uri.pathSegments.contains('..') ||
        _isForbiddenMetadataOrWildcardHost(uri.host) ||
        (uri.hasPort && (uri.port < 1 || uri.port > 65535))) {
      return defaultUrl;
    }
    var cleanPath = uri.path;
    while (cleanPath.endsWith('/') && cleanPath.length > 1) {
      cleanPath = cleanPath.substring(0, cleanPath.length - 1);
    }
    if (cleanPath == '/') cleanPath = '';
    final normalized = Uri(
      scheme: uri.scheme,
      host: uri.host,
      port: uri.hasPort ? uri.port : null,
      path: cleanPath,
    ).toString();
    return normalized.endsWith('/')
        ? normalized.substring(0, normalized.length - 1)
        : normalized;
  }

  OllamaClient({
    String host = 'http://localhost:11434',
    this.timeout = const Duration(seconds: 5),
    http.Client? httpClient,
  }) : host = normalizeEndpointUrl(host),
       _injectedClient = httpClient;

  Future<T> _withClient<T>(Future<T> Function(http.Client client) fn) async {
    final injected = _injectedClient;
    if (injected != null) return fn(injected);
    final transient = http.Client();
    try {
      return await fn(transient);
    } finally {
      transient.close();
    }
  }

  Future<http.Response> _sendBoundedRequest(
    String method,
    Uri uri, {
    Map<String, String>? headers,
    String? body,
    Duration? requestTimeout,
  }) {
    final effectiveTimeout = requestTimeout ?? timeout;
    return _withClient((client) async {
      final req = http.Request(method, uri)..followRedirects = false;
      if (headers != null) req.headers.addAll(headers);
      if (body != null) req.body = body;
      final streamed = await client.send(req).timeout(effectiveTimeout);
      if (streamed.statusCode >= 300 && streamed.statusCode < 400) {
        await streamed.stream.drain<void>().catchError((_) {});
        return http.Response.bytes(
          const <int>[],
          streamed.statusCode,
          headers: streamed.headers,
          request: req,
          reasonPhrase: streamed.reasonPhrase,
        );
      }
      final declaredLen = streamed.contentLength;
      if (declaredLen != null &&
          (declaredLen < 0 || declaredLen > maxResponseBytes)) {
        await streamed.stream.drain<void>().catchError((_) {});
        throw const FormatException('Ollama response exceeded maxResponseBytes');
      }
      final builder = BytesBuilder(copy: false);
      await for (final chunk in streamed.stream.timeout(effectiveTimeout)) {
        if (builder.length + chunk.length > maxResponseBytes) {
          throw const FormatException(
            'Ollama response exceeded maxResponseBytes',
          );
        }
        builder.add(chunk);
      }
      return http.Response.bytes(
        builder.takeBytes(),
        streamed.statusCode,
        headers: streamed.headers,
        request: req,
        reasonPhrase: streamed.reasonPhrase,
      );
    });
  }

  /// Check if Ollama server is reachable.
  Future<OllamaStatus> checkStatus() async {
    try {
      final resp = await _sendBoundedRequest('GET', Uri.parse(host));
      if (resp.statusCode == 200) {
        _logger.d('[OllamaClient] checkStatus: Server is running at $host');
        return OllamaStatus.running;
      }
      _logger.w(
        '[OllamaClient] checkStatus: Server returned code ${resp.statusCode} at $host',
      );
      return OllamaStatus.notRunning;
    } catch (e) {
      _logger.w(
        '[OllamaClient] checkStatus: Server unreachable at $host. Error: $e',
      );
      return OllamaStatus.notRunning;
    }
  }

  /// Whether Ollama is running (convenience).
  Future<bool> get isRunning async =>
      await checkStatus() == OllamaStatus.running;

  /// List all locally available models.
  Future<List<OllamaModel>> listModels() async {
    try {
      _logger.d('[OllamaClient] Listing local models from host: $host');
      final resp = await _sendBoundedRequest('GET', Uri.parse('$host/api/tags'));
      if (resp.statusCode != 200 || resp.bodyBytes.length > maxResponseBytes) {
        _logger.w(
          '[OllamaClient] listModels: Server returned code ${resp.statusCode}',
        );
        return [];
      }

      final body = jsonDecode(resp.body);
      if (body is! Map) return [];
      final models = body['models'] as List? ?? const [];
      _logger.d(
        '[OllamaClient] listModels: Found ${models.length} models locally',
      );
      return models
          .whereType<Map>()
          .map(OllamaModel.fromJson)
          .where((m) => m.name.isNotEmpty)
          .toList()
        ..sort((a, b) => a.name.compareTo(b.name));
    } catch (e) {
      _logger.e(
        '[OllamaClient] listModels: Failed to query local models. Error: $e',
      );
      return [];
    }
  }

  /// Check if a specific model is available locally.
  Future<bool> hasModel(String name) async {
    final list = await listModels();
    return list.any((m) => m.name == name || m.displayName == name);
  }

  /// Get detailed info about a model.
  Future<OllamaModel?> showModel(String name) async {
    if (!isValidModelName(name)) return null;
    try {
      _logger.d('[OllamaClient] showModel: Querying metadata for $name');
      final resp = await _sendBoundedRequest(
        'POST',
        Uri.parse('$host/api/show'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'name': name}),
      );
      if (resp.statusCode != 200 || resp.bodyBytes.length > maxResponseBytes) {
        _logger.w(
          '[OllamaClient] showModel ($name): Server returned code ${resp.statusCode}',
        );
        return null;
      }

      final body = jsonDecode(resp.body);
      if (body is! Map) return null;
      return OllamaModel.fromJson({'name': name, ...body});
    } catch (e) {
      _logger.e(
        '[OllamaClient] showModel ($name): Failed to fetch info. Error: $e',
      );
      return null;
    }
  }

  /// Resolves the active context window for [model] from Ollama `/api/show`
  /// (`num_ctx` first, then architecture `context_length`), falling back to
  /// [fallback] when unavailable instead of assuming a 128k default.
  Future<int> resolveContextWindow(
    String model, {
    int fallback = 8192,
  }) async {
    final safeFallback = fallback.clamp(512, 2097152);
    final info = await showModel(model);
    final resolved = info?.effectiveContextLength;
    if (resolved == null) return safeFallback;
    return resolved.clamp(512, 2097152);
  }

  /// Pull (download) a model. Streams progress updates with safe NDJSON
  /// line reassembly across TCP chunk boundaries and guaranteed client cleanup.
  Stream<OllamaPullProgress> pullModel(String name) async* {
    if (!isValidModelName(name)) {
      yield const OllamaPullProgress(status: 'error: invalid model name');
      return;
    }
    http.Client? ownedClient;
    try {
      _logger.i('[OllamaClient] Initiating model download: $name');
      final request = http.Request('POST', Uri.parse('$host/api/pull'))
        ..followRedirects = false;
      request.headers['Content-Type'] = 'application/json';
      request.body = jsonEncode({'name': name, 'stream': true});

      final client = _injectedClient ?? (ownedClient = http.Client());
      final response = await client.send(request).timeout(
        const Duration(seconds: 60),
      );
      if (response.statusCode != 200) {
        yield OllamaPullProgress(status: 'error: HTTP ${response.statusCode}');
        return;
      }

      var lineBuf = '';
      OllamaPullProgress? parseLine(String rawLine) {
        final trimmed = rawLine.trim();
        if (trimmed.isEmpty) return null;
        try {
          final json = jsonDecode(trimmed);
          if (json is! Map) return null;
          return OllamaPullProgress(
            status: json['status']?.toString() ?? '',
            total: OllamaModel._parseFiniteInt(json['total']),
            completed: OllamaModel._parseFiniteInt(json['completed']),
            digest: json['digest']?.toString(),
          );
        } catch (_) {
          return null;
        }
      }

      await for (final chunk in response.stream.transform(
        const Utf8Decoder(allowMalformed: true),
      )) {
        lineBuf += chunk;
        if (lineBuf.length > maxStreamLineChars && !lineBuf.contains('\n')) {
          lineBuf = '';
          yield const OllamaPullProgress(
            status: 'error: stream line exceeded cap',
          );
          return;
        }
        while (true) {
          final nl = lineBuf.indexOf('\n');
          if (nl < 0) break;
          final line = lineBuf.substring(0, nl);
          lineBuf = lineBuf.substring(nl + 1);
          final progress = parseLine(line);
          if (progress != null) yield progress;
        }
      }
      final trailing = parseLine(lineBuf);
      if (trailing != null) yield trailing;
    } catch (e) {
      _logger.e('[OllamaClient] Error pulling model $name. Detail: $e');
      yield OllamaPullProgress(status: 'error: $e');
    } finally {
      ownedClient?.close();
    }
  }

  /// Create a model from a Modelfile string.
  Future<bool> createModel(String name, String modelfile) async {
    if (!isValidModelName(name) || modelfile.length > maxResponseBytes) {
      return false;
    }
    try {
      final resp = await _sendBoundedRequest(
        'POST',
        Uri.parse('$host/api/create'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'name': name,
          'modelfile': modelfile,
          'stream': false,
        }),
        requestTimeout: const Duration(seconds: 120),
      );
      return resp.statusCode == 200;
    } catch (e) {
      _logger.e(
        '[OllamaClient] Exception during model creation for "$name". Error: $e',
      );
      return false;
    }
  }

  /// Delete a local model.
  Future<bool> deleteModel(String name) async {
    if (!isValidModelName(name)) return false;
    try {
      final resp = await _sendBoundedRequest(
        'DELETE',
        Uri.parse('$host/api/delete'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'name': name}),
      );
      return resp.statusCode == 200;
    } catch (e) {
      _logger.e(
        '[OllamaClient] Exception during model deletion for "$name". Error: $e',
      );
      return false;
    }
  }

  /// Single-turn chat returning the decoded Ollama JSON map (including timing
  /// metrics for [ModelOptimizer]) with `followRedirects = false` and bounded
  /// response reading.
  Future<Map<String, dynamic>> chatRaw(
    String model,
    String prompt, {
    String? system,
    bool enableThinking = true,
  }) async {
    if (!isValidModelName(model)) {
      throw ArgumentError.value(model, 'model', 'Invalid Ollama model name');
    }
    final messages = <Map<String, String>>[];
    if (system != null) messages.add({'role': 'system', 'content': system});
    messages.add({'role': 'user', 'content': prompt});

    final sw = Stopwatch()..start();
    try {
      final resp = await _sendBoundedRequest(
        'POST',
        Uri.parse('$host/api/chat'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'model': model,
          'messages': messages,
          'stream': false,
          if (!enableThinking) 'think': false,
        }),
        requestTimeout: const Duration(seconds: 60),
      );

      sw.stop();

      if (resp.statusCode != 200) {
        throw Exception('Ollama error: ${resp.statusCode}');
      }
      if (resp.bodyBytes.length > maxResponseBytes) {
        throw const FormatException('Ollama response exceeded maxResponseBytes');
      }

      final body = jsonDecode(resp.body);
      if (body is! Map) {
        throw const FormatException('Invalid Ollama JSON response');
      }
      return Map<String, dynamic>.from(body);
    } catch (e) {
      sw.stop();
      _logger.e(
        '[OllamaClient] Chat request failed after ${sw.elapsedMilliseconds}ms. Error: $e',
      );
      rethrow;
    }
  }

  /// Single-turn chat (native Ollama API). Returns response text with any
  /// `<think>…</think>` reasoning block stripped (user-visible content only).
  Future<String> chat(String model, String prompt, {String? system}) async {
    final result = await chatWithThinking(model, prompt, system: system);
    return result.content;
  }

  /// Single-turn chat that also exposes the reasoning trace.
  Future<OllamaChatResult> chatWithThinking(
    String model,
    String prompt, {
    String? system,
    bool enableThinking = true,
  }) async {
    final body = await chatRaw(
      model,
      prompt,
      system: system,
      enableThinking: enableThinking,
    );
    final rawMessage = body['message'];
    final message = rawMessage is Map
        ? Map<String, dynamic>.from(rawMessage)
        : const <String, dynamic>{};
    final rawContent = message['content']?.toString() ?? '';
    final serverThinking = message['thinking']?.toString() ?? '';

    if (serverThinking.isNotEmpty) {
      return OllamaChatResult(
        thinking: serverThinking,
        content: ThinkingParser.stripThinking(rawContent),
      );
    }
    final split = ThinkingParser.split(rawContent);
    return OllamaChatResult(thinking: split.thinking, content: split.content);
  }

  /// Streaming chat (native Ollama API). Yields text chunks with NDJSON line
  /// buffering across TCP frames and deterministic client disposal.
  Stream<String> chatStream(
    String model,
    String prompt, {
    String? system,
  }) async* {
    if (!isValidModelName(model)) {
      throw ArgumentError.value(model, 'model', 'Invalid Ollama model name');
    }
    final messages = <Map<String, String>>[];
    if (system != null) messages.add({'role': 'system', 'content': system});
    messages.add({'role': 'user', 'content': prompt});

    http.Client? ownedClient;
    try {
      final request = http.Request('POST', Uri.parse('$host/api/chat'))
        ..followRedirects = false;
      request.headers['Content-Type'] = 'application/json';
      request.body = jsonEncode({
        'model': model,
        'messages': messages,
        'stream': true,
      });

      final client = _injectedClient ?? (ownedClient = http.Client());
      final response = await client.send(request).timeout(
        const Duration(seconds: 60),
      );
      if (response.statusCode != 200) {
        throw Exception('Ollama stream error: ${response.statusCode}');
      }

      String? extractChunk(String rawLine) {
        final trimmed = rawLine.trim();
        if (trimmed.isEmpty) return null;
        try {
          final json = jsonDecode(trimmed);
          if (json is! Map) return null;
          final msg = json['message'];
          if (msg is! Map) return null;
          final content = msg['content']?.toString();
          return (content != null && content.isNotEmpty) ? content : null;
        } catch (_) {
          return null;
        }
      }

      var lineBuf = '';
      await for (final chunk in response.stream
          .timeout(const Duration(seconds: 60))
          .transform(const Utf8Decoder(allowMalformed: true))) {
        lineBuf += chunk;
        if (lineBuf.length > maxStreamLineChars && !lineBuf.contains('\n')) {
          throw const FormatException('Ollama stream line exceeded cap');
        }
        while (true) {
          final nl = lineBuf.indexOf('\n');
          if (nl < 0) break;
          final line = lineBuf.substring(0, nl);
          lineBuf = lineBuf.substring(nl + 1);
          final text = extractChunk(line);
          if (text != null) yield text;
        }
      }
      final tail = extractChunk(lineBuf);
      if (tail != null) yield tail;
    } finally {
      ownedClient?.close();
    }
  }

  /// Load a model into RAM (keep-alive indefinitely).
  Future<bool> loadModel(String name) async {
    if (!isValidModelName(name)) return false;
    try {
      final resp = await _sendBoundedRequest(
        'POST',
        Uri.parse('$host/api/generate'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'model': name,
          'prompt': '',
          'keep_alive': -1,
        }),
        requestTimeout: const Duration(seconds: 120),
      );
      return resp.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  /// Unload a model from RAM.
  Future<bool> unloadModel(String name) async {
    if (!isValidModelName(name)) return false;
    try {
      final resp = await _sendBoundedRequest(
        'POST',
        Uri.parse('$host/api/generate'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'model': name, 'keep_alive': 0}),
        requestTimeout: const Duration(seconds: 10),
      );
      return resp.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  /// List models currently loaded in memory/RAM.
  Future<List<String>> listRunningModels() async {
    try {
      final resp = await _sendBoundedRequest('GET', Uri.parse('$host/api/ps'));
      if (resp.statusCode != 200 || resp.bodyBytes.length > maxResponseBytes) {
        return [];
      }
      final body = jsonDecode(resp.body);
      if (body is! Map) return [];
      final models = body['models'] as List? ?? const [];
      return models
          .whereType<Map>()
          .map((m) => m['name']?.toString() ?? '')
          .where((n) => n.isNotEmpty)
          .toList();
    } catch (_) {
      return [];
    }
  }

  /// OpenAI-compatible base URL for providers that use /v1/chat/completions.
  String get openAiBaseUrl => '$host/v1';
}

/// Result of [OllamaClient.chatWithThinking].
class OllamaChatResult {
  final String thinking;
  final String content;

  const OllamaChatResult({required this.thinking, required this.content});

  bool get hasThinking => thinking.isNotEmpty;

  @override
  String toString() =>
      'OllamaChatResult(thinking: ${thinking.length} chars, content: ${content.length} chars)';
}
