import 'dart:convert';

/// A tool-call recovered from plain-text model output.
///
/// Returned by [PlainTextToolCallParser.parse]. The block keeps both the
/// structured payload (`name` + `arguments`) and the raw substring it came
/// from, so callers can either send it down a normal tool-execution path or
/// strip it from the user-visible content with [PlainTextToolCallParser.strip].
class PlainTextToolCallBlock {
  /// Tool name, exactly as the model emitted it.
  final String name;

  /// Arguments object. Always a `Map<String, Object?>` (never a list / scalar).
  final Map<String, Object?> arguments;

  /// Inclusive start index in the original text.
  final int start;

  /// Exclusive end index in the original text.
  final int end;

  /// The raw substring `text.substring(start, end)` — useful for diagnostics.
  final String raw;

  /// Which detector matched this block (debug aid).
  final PlainTextToolCallFormat format;

  const PlainTextToolCallBlock({
    required this.name,
    required this.arguments,
    required this.start,
    required this.end,
    required this.raw,
    required this.format,
  });

  @override
  String toString() =>
      'PlainTextToolCallBlock(name: $name, format: ${format.name}, args: ${arguments.length} keys, '
      'span: $start..$end)';
}

/// Which embedded format produced a [PlainTextToolCallBlock].
enum PlainTextToolCallFormat {
  /// `[toolName]\n{...}\n[END_TOOL_REQUEST]` or `[toolName]\n{...}\n[/toolName]`
  /// (LMStudio / Mistral / some Llama fine-tunes).
  bracket,

  /// `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
  /// (Hermes-2-Pro, Qwen2.5, NousResearch chat models).
  toolCallTag,

  /// `<function_call>{"name": "...", "arguments": {...}}</function_call>`
  /// (some Functionary / OpenAI-mimic fine-tunes).
  functionCallTag,

  /// ```` ```json\n{"name": "...", "arguments": {...}}\n``` ````
  /// or ```` ```tool_call\n{...}\n``` ```` (fenced-block convention).
  fencedJson,

  /// Bare `{"name": "...", "arguments": {...}}` taking up the entire trailing
  /// line(s) with no surrounding markup.
  bareJson,
}

/// Parses tool calls embedded as text inside model responses.
///
/// Many local Ollama / LM Studio / llama.cpp models (Llama 3.2, Phi-3, Qwen
/// 7B, Hermes-2-Pro, DeepSeek-R1, Mistral 7B Instruct …) do not always populate
/// the structured `tool_calls` field of the chat response. They invoke tools
/// by writing text in the assistant message. The Ollama server passes that
/// text through verbatim, leaving the host to recover the call.
///
/// This parser is **read-only**: it never executes anything. Callers decide
/// whether to dispatch the recovered calls.
///
/// ```dart
/// final blocks = PlainTextToolCallParser.parse(response.content);
/// if (blocks.isEmpty) {
///   // No embedded calls — show the response as is.
///   showText(response.content);
/// } else {
///   // Run each tool, then strip the tags from the user-visible text.
///   for (final block in blocks) {
///     await runTool(block.name, block.arguments);
///   }
///   showText(PlainTextToolCallParser.strip(response.content, blocks));
/// }
/// ```
class PlainTextToolCallParser {
  PlainTextToolCallParser._();

  /// Maximum bytes we will scan inside a single JSON payload before giving up.
  /// Prevents pathological input from blowing up memory or time. 256 KB
  /// matches OpenClaw's upstream budget.
  static const int defaultMaxPayloadBytes = 256 * 1024;

  /// Scans [text] for tool calls in any of the supported formats.
  ///
  /// * [allowedToolNames] — when provided, blocks whose `name` is not in this
  ///   set are silently dropped (safer when the host has a tool allowlist).
  /// * [maxPayloadBytes] — per-block JSON cap (default 256 KB).
  ///
  /// Returns blocks in document order; never throws on malformed input — bad
  /// matches are simply not returned.
  static List<PlainTextToolCallBlock> parse(
    String text, {
    Set<String>? allowedToolNames,
    int maxPayloadBytes = defaultMaxPayloadBytes,
  }) {
    if (text.isEmpty) return const [];

    final blocks = <PlainTextToolCallBlock>[];
    blocks.addAll(_parseBracketBlocks(text, maxPayloadBytes));
    blocks.addAll(
      _parseTagBlocks(text, 'tool_call', PlainTextToolCallFormat.toolCallTag),
    );
    blocks.addAll(
      _parseTagBlocks(
        text,
        'function_call',
        PlainTextToolCallFormat.functionCallTag,
      ),
    );
    blocks.addAll(_parseFencedBlocks(text));

    // Bare-JSON detection runs only when nothing else matched. Otherwise we
    // would double-count the inner JSON of a tag- or fence-wrapped block.
    if (blocks.isEmpty) {
      final bare = _parseBareJson(text);
      if (bare != null) blocks.add(bare);
    }

    // Document order, then dedupe overlapping ranges (paranoia).
    blocks.sort((a, b) => a.start.compareTo(b.start));
    final filtered = <PlainTextToolCallBlock>[];
    for (final block in blocks) {
      if (filtered.isNotEmpty && block.start < filtered.last.end) continue;
      if (allowedToolNames != null && !allowedToolNames.contains(block.name)) {
        continue;
      }
      filtered.add(block);
    }
    return filtered;
  }

  /// Returns [text] with the spans of every block in [blocks] removed.
  /// The blocks must come from [parse] on the same string.
  ///
  /// The output is trimmed of the trailing/leading whitespace that often
  /// surrounds removed regions, so chat UIs can render it directly without
  /// stray blank lines.
  static String strip(String text, List<PlainTextToolCallBlock> blocks) {
    if (blocks.isEmpty) return text;
    final sorted = [...blocks]..sort((a, b) => a.start.compareTo(b.start));
    final buf = StringBuffer();
    var cursor = 0;
    for (final block in sorted) {
      if (block.start < cursor) continue; // Should not happen post-dedupe.
      buf.write(text.substring(cursor, block.start));
      cursor = block.end;
    }
    buf.write(text.substring(cursor));
    return buf.toString().trim();
  }

  /// Convenience: returns `true` if [text] contains at least one detectable
  /// tool-call block in any supported format.
  static bool hasToolCalls(String text) => parse(text).isNotEmpty;

  // ─────────────────────────────────────────────────────────
  //  Bracket form: [name]\n{...}\n[END_TOOL_REQUEST] | [/name]
  // ─────────────────────────────────────────────────────────

  static const String _endToolRequest = '[END_TOOL_REQUEST]';

  static List<PlainTextToolCallBlock> _parseBracketBlocks(
    String text,
    int maxPayloadBytes,
  ) {
    final blocks = <PlainTextToolCallBlock>[];
    final pattern = RegExp(r'\[([A-Za-z0-9_\-]+)\]\s*\n', multiLine: true);
    for (final match in pattern.allMatches(text)) {
      final name = match.group(1)!;
      final jsonStart = match.end;
      final jsonResult = _consumeJsonObject(text, jsonStart, maxPayloadBytes);
      if (jsonResult == null) continue;
      final closingIndex = _findBracketClosing(text, jsonResult.end, name);
      if (closingIndex == null) continue;
      blocks.add(
        PlainTextToolCallBlock(
          name: name,
          arguments: jsonResult.value,
          start: match.start,
          end: closingIndex,
          raw: text.substring(match.start, closingIndex),
          format: PlainTextToolCallFormat.bracket,
        ),
      );
    }
    return blocks;
  }

  static int? _findBracketClosing(String text, int start, String name) {
    final cursor = _skipWhitespace(text, start);
    if (text.startsWith(_endToolRequest, cursor)) {
      return cursor + _endToolRequest.length;
    }
    final named = '[/$name]';
    if (text.startsWith(named, cursor)) {
      return cursor + named.length;
    }
    return null;
  }

  // ─────────────────────────────────────────────────────────
  //  Tag forms: <tool_call>{...}</tool_call>, <function_call>…
  // ─────────────────────────────────────────────────────────

  static List<PlainTextToolCallBlock> _parseTagBlocks(
    String text,
    String tag,
    PlainTextToolCallFormat format,
  ) {
    final blocks = <PlainTextToolCallBlock>[];
    final escaped = RegExp.escape(tag);
    final pattern = RegExp(
      '<$escaped>([\\s\\S]*?)</$escaped>',
      caseSensitive: false,
    );
    for (final match in pattern.allMatches(text)) {
      final inner = match.group(1)!.trim();
      final parsed = _parseInlineToolCallJson(inner);
      if (parsed == null) continue;
      blocks.add(
        PlainTextToolCallBlock(
          name: parsed.name,
          arguments: parsed.arguments,
          start: match.start,
          end: match.end,
          raw: match.group(0)!,
          format: format,
        ),
      );
    }
    return blocks;
  }

  // ─────────────────────────────────────────────────────────
  //  Fenced form: ```json {...} ```  or  ```tool_call {...} ```
  // ─────────────────────────────────────────────────────────

  static final RegExp _fencePattern = RegExp(
    r'```(?:json|tool_call|tool|function_call)\s*\n([\s\S]*?)\n\s*```',
    caseSensitive: false,
  );

  static List<PlainTextToolCallBlock> _parseFencedBlocks(String text) {
    final blocks = <PlainTextToolCallBlock>[];
    for (final match in _fencePattern.allMatches(text)) {
      final inner = match.group(1)!.trim();
      final parsed = _parseInlineToolCallJson(inner);
      if (parsed == null) continue;
      blocks.add(
        PlainTextToolCallBlock(
          name: parsed.name,
          arguments: parsed.arguments,
          start: match.start,
          end: match.end,
          raw: match.group(0)!,
          format: PlainTextToolCallFormat.fencedJson,
        ),
      );
    }
    return blocks;
  }

  // ─────────────────────────────────────────────────────────
  //  Bare-JSON last resort: whole-message JSON object with name+arguments.
  // ─────────────────────────────────────────────────────────

  static PlainTextToolCallBlock? _parseBareJson(String text) {
    final trimmed = text.trim();
    if (!trimmed.startsWith('{') || !trimmed.endsWith('}')) return null;
    final parsed = _parseInlineToolCallJson(trimmed);
    if (parsed == null) return null;
    final start = text.indexOf(trimmed);
    return PlainTextToolCallBlock(
      name: parsed.name,
      arguments: parsed.arguments,
      start: start,
      end: start + trimmed.length,
      raw: trimmed,
      format: PlainTextToolCallFormat.bareJson,
    );
  }

  // ─────────────────────────────────────────────────────────
  //  Shared helpers.
  // ─────────────────────────────────────────────────────────

  /// Parses `{"name": "X", "arguments": {...}}` (the canonical OpenAI shape
  /// emitted by Hermes / Qwen / Functionary models).
  ///
  /// Tolerates `parameters` as an alias for `arguments`. Returns `null` if
  /// the input is not valid JSON, missing a `name`, or has a non-object
  /// argument payload.
  static _ParsedCall? _parseInlineToolCallJson(String input) {
    Object? decoded;
    try {
      decoded = jsonDecode(input);
    } catch (_) {
      return null;
    }
    if (decoded is! Map) return null;
    final name = decoded['name'];
    if (name is! String || name.isEmpty) return null;
    final argsRaw = decoded['arguments'] ?? decoded['parameters'] ?? const {};
    Map<String, Object?> args;
    if (argsRaw is Map) {
      args = argsRaw.map((k, v) => MapEntry(k.toString(), v));
    } else if (argsRaw is String) {
      // Some models emit `"arguments": "{\"x\":1}"` (string-escaped JSON).
      try {
        final inner = jsonDecode(argsRaw);
        if (inner is Map) {
          args = inner.map((k, v) => MapEntry(k.toString(), v));
        } else {
          return null;
        }
      } catch (_) {
        return null;
      }
    } else {
      return null;
    }
    return _ParsedCall(name: name, arguments: args);
  }

  static _JsonResult? _consumeJsonObject(
    String text,
    int start,
    int maxPayloadBytes,
  ) {
    final cursor = _skipWhitespace(text, start);
    if (cursor >= text.length || text[cursor] != '{') return null;
    var depth = 0;
    var inString = false;
    var escaped = false;
    for (var i = cursor; i < text.length; i++) {
      if (i + 1 - cursor > maxPayloadBytes) return null;
      final ch = text[i];
      if (inString) {
        if (escaped) {
          escaped = false;
        } else if (ch == r'\') {
          escaped = true;
        } else if (ch == '"') {
          inString = false;
        }
        continue;
      }
      if (ch == '"') {
        inString = true;
        continue;
      }
      if (ch == '{') {
        depth += 1;
      } else if (ch == '}') {
        depth -= 1;
        if (depth == 0) {
          final raw = text.substring(cursor, i + 1);
          try {
            final decoded = jsonDecode(raw);
            if (decoded is! Map) return null;
            return _JsonResult(
              end: i + 1,
              value: decoded.map((k, v) => MapEntry(k.toString(), v)),
            );
          } catch (_) {
            return null;
          }
        }
      }
    }
    return null;
  }

  static int _skipWhitespace(String text, int start) {
    var i = start;
    while (i < text.length && _isWhitespace(text.codeUnitAt(i))) {
      i++;
    }
    return i;
  }

  static bool _isWhitespace(int code) =>
      code == 0x20 || code == 0x09 || code == 0x0A || code == 0x0D;
}

class _ParsedCall {
  final String name;
  final Map<String, Object?> arguments;
  _ParsedCall({required this.name, required this.arguments});
}

class _JsonResult {
  final int end;
  final Map<String, Object?> value;
  _JsonResult({required this.end, required this.value});
}
