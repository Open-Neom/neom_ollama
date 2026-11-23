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

  /// Gemma-4 chat template tag format:
  /// `<|tool_call>call:toolName{"arg": "val"}<tool_call|>` or
  /// `<|tool_call>{"name": "toolName", "arguments": {...}}<tool_call|>`.
  gemma4Tag,

  /// Standalone Gemma 4 function-call syntax (`tool_name(k="v")` or
  /// `call:tool_name{k: "v"}`) gated to start-of-reply with a declared tool
  /// name and no trailing explanatory prose (strips `;` / `</tool_call>` residue).
  functionSyntax,
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
  /// matches Gateway's upstream budget.
  static const int defaultMaxPayloadBytes = 256 * 1024;
  static const int _maxSanitizeDepth = 16;
  static const int _maxArgumentCollectionItems = 1024;

  static final RegExp _validToolNameRe = RegExp(r'^[A-Za-z0-9_.\-:]{1,128}$');
  static const Set<String> _prototypePollutionKeys = {
    '__proto__',
    'constructor',
    'prototype',
  };

  static bool _isValidToolName(String name) =>
      _validToolNameRe.hasMatch(name) &&
      name != '.' &&
      !name.contains('..') &&
      !(name.startsWith('__') && name.endsWith('__')) &&
      !_prototypePollutionKeys.contains(name.toLowerCase());

  /// Closed and unclosed reasoning or untrusted memory/context spans
  /// (`<think>`, `<thinking>`, `<thought>`, `<memory_recall>`, `<auto_memory>`,
  /// `<system_context>`, `[BIOCHIP]...[/BIOCHIP]`, Gemma-4
  /// `<|channel>thought...<channel|>` / `<|turn>thought...<turn|>`) inside
  /// which hypothetical or prompt-injected tool calls must NEVER be extracted.
  static final RegExp _closedReasoningSpanRe = RegExp(
    r'(?:<\s*(think(?:ing)?|thought|memory_recall|auto_memory|system_context)\b[^>]*>[\s\S]*?<\s*/\s*\1\s*>|\[\s*(BIOCHIP|MEMORY_RECALL|AUTO_MEMORY|SYSTEM_CONTEXT)\s*\](?:(?!\[\s*\2\s*\])[\s\S])*?\[\s*/\s*\2\s*\]|<\|(?:channel|turn)\|?>\s*thought\b[\s\S]*?(?:<\|?(?:channel|turn)\|>|</(?:channel|turn)>))',
    caseSensitive: false,
  );
  static final RegExp _openReasoningSpanRe = RegExp(
    r'(?:<\s*(?:think(?:ing)?|thought|memory_recall|auto_memory|system_context)\b[^>]*>|<\|(?:channel|turn)\|?>\s*thought\b)[\s\S]*$',
    caseSensitive: false,
  );

  static List<(int, int)> _findReasoningSpans(String text) {
    final spans = <(int, int)>[];
    var cursor = 0;
    for (final m in _closedReasoningSpanRe.allMatches(text)) {
      spans.add((m.start, m.end));
      if (m.end > cursor) cursor = m.end;
    }
    if (cursor < text.length) {
      final tailMatch = _openReasoningSpanRe.firstMatch(text.substring(cursor));
      if (tailMatch != null) {
        spans.add((cursor + tailMatch.start, text.length));
      }
    }
    return spans;
  }

  static bool _isInsideReasoningSpan(
    int start,
    int end,
    List<(int, int)> spans,
  ) {
    for (final (rStart, rEnd) in spans) {
      if (start < rEnd && end > rStart) return true;
    }
    return false;
  }

  static Object? _sanitizeArgumentValue(Object? value, [int depth = 0]) {
    if (value == null || value is bool || value is String) {
      return value;
    }
    if (value is num) {
      return value.isFinite ? value : null;
    }
    if (depth >= _maxSanitizeDepth) return null;
    if (value is Map) {
      final out = <String, Object?>{};
      for (final entry in value.entries) {
        if (out.length >= _maxArgumentCollectionItems) break;
        final key = entry.key.toString();
        final cleanKey = key.trim();
        if (_prototypePollutionKeys.contains(cleanKey.toLowerCase()) ||
            (cleanKey.startsWith('__') && cleanKey.endsWith('__'))) {
          continue;
        }
        out[key] = _sanitizeArgumentValue(entry.value, depth + 1);
      }
      return out;
    }
    if (value is List) {
      return value
          .take(_maxArgumentCollectionItems)
          .map((e) => _sanitizeArgumentValue(e, depth + 1))
          .toList();
    }
    return value;
  }

  static Map<String, Object?> _sanitizeArgumentMap(Map<Object?, Object?> raw) {
    final out = <String, Object?>{};
    for (final entry in raw.entries) {
      if (out.length >= _maxArgumentCollectionItems) break;
      final key = entry.key.toString();
      final cleanKey = key.trim();
      if (_prototypePollutionKeys.contains(cleanKey.toLowerCase()) ||
          (cleanKey.startsWith('__') && cleanKey.endsWith('__'))) {
        continue;
      }
      out[key] = _sanitizeArgumentValue(entry.value, 1);
    }
    return out;
  }

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
    if (text.isEmpty || !maxPayloadBytes.isFinite || maxPayloadBytes <= 0) {
      return const [];
    }

    final reasoningSpans = _findReasoningSpans(text);
    final blocks = <PlainTextToolCallBlock>[];
    blocks.addAll(_parseBracketBlocks(text, maxPayloadBytes));
    blocks.addAll(
      _parseTagBlocks(
        text,
        'tool_call',
        PlainTextToolCallFormat.toolCallTag,
        maxPayloadBytes,
      ),
    );
    blocks.addAll(
      _parseTagBlocks(
        text,
        'function_call',
        PlainTextToolCallFormat.functionCallTag,
        maxPayloadBytes,
      ),
    );
    blocks.addAll(_parseGemma4Blocks(text, maxPayloadBytes));
    blocks.addAll(_parseFencedBlocks(text, maxPayloadBytes));

    // Filter out any candidate blocks inside chain-of-thought reasoning spans
    // BEFORE checking whether bare JSON fallback should run.
    blocks.removeWhere(
      (b) => _isInsideReasoningSpan(b.start, b.end, reasoningSpans),
    );

    // Bare-JSON detection runs only when nothing else matched. Otherwise we
    // would double-count the inner JSON of a tag- or fence-wrapped block.
    if (blocks.isEmpty && reasoningSpans.isEmpty) {
      final bare = _parseBareJson(text, maxPayloadBytes);
      if (bare != null) blocks.add(bare);
    }

    // Gemma 4 standalone function-syntax fallback: strictly gated to start of
    // visible reply (after any leading reasoning span), declared in
    // `allowedToolNames`, balanced closing delimiter, and no trailing prose.
    if (blocks.isEmpty &&
        allowedToolNames != null &&
        allowedToolNames.isNotEmpty) {
      final fnBlock = _parseGemmaFunctionSyntax(
        text,
        reasoningSpans,
        allowedToolNames,
        maxPayloadBytes,
      );
      if (fnBlock != null) blocks.add(fnBlock);
    }

    // Document order, then dedupe overlapping ranges (paranoia).
    blocks.sort((a, b) => a.start.compareTo(b.start));
    final filtered = <PlainTextToolCallBlock>[];
    for (final block in blocks) {
      if (filtered.isNotEmpty && block.start < filtered.last.end) continue;
      if (!_isValidToolName(block.name)) continue;
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
    int maxPayloadBytes,
  ) {
    final blocks = <PlainTextToolCallBlock>[];
    final escaped = RegExp.escape(tag);
    final pattern = RegExp(
      '<$escaped>([\\s\\S]*?)</$escaped>',
      caseSensitive: false,
    );
    for (final match in pattern.allMatches(text)) {
      final inner = match.group(1)!.trim();
      if (inner.length > maxPayloadBytes) continue;
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
  //  Gemma-4 tag form: <|tool_call>call:name{...}<tool_call|>
  // ─────────────────────────────────────────────────────────

  static final RegExp _gemma4Pattern = RegExp(
    r'<\|tool_call>([\s\S]*?)<tool_call\|>',
    caseSensitive: false,
  );

  static List<PlainTextToolCallBlock> _parseGemma4Blocks(
    String text,
    int maxPayloadBytes,
  ) {
    final blocks = <PlainTextToolCallBlock>[];
    for (final match in _gemma4Pattern.allMatches(text)) {
      final inner = match.group(1)!.trim();
      if (inner.length > maxPayloadBytes) continue;
      if (inner.startsWith('call:')) {
        final afterCall = inner.substring('call:'.length).trim();
        final braceIdx = afterCall.indexOf('{');
        if (braceIdx <= 0) continue;
        final name = afterCall.substring(0, braceIdx).trim();
        if (!_isValidToolName(name)) continue;
        final jsonPart = afterCall.substring(braceIdx).trim();
        if (jsonPart.length > maxPayloadBytes) continue;
        try {
          final decoded = jsonDecode(jsonPart);
          if (decoded is Map) {
            blocks.add(
              PlainTextToolCallBlock(
                name: name,
                arguments: _sanitizeArgumentMap(decoded),
                start: match.start,
                end: match.end,
                raw: match.group(0)!,
                format: PlainTextToolCallFormat.gemma4Tag,
              ),
            );
          }
        } catch (_) {}
      } else {
        final parsed = _parseInlineToolCallJson(inner);
        if (parsed != null) {
          blocks.add(
            PlainTextToolCallBlock(
              name: parsed.name,
              arguments: parsed.arguments,
              start: match.start,
              end: match.end,
              raw: match.group(0)!,
              format: PlainTextToolCallFormat.gemma4Tag,
            ),
          );
        }
      }
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

  static List<PlainTextToolCallBlock> _parseFencedBlocks(
    String text,
    int maxPayloadBytes,
  ) {
    final blocks = <PlainTextToolCallBlock>[];
    for (final match in _fencePattern.allMatches(text)) {
      final inner = match.group(1)!.trim();
      if (inner.length > maxPayloadBytes) continue;
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

  static PlainTextToolCallBlock? _parseBareJson(
    String text,
    int maxPayloadBytes,
  ) {
    final trimmed = text.trim();
    if (trimmed.length > maxPayloadBytes) return null;
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
  //  Gemma 4 standalone function syntax:
  //    fn_name(k="v", n=2)   or   call:fn_name{k: <|"|>v<|"|>}
  //  Gated to reply start + declared tool + closed args + residue-only tail.
  // ─────────────────────────────────────────────────────────

  static final RegExp _gemmaFnHeaderRe = RegExp(
    r'^(?:call\s*:\s*)?([A-Za-z0-9_.\-:]{1,128})\s*([({])',
  );
  static final RegExp _gemmaTrailingResidueRe = RegExp(
    r'^(?:\s|;|</tool_call>|<\|call\|>|<call\|>)*$',
    caseSensitive: false,
  );

  static PlainTextToolCallBlock? _parseGemmaFunctionSyntax(
    String text,
    List<(int, int)> reasoningSpans,
    Set<String> allowedToolNames,
    int maxPayloadBytes,
  ) {
    // Advance past leading whitespace and any leading reasoning spans.
    var cursor = _skipWhitespace(text, 0);
    var advanced = true;
    while (advanced && cursor < text.length) {
      advanced = false;
      for (final (rStart, rEnd) in reasoningSpans) {
        if (cursor >= rStart && cursor < rEnd) {
          cursor = _skipWhitespace(text, rEnd);
          advanced = true;
          break;
        }
      }
    }
    if (cursor >= text.length) return null;

    final tail = text.substring(cursor);
    final match = _gemmaFnHeaderRe.firstMatch(tail);
    if (match == null) return null;

    final name = match.group(1)!;
    final openChar = match.group(2)!;
    if (!_isValidToolName(name) || !allowedToolNames.contains(name)) {
      return null;
    }

    final closeChar = openChar == '(' ? ')' : '}';
    final openIdx = cursor + match.end - 1;
    final closeIdx = _findMatchingDelimiter(
      text,
      openIdx,
      openChar,
      closeChar,
      maxPayloadBytes,
    );
    if (closeIdx == null) return null;

    // Gate: trailing text after the closing delimiter must ONLY be whitespace
    // or recognized `;` / `</tool_call>` residue (never explanatory prose like
    // `get_weather(location="Paris") would retrieve the forecast.`).
    final afterClose = text.substring(closeIdx + 1);
    if (!_gemmaTrailingResidueRe.hasMatch(afterClose)) {
      return null;
    }

    final innerArgs = text.substring(openIdx + 1, closeIdx).trim();
    final parsedArgs = openChar == '{'
        ? _parseGemmaObjectArgs('{$innerArgs}')
        : _parseFunctionKeywordArgs(innerArgs);
    if (parsedArgs == null) return null;

    // Consume trailing `;` / `</tool_call>` residue into the block span so
    // `strip()` leaves clean user-visible output with zero leaked tags.
    final fullEnd = text.length;
    return PlainTextToolCallBlock(
      name: name,
      arguments: _sanitizeArgumentMap(parsedArgs),
      start: cursor,
      end: fullEnd,
      raw: text.substring(cursor, fullEnd),
      format: PlainTextToolCallFormat.functionSyntax,
    );
  }

  static int? _findMatchingDelimiter(
    String text,
    int openIdx,
    String openChar,
    String closeChar,
    int maxPayloadBytes,
  ) {
    var depth = 0;
    String? quote;
    var inGemmaQuote = false;
    var escaped = false;

    for (var i = openIdx; i < text.length; i++) {
      if (i - openIdx > maxPayloadBytes) return null;
      if (inGemmaQuote) {
        if (text.startsWith('<|"|>', i)) {
          inGemmaQuote = false;
          i += 4;
        }
        continue;
      }
      final ch = text[i];
      if (quote != null) {
        if (escaped) {
          escaped = false;
        } else if (ch == r'\') {
          escaped = true;
        } else if (ch == quote) {
          quote = null;
        }
        continue;
      }
      if (text.startsWith('<|"|>', i)) {
        inGemmaQuote = true;
        i += 4;
        continue;
      }
      if (ch == '"' || ch == "'") {
        quote = ch;
        continue;
      }
      if (ch == openChar || ch == '{' || ch == '(' || ch == '[') {
        depth += 1;
      } else if (ch == closeChar || ch == '}' || ch == ')' || ch == ']') {
        depth -= 1;
        if (depth == 0) {
          return ch == closeChar ? i : null;
        }
        if (depth < 0) return null;
      }
    }
    return null;
  }

  static Map<Object?, Object?>? _parseGemmaObjectArgs(String rawObj) {
    // Normalize Gemma 4 `<|"|>...<|"|>` delimiters to standard JSON quotes.
    final normalized = rawObj.replaceAll('<|"|>', '"');
    try {
      final decoded = jsonDecode(normalized);
      if (decoded is Map) return decoded;
    } catch (_) {}
    // Fallback for unquoted keys `{location: "Paris", units: "c"}`.
    final inner = normalized.trim();
    if (!inner.startsWith('{') || !inner.endsWith('}')) return null;
    return _parseFunctionKeywordArgs(
      inner.substring(1, inner.length - 1).trim(),
      allowColonSeparator: true,
    );
  }

  static Map<Object?, Object?>? _parseFunctionKeywordArgs(
    String raw, {
    bool allowColonSeparator = false,
  }) {
    if (raw.isEmpty) return <Object?, Object?>{};
    final out = <Object?, Object?>{};
    var i = 0;

    while (i < raw.length) {
      while (i < raw.length &&
          (_isWhitespace(raw.codeUnitAt(i)) || raw[i] == ',')) {
        i++;
      }
      if (i >= raw.length) break;

      // Parse key (bare identifier or quoted string).
      String key;
      if (raw[i] == '"' || raw[i] == "'") {
        final q = raw[i++];
        final buf = StringBuffer();
        var closed = false;
        while (i < raw.length) {
          final c = raw[i++];
          if (c == r'\' && i < raw.length) {
            buf.write(raw[i++]);
          } else if (c == q) {
            closed = true;
            break;
          } else {
            buf.write(c);
          }
        }
        if (!closed) return null;
        key = buf.toString();
      } else {
        final start = i;
        while (i < raw.length &&
            RegExp(r'[A-Za-z0-9_.\-]').hasMatch(raw[i])) {
          i++;
        }
        if (i == start) return null;
        key = raw.substring(start, i);
      }

      i = _skipWhitespace(raw, i);
      if (i >= raw.length) return null;
      final sep = raw[i];
      if (sep != '=' && !(allowColonSeparator && sep == ':')) return null;
      i = _skipWhitespace(raw, i + 1);
      if (i >= raw.length) return null;

      // Extract value up to top-level comma.
      final valStart = i;
      var depth = 0;
      String? quote;
      var escaped = false;
      while (i < raw.length) {
        final ch = raw[i];
        if (quote != null) {
          if (escaped) {
            escaped = false;
          } else if (ch == r'\') {
            escaped = true;
          } else if (ch == quote) {
            quote = null;
          }
        } else if (ch == '"' || ch == "'") {
          quote = ch;
        } else if (ch == '{' || ch == '[' || ch == '(') {
          depth++;
        } else if (ch == '}' || ch == ']' || ch == ')') {
          depth--;
          if (depth < 0) return null;
        } else if (ch == ',' && depth == 0) {
          break;
        }
        i++;
      }
      if (quote != null || depth != 0) return null;
      final rawVal = raw.substring(valStart, i).trim();
      final decodedVal = _decodeLooseScalarOrJson(rawVal);
      if (decodedVal == _parseFailureSentinel) return null;
      out[key] = decodedVal;
    }
    return out;
  }

  static const Object _parseFailureSentinel = Object();

  static Object? _decodeLooseScalarOrJson(String raw) {
    if (raw.isEmpty) return _parseFailureSentinel;
    if (raw == 'true' || raw == 'True') return true;
    if (raw == 'false' || raw == 'False') return false;
    if (raw == 'null' || raw == 'None') return null;
    if (raw.startsWith("'") && raw.endsWith("'") && raw.length >= 2) {
      return raw
          .substring(1, raw.length - 1)
          .replaceAll(r"\'", "'")
          .replaceAll(r'\\', r'\');
    }
    try {
      return jsonDecode(raw);
    } catch (_) {
      return _parseFailureSentinel;
    }
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
    if (name is! String || !_isValidToolName(name)) return null;
    final argsRaw =
        decoded['arguments'] ?? decoded['parameters'] ?? const <String, Object?>{};
    Map<String, Object?> args;
    if (argsRaw is Map) {
      args = _sanitizeArgumentMap(argsRaw);
    } else if (argsRaw is String) {
      // Some models emit `"arguments": "{\"x\":1}"` (string-escaped JSON).
      try {
        final inner = jsonDecode(argsRaw);
        if (inner is Map) {
          args = _sanitizeArgumentMap(inner);
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
              value: _sanitizeArgumentMap(decoded),
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
