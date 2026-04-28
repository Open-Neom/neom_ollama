/// Parses Ollama/reasoning-model responses that emit a visible chain-of-thought
/// block (Qwen3, DeepSeek-R1, QwQ, gpt-oss, etc.).
///
/// The convention across reasoning models is a `<think>…</think>` block
/// followed by the "real" answer. Some models use `<thinking>`, and a few
/// emit a bare `Thought: …\nAnswer: …` pattern. This parser normalises all
/// of them into a single [ThinkingSplit] result.
///
/// Zero dependencies — pure Dart regex.
class ThinkingParser {
  ThinkingParser._();

  // Matches <think>…</think> or <thinking>…</thinking>, case-insensitive,
  // multi-line, greedy-non (first closing tag wins).
  static final RegExp _tagRe = RegExp(
    r'<think(?:ing)?>([\s\S]*?)</think(?:ing)?>',
    caseSensitive: false,
  );

  // Matches an open `<think>` that never closes (streaming / truncated).
  static final RegExp _openRe = RegExp(
    r'<think(?:ing)?>([\s\S]*)$',
    caseSensitive: false,
  );

  // Matches bare `Thought: …` / `Reasoning: …` blocks when no tags.
  static final RegExp _bareRe = RegExp(
    r'^\s*(?:Thought|Reasoning|Thinking)\s*:\s*([\s\S]*?)\n\s*(?:Answer|Response|Final)\s*:\s*([\s\S]+)$',
    caseSensitive: false,
  );

  /// Splits [raw] into `(thinking, content)`. Either part may be empty.
  ///
  /// * If no reasoning marker is present, `thinking` is empty and `content`
  ///   is the full input (trimmed).
  /// * If a tag is open but never closed (streaming mid-thought), everything
  ///   after `<think>` is treated as thinking and content is empty.
  /// * Multiple `<think>` blocks are concatenated in the `thinking` field
  ///   (separated by a blank line) and stripped from `content`.
  static ThinkingSplit split(String raw) {
    if (raw.isEmpty) return const ThinkingSplit(thinking: '', content: '');

    // Try closed tags first.
    final matches = _tagRe.allMatches(raw).toList();
    if (matches.isNotEmpty) {
      final thinkingParts =
          matches.map((m) => m.group(1)!.trim()).where((s) => s.isNotEmpty);
      final content = raw.replaceAll(_tagRe, '').trim();
      return ThinkingSplit(
        thinking: thinkingParts.join('\n\n'),
        content: content,
      );
    }

    // Try a dangling open tag.
    final open = _openRe.firstMatch(raw);
    if (open != null) {
      final before = raw.substring(0, open.start).trim();
      final thinking = open.group(1)!.trim();
      return ThinkingSplit(thinking: thinking, content: before);
    }

    // Try bare "Thought: … Answer: …".
    final bare = _bareRe.firstMatch(raw);
    if (bare != null) {
      return ThinkingSplit(
        thinking: bare.group(1)!.trim(),
        content: bare.group(2)!.trim(),
      );
    }

    return ThinkingSplit(thinking: '', content: raw.trim());
  }

  /// Strips any thinking markers and returns just the user-visible content.
  ///
  /// Convenience for callers that don't care about the reasoning trace.
  static String stripThinking(String raw) => split(raw).content;

  /// Returns `true` if [raw] contains any recognised thinking marker.
  static bool hasThinking(String raw) =>
      _tagRe.hasMatch(raw) || _openRe.hasMatch(raw) || _bareRe.hasMatch(raw);
}

/// Result of [ThinkingParser.split].
class ThinkingSplit {
  /// The chain-of-thought / reasoning trace. Empty when the response had no
  /// thinking markers.
  final String thinking;

  /// The user-visible answer, with all thinking markers removed.
  final String content;

  const ThinkingSplit({required this.thinking, required this.content});

  bool get hasThinking => thinking.isNotEmpty;

  @override
  String toString() =>
      'ThinkingSplit(thinking: ${thinking.length} chars, content: ${content.length} chars)';
}
