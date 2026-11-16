/// Parses Ollama/reasoning-model responses that emit a visible chain-of-thought
/// block (Qwen3, DeepSeek-R1, QwQ, Gemma 4, gpt-oss, etc.).
///
/// The convention across reasoning models is a `<think>…</think>` block
/// followed by the "real" answer. Some models use `<thinking>` or `<thought>`,
/// and a few emit a bare `Thought: …\nAnswer: …` pattern. This parser
/// normalises all of them into a single [ThinkingSplit] result.
///
/// Zero dependencies — pure Dart regex.
class ThinkingParser {
  ThinkingParser._();

  // Matches <think>…</think>, <thinking>…</thinking>, <thought>…</thought>, or
  // Gemma 4's <|channel>thought…<channel|> / <|turn>thought…<turn|>,
  // case-insensitive, multi-line, non-greedy (first closing tag wins).
  static final RegExp _tagRe = RegExp(
    r'(?:<(?:think(?:ing)?|thought)>([\s\S]*?)</(?:think(?:ing)?|thought)>|<\|(?:channel|turn)\|?>\s*thought\b([\s\S]*?)(?:<\|?(?:channel|turn)\|>|</(?:channel|turn)>))',
    caseSensitive: false,
  );

  // Matches an open `<think>`, `<thought>`, `<|channel>thought`, or `<|turn>thought` that never closes.
  static final RegExp _openRe = RegExp(
    r'(?:<(?:think(?:ing)?|thought)>|<\|(?:channel|turn)\|?>\s*thought\b)([\s\S]*)$',
    caseSensitive: false,
  );

  static final RegExp _residualTagsRe = RegExp(
    r'(?:</?(?:think(?:ing)?|thought)>|<\|(?:channel|turn)\|?>\s*thought\b|<\|?(?:channel|turn)\|>|</(?:channel|turn)>|<\|/?(?:turn|im_start|im_end)\|?>|<\|eot_id\|>|<end_of_turn>)',
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
  /// * If a tag is open but never closed (even after an earlier closed block),
  ///   everything after the dangling open tag is treated as thinking and never
  ///   leaks into `content`.
  /// * Multiple `<think>` blocks are concatenated in the `thinking` field
  ///   (separated by a blank line) and stripped from `content`.
  static ThinkingSplit split(String raw) {
    if (raw.isEmpty) return const ThinkingSplit(thinking: '', content: '');

    String cleanTags(String text) =>
        text.replaceAll(_residualTagsRe, '').trim();

    // 1. Extract all closed tags first, then check if a dangling open tag
    // remains in the stripped remainder (e.g. `<think>a</think>Visible<think>b`).
    final matches = _tagRe.allMatches(raw).toList();
    if (matches.isNotEmpty) {
      final thinkingParts = matches
          .map((m) => cleanTags(m.group(1) ?? m.group(2) ?? ''))
          .where((s) => s.isNotEmpty)
          .toList();
      var remainder = raw.replaceAll(_tagRe, '');
      final trailingOpen = _openRe.firstMatch(remainder);
      if (trailingOpen != null) {
        final trailingThought = cleanTags(trailingOpen.group(1)!);
        if (trailingThought.isNotEmpty) {
          thinkingParts.add(trailingThought);
        }
        remainder = remainder.substring(0, trailingOpen.start);
      }
      final content = cleanTags(remainder);
      return ThinkingSplit(
        thinking: thinkingParts.join('\n\n'),
        content: content,
      );
    }

    // 2. Try a single dangling open tag.
    final open = _openRe.firstMatch(raw);
    if (open != null) {
      final before = cleanTags(raw.substring(0, open.start));
      final thinking = cleanTags(open.group(1)!);
      return ThinkingSplit(thinking: thinking, content: before);
    }

    // 3. Try bare "Thought: … Answer: …".
    final bare = _bareRe.firstMatch(raw);
    if (bare != null) {
      return ThinkingSplit(
        thinking: bare.group(1)!.trim(),
        content: bare.group(2)!.trim(),
      );
    }

    return ThinkingSplit(thinking: '', content: cleanTags(raw));
  }

  /// Strips any thinking markers and returns just the user-visible content.
  static String stripThinking(String raw) => split(raw).content;

  /// Returns `true` if [raw] contains any recognised thinking marker.
  static bool hasThinking(String raw) =>
      _tagRe.hasMatch(raw) || _openRe.hasMatch(raw) || _bareRe.hasMatch(raw);
}

/// Result of [ThinkingParser.split].
class ThinkingSplit {
  final String thinking;
  final String content;

  const ThinkingSplit({required this.thinking, required this.content});

  bool get hasThinking => thinking.isNotEmpty;

  @override
  String toString() =>
      'ThinkingSplit(thinking: ${thinking.length} chars, content: ${content.length} chars)';
}
