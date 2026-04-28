import 'package:neom_ollama/neom_ollama.dart';
import 'package:test/test.dart';

void main() {
  group('PlainTextToolCallParser - bracket form', () {
    test('parses [name]\\n{...}\\n[END_TOOL_REQUEST]', () {
      final text =
          'Sure, let me check.\n\n[get_weather]\n{"city": "Mexico City"}\n[END_TOOL_REQUEST]';
      final blocks = PlainTextToolCallParser.parse(text);
      expect(blocks, hasLength(1));
      final block = blocks.first;
      expect(block.name, equals('get_weather'));
      expect(block.format, PlainTextToolCallFormat.bracket);
      expect(block.arguments, equals({'city': 'Mexico City'}));
    });

    test('parses [name]\\n{...}\\n[/name] alternative closer', () {
      final text = '[search]\n{"query": "dart"}\n[/search]';
      final blocks = PlainTextToolCallParser.parse(text);
      expect(blocks, hasLength(1));
      expect(blocks.first.arguments, equals({'query': 'dart'}));
    });

    test('parses multiple bracket blocks in order', () {
      final text =
          '[a]\n{"x":1}\n[END_TOOL_REQUEST]\n[b]\n{"y":2}\n[END_TOOL_REQUEST]';
      final blocks = PlainTextToolCallParser.parse(text);
      expect(blocks, hasLength(2));
      expect(blocks[0].name, equals('a'));
      expect(blocks[1].name, equals('b'));
      expect(blocks[0].start, lessThan(blocks[1].start));
    });
  });

  group('PlainTextToolCallParser - tag forms', () {
    test('parses <tool_call>{...}</tool_call> (Hermes / Qwen style)', () {
      final text =
          'I need to look that up.\n<tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</tool_call>';
      final blocks = PlainTextToolCallParser.parse(text);
      expect(blocks, hasLength(1));
      expect(blocks.first.name, equals('get_weather'));
      expect(blocks.first.format, PlainTextToolCallFormat.toolCallTag);
      expect(blocks.first.arguments, equals({'city': 'Tokyo'}));
    });

    test('parses <function_call>{...}</function_call>', () {
      final text =
          '<function_call>{"name": "calc", "arguments": {"expr": "2+2"}}</function_call>';
      final blocks = PlainTextToolCallParser.parse(text);
      expect(blocks, hasLength(1));
      expect(blocks.first.format, PlainTextToolCallFormat.functionCallTag);
      expect(blocks.first.arguments, equals({'expr': '2+2'}));
    });

    test('accepts "parameters" as alias for "arguments"', () {
      final text =
          '<tool_call>{"name": "search", "parameters": {"q": "test"}}</tool_call>';
      final blocks = PlainTextToolCallParser.parse(text);
      expect(blocks, hasLength(1));
      expect(blocks.first.arguments, equals({'q': 'test'}));
    });

    test('decodes string-escaped arguments (some Llama variants)', () {
      final text =
          r'<tool_call>{"name": "x", "arguments": "{\"foo\":42}"}</tool_call>';
      final blocks = PlainTextToolCallParser.parse(text);
      expect(blocks, hasLength(1));
      expect(blocks.first.arguments, equals({'foo': 42}));
    });

    test('handles nested objects in arguments', () {
      final text =
          '<tool_call>{"name": "draw", "arguments": {"shape": {"type": "circle", "r": 3}}}</tool_call>';
      final blocks = PlainTextToolCallParser.parse(text);
      expect(blocks, hasLength(1));
      expect(
        blocks.first.arguments,
        equals({
          'shape': {'type': 'circle', 'r': 3},
        }),
      );
    });
  });

  group('PlainTextToolCallParser - fenced JSON', () {
    test('parses ```json {...} ```', () {
      final text =
          'Looking up:\n```json\n{"name": "lookup", "arguments": {"id": 7}}\n```';
      final blocks = PlainTextToolCallParser.parse(text);
      expect(blocks, hasLength(1));
      expect(blocks.first.format, PlainTextToolCallFormat.fencedJson);
      expect(blocks.first.arguments, equals({'id': 7}));
    });

    test('parses ```tool_call {...} ```', () {
      final text =
          '```tool_call\n{"name": "fetch", "arguments": {"url": "https://x.io"}}\n```';
      final blocks = PlainTextToolCallParser.parse(text);
      expect(blocks, hasLength(1));
      expect(blocks.first.name, equals('fetch'));
    });
  });

  group('PlainTextToolCallParser - bare JSON', () {
    test('parses whole-message JSON tool call', () {
      final text = '{"name": "ping", "arguments": {}}';
      final blocks = PlainTextToolCallParser.parse(text);
      expect(blocks, hasLength(1));
      expect(blocks.first.format, PlainTextToolCallFormat.bareJson);
      expect(blocks.first.name, equals('ping'));
    });

    test('does NOT match bare JSON when other markers are present', () {
      final text =
          '<tool_call>{"name": "a", "arguments": {}}</tool_call>\n\n{"name": "b", "arguments": {}}';
      final blocks = PlainTextToolCallParser.parse(text);
      // Only the tag block matches; bare JSON is suppressed when tags exist.
      expect(blocks, hasLength(1));
      expect(blocks.first.name, equals('a'));
    });

    test('rejects bare JSON without name field', () {
      final blocks = PlainTextToolCallParser.parse('{"foo": "bar"}');
      expect(blocks, isEmpty);
    });
  });

  group('PlainTextToolCallParser - allowedToolNames', () {
    test('drops blocks not in the allowlist', () {
      final text =
          '<tool_call>{"name": "rm", "arguments": {}}</tool_call><tool_call>{"name": "ls", "arguments": {}}</tool_call>';
      final blocks = PlainTextToolCallParser.parse(
        text,
        allowedToolNames: {'ls'},
      );
      expect(blocks, hasLength(1));
      expect(blocks.first.name, equals('ls'));
    });
  });

  group('PlainTextToolCallParser - resilience', () {
    test('returns empty for empty input', () {
      expect(PlainTextToolCallParser.parse(''), isEmpty);
    });

    test('returns empty for plain text with no markers', () {
      expect(
        PlainTextToolCallParser.parse('Just a regular response, no tool call.'),
        isEmpty,
      );
    });

    test('ignores malformed JSON inside tags', () {
      final text = '<tool_call>{not valid json}</tool_call>';
      expect(PlainTextToolCallParser.parse(text), isEmpty);
    });

    test('ignores tool_call tag without name field', () {
      final text = '<tool_call>{"arguments": {}}</tool_call>';
      expect(PlainTextToolCallParser.parse(text), isEmpty);
    });

    test('caps payload size', () {
      final big = '"a": "${'x' * 1000}"';
      final blocks = PlainTextToolCallParser.parse(
        '[t]\n{$big}\n[END_TOOL_REQUEST]',
        maxPayloadBytes: 100,
      );
      expect(blocks, isEmpty);
    });
  });

  group('PlainTextToolCallParser.strip', () {
    test('removes a single tag block and trims whitespace', () {
      final text =
          'Here is the call:\n<tool_call>{"name": "x", "arguments": {}}</tool_call>\n';
      final blocks = PlainTextToolCallParser.parse(text);
      final stripped = PlainTextToolCallParser.strip(text, blocks);
      expect(stripped, equals('Here is the call:'));
    });

    test('removes multiple bracket blocks in order', () {
      final text =
          'Step 1: [a]\n{"x":1}\n[END_TOOL_REQUEST]\nStep 2: [b]\n{"y":2}\n[END_TOOL_REQUEST]';
      final blocks = PlainTextToolCallParser.parse(text);
      final stripped = PlainTextToolCallParser.strip(text, blocks);
      expect(stripped, contains('Step 1:'));
      expect(stripped, contains('Step 2:'));
      expect(stripped, isNot(contains('END_TOOL_REQUEST')));
      expect(stripped, isNot(contains(r'{"x":1}')));
    });

    test('returns input unchanged when blocks is empty', () {
      expect(
        PlainTextToolCallParser.strip('hello', const []),
        equals('hello'),
      );
    });
  });

  group('PlainTextToolCallParser.hasToolCalls', () {
    test('detects presence', () {
      expect(
        PlainTextToolCallParser.hasToolCalls(
          '<tool_call>{"name": "x", "arguments": {}}</tool_call>',
        ),
        isTrue,
      );
      expect(
        PlainTextToolCallParser.hasToolCalls('plain answer, no tools'),
        isFalse,
      );
    });
  });
}
