import 'package:test/test.dart';
import 'package:neom_ollama/src/thinking_parser.dart';

void main() {
  group('ThinkingParser.split', () {
    test('empty input', () {
      final r = ThinkingParser.split('');
      expect(r.thinking, isEmpty);
      expect(r.content, isEmpty);
      expect(r.hasThinking, isFalse);
    });

    test('no markers returns input as content', () {
      final r = ThinkingParser.split('Hello world');
      expect(r.thinking, isEmpty);
      expect(r.content, equals('Hello world'));
    });

    test('closed <think> block', () {
      const input = '<think>Let me reason about this.</think>The answer is 42.';
      final r = ThinkingParser.split(input);
      expect(r.thinking, equals('Let me reason about this.'));
      expect(r.content, equals('The answer is 42.'));
      expect(r.hasThinking, isTrue);
    });

    test('closed <thinking> block (alt tag)', () {
      const input = '<thinking>step 1\nstep 2</thinking>Result: ok';
      final r = ThinkingParser.split(input);
      expect(r.thinking, equals('step 1\nstep 2'));
      expect(r.content, equals('Result: ok'));
    });

    test('multiple think blocks are concatenated', () {
      const input = '<think>first</think>intermediate<think>second</think>final';
      final r = ThinkingParser.split(input);
      expect(r.thinking, equals('first\n\nsecond'));
      expect(r.content, equals('intermediatefinal'));
    });

    test('dangling open <think> (streaming mid-thought)', () {
      const input = 'partial answer<think>I am still thinking about';
      final r = ThinkingParser.split(input);
      expect(r.thinking, equals('I am still thinking about'));
      expect(r.content, equals('partial answer'));
    });

    test('bare "Thought: … Answer: …" pattern', () {
      const input = 'Thought: This is a math problem.\nAnswer: 7';
      final r = ThinkingParser.split(input);
      expect(r.thinking, equals('This is a math problem.'));
      expect(r.content, equals('7'));
    });

    test('case-insensitive tag matching', () {
      const input = '<THINK>reasoning</THINK>answer';
      final r = ThinkingParser.split(input);
      expect(r.thinking, equals('reasoning'));
      expect(r.content, equals('answer'));
    });

    test('multi-line content inside tags', () {
      const input = '''<think>
line 1
line 2
line 3
</think>
Final answer here.''';
      final r = ThinkingParser.split(input);
      expect(r.thinking, contains('line 1'));
      expect(r.thinking, contains('line 3'));
      expect(r.content, equals('Final answer here.'));
    });
  });

  group('ThinkingParser.stripThinking', () {
    test('removes tags and returns clean content', () {
      expect(
        ThinkingParser.stripThinking('<think>x</think>hello'),
        equals('hello'),
      );
    });

    test('no-op on clean input', () {
      expect(ThinkingParser.stripThinking('hello'), equals('hello'));
    });
  });

  group('ThinkingParser.hasThinking', () {
    test('true for closed tag', () {
      expect(ThinkingParser.hasThinking('x<think>y</think>z'), isTrue);
    });
    test('true for dangling tag', () {
      expect(ThinkingParser.hasThinking('x<think>y'), isTrue);
    });
    test('false for clean content', () {
      expect(ThinkingParser.hasThinking('clean'), isFalse);
    });
  });
}
