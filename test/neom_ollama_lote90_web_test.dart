import 'dart:convert';

import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:neom_ollama/neom_ollama.dart';
import 'package:test/test.dart';

void main() {
  group('Lote 90 — neom_ollama Web & VM hardening', () {
    test('ThinkingParser strips Gemma-4 <|turn>thought...<turn|> and standalone <|turn|> tokens', () {
      final closed = ThinkingParser.split(
        '<|turn>thought\nLet me check carefully.\n<turn|>Respuesta limpia.<|turn|>',
      );
      expect(closed.thinking, 'Let me check carefully.');
      expect(closed.content, 'Respuesta limpia.');

      final unclosed = ThinkingParser.split(
        'Visible<|turn>thought\ndangling internal plan',
      );
      expect(unclosed.thinking, 'dangling internal plan');
      expect(unclosed.content, 'Visible');

      expect(
        ThinkingParser.stripThinking('Hola mundo<|turn|><end_of_turn>'),
        'Hola mundo',
      );
    });

    test('PlainTextToolCallParser suppresses tool calls inside <|turn>thought and rejects prototype/traversal names', () {
      const thoughtLeak =
          '<|turn>thought\nMaybe call <tool_call>{"name": "delete_file", "arguments": {"path": "/tmp/a"}}</tool_call>\n<turn|>'
          'No tool needed.';
      expect(PlainTextToolCallParser.parse(thoughtLeak), isEmpty);

      for (final badName in <String>[
        '__proto__',
        'constructor',
        'prototype',
        '.',
        'foo..bar',
      ]) {
        final payload = '<tool_call>{"name": "$badName", "arguments": {}}</tool_call>';
        expect(
          PlainTextToolCallParser.parse(payload),
          isEmpty,
          reason: 'Must reject $badName',
        );
      }
    });

    test('OllamaChatService strips standalone <|turn|>, guards UTF-16 error snippets, and scrubs rebuildHistory', () async {
      final emojiTail = '${'A' * 511}🔥EXTRA';
      var callIndex = 0;
      final mock = MockClient((req) async {
        callIndex++;
        if (callIndex == 1) {
          return http.Response(
            jsonEncode({
              'choices': [
                {
                  'message': {'role': 'assistant', 'content': 'Respuesta final<|turn|>'},
                },
              ],
            }),
            200,
          );
        }
        return http.Response.bytes(utf8.encode(emojiTail), 500);
      });

      final chat = OllamaChatService(model: 'gemma4:e4b', httpClient: mock);
      chat.rebuildHistory([
        {
          'role': 'user',
          'content': 'hola',
          '__proto__': 'evil',
          'constructor': 'evil',
        },
      ]);
      expect(chat.conversationMessages.single.containsKey('__proto__'), isFalse);
      expect(chat.conversationMessages.single.containsKey('constructor'), isFalse);

      final answer = await chat.sendMessage('pregunta');
      expect(answer, 'Respuesta final');
      expect(chat.conversationMessages.last['content'], 'Respuesta final');

      Object? caught;
      try {
        await chat.sendMessage('falla');
      } catch (e) {
        caught = e;
      }
      expect(caught, isNotNull);
      final msg = caught.toString();
      // Must not leave an unpaired high surrogate (0xD800..0xDBFF) before '...'
      for (var i = 0; i < msg.length; i++) {
        final cu = msg.codeUnitAt(i);
        if (cu >= 0xD800 && cu <= 0xDBFF) {
          expect(i + 1 < msg.length, isTrue);
          final next = msg.codeUnitAt(i + 1);
          expect(next >= 0xDC00 && next <= 0xDFFF, isTrue);
        }
      }
    });

    test('OllamaClient.normalizeEndpointUrl blocks TEST-NET/IETF/benchmark while preserving LAN, and validates model names', () {
      for (final blocked in <String>[
        'http://192.0.0.1:11434',
        'http://192.0.2.1:11434',
        'http://198.51.100.1:11434',
        'http://203.0.113.1:11434',
        'http://198.18.0.1:11434',
      ]) {
        expect(
          OllamaClient.normalizeEndpointUrl(blocked),
          'http://localhost:11434',
          reason: 'Should block $blocked',
        );
      }

      expect(
        OllamaClient.normalizeEndpointUrl('http://192.168.1.50:11434/'),
        'http://192.168.1.50:11434',
      );
      expect(
        OllamaClient.normalizeEndpointUrl('https://ollama.tail123.ts.net'),
        'https://ollama.tail123.ts.net',
      );

      expect(OllamaClient.isValidModelName('constructor'), isFalse);
      expect(OllamaClient.isValidModelName('prototype'), isFalse);
      expect(OllamaClient.isValidModelName('__proto__'), isFalse);
      expect(OllamaClient.isValidModelName('.'), isFalse);
      expect(OllamaClient.isValidModelName('gemma4\u2028:e4b'), isFalse);
    });
  });
}
