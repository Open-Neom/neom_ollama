import 'dart:io';
import 'package:neom_ollama/neom_ollama.dart';

void main() async {
  final client = OllamaClient();

  // Check status
  final status = await client.checkStatus();
  print('Ollama: $status');

  if (status != OllamaStatus.running) {
    print('Start Ollama first: ollama serve');
    return;
  }

  // List models
  final models = await client.listModels();
  for (final m in models) {
    print('  ${m.displayName} — ${m.sizeLabel}');
  }

  // Chat
  if (models.isNotEmpty) {
    final model = models.first.name;
    print('\nChatting with $model...');

    await for (final chunk in client.chatStream(model, 'Hello! Who are you?')) {
      stdout.write(chunk);
    }
    print('');
  }
}
