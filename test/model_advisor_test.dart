import 'package:neom_ollama/neom_ollama.dart';
import 'package:test/test.dart';

HardwareProfile _macProfile(int ramGb, HardwareTier tier) => HardwareProfile(
      totalRamGB: ramGb,
      availableRamGB: (ramGb * 0.7).round(),
      cpuName: 'Apple M4',
      cpuCores: 12,
      gpuBackend: GpuBackend.metal,
      gpuVramGB: ramGb,
      availableDiskGB: 200,
      isMac: true,
      macChip: 'M4',
      macTier: ramGb >= 32 ? 'max' : (ramGb >= 24 ? 'pro' : 'base'),
      tier: tier,
    );

void main() {
  const advisor = ModelAdvisor();

  group('ModelAdvisor Gemma 4 migration', () {
    test('never recommends legacy gemma2 across hardware tiers or categories', () {
      final profiles = [
        _macProfile(4, HardwareTier.minimal),
        _macProfile(8, HardwareTier.low),
        _macProfile(16, HardwareTier.medium),
        _macProfile(24, HardwareTier.high),
        _macProfile(36, HardwareTier.ultra),
      ];
      for (final hw in profiles) {
        for (final category in ['general', 'reasoning', 'coding']) {
          final rec = advisor.recommend(hw, category: category);
          expect(rec.recommended.name, isNot('gemma2'));
          expect(rec.upgrade?.name, isNot('gemma2'));
          for (final alt in rec.alternatives) {
            expect(alt.name, isNot('gemma2'));
          }
        }
      }
    });

    test('recommends gemma4:e4b on 8 GB Mac for reasoning', () {
      final rec = advisor.recommend(
        _macProfile(8, HardwareTier.low),
        category: 'reasoning',
      );
      expect(rec.fitsInMemory, isTrue);
      expect(rec.recommended.ollamaTag, 'gemma4:e4b');
      expect(rec.alternatives.any((m) => m.ollamaTag == 'gemma4:e2b'), isTrue);
    });

    test('recommends gemma4:26b MoE on 24 GB Mac with fast active-param tok/s', () {
      final hw = _macProfile(24, HardwareTier.high);
      final rec = advisor.recommend(hw, category: 'reasoning');
      expect(rec.fitsInMemory, isTrue);
      expect(rec.recommended.ollamaTag, 'gemma4:26b');
      // 26B-A4B has 3.8B active params -> should estimate fast 4B-class throughput
      // (35 * 1.2 = 42 tok/s), much faster than a dense 26B (8 * 1.2 = 10 tok/s).
      expect(rec.estimatedTokensPerSecond, greaterThanOrEqualTo(35));
    });

    test('recommends gemma4:31b on 32 GB Mac for reasoning', () {
      final rec = advisor.recommend(
        _macProfile(32, HardwareTier.ultra),
        category: 'reasoning',
      );
      expect(rec.fitsInMemory, isTrue);
      expect(rec.recommended.ollamaTag, 'gemma4:31b');
      expect(rec.alternatives.any((m) => m.ollamaTag == 'gemma4:26b'), isTrue);
    });

    test('uses total PLE parameters (5.1B) for gemma4:e2b memory sizing', () {
      // On 4 GB Mac (maxModelSizeGB = 2.6 GB), gemma4:e2b (~2.86 GB at Q4_K_M,
      // ~2.65 GB at Q4_K_S) should NOT falsely claim fitsInMemory under Q4_K_M.
      final rec = advisor.recommend(
        _macProfile(4, HardwareTier.minimal),
        category: 'reasoning',
      );
      expect(rec.recommended.ollamaTag, 'gemma4:e2b');
      expect(rec.recommended.paramsBillions, 5.1);
      expect(rec.fitsInMemory, isFalse);
    });
  });
}
