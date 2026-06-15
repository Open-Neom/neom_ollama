// Stub fallback — should never be reached in practice.

import 'hardware_profile_model.dart';

Future<HardwareProfile> detectHardware() async {
  return const HardwareProfile(
    totalRamGB: 8,
    availableRamGB: 4,
    cpuName: 'Unknown',
    cpuCores: 4,
    gpuBackend: GpuBackend.none,
    gpuVramGB: 0,
    availableDiskGB: 0,
    isMac: false,
    isWeb: false,
    tier: HardwareTier.low,
    isEstimated: true,
  );
}
