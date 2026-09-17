#pragma once
#include "common.h"
namespace StreamCompaction {
namespace Radix {
Common::PerformanceTimer& timer();
// Stable LSD sort of signed 32-bit integers, reusing Shared::scanDevice.
void sort(int n, int* odata, const int* idata);
}
}
