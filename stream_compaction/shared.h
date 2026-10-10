#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Shared {
        StreamCompaction::Common::PerformanceTimer& timer();

        extern int blockSize;
        extern bool avoidBankConflicts;

        // work-efficient scan in shared memory (GPU Gems 3, Example 39-2)
        void scan(int n, int *odata, const int *idata);

        // naive scan in shared memory (GPU Gems 3, Example 39-1)
        void scanNaive(int n, int *odata, const int *idata);
    }
}
