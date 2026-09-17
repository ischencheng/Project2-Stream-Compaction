#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);

        int compact(int n, int *odata, const int *idata);
        void setBlockSize(int blockSize);
        int blockSize();
        // Comparison for Part 5: one thread per padded element at every level.
        void scanUnoptimized(int n, int *odata, const int *idata);
    }
}
