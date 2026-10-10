#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Naive {
        StreamCompaction::Common::PerformanceTimer& timer();

        extern int blockSize;

        void scan(int n, int *odata, const int *idata);
    }
}
