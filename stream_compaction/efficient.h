#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        extern int blockSize;

        void scan(int n, int *odata, const int *idata);

        int compact(int n, int *odata, const int *idata);

        void scanOnDevice(int n, int *dev_data);
    }
}
