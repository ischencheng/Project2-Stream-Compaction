#pragma once
#include "common.h"

namespace StreamCompaction {
namespace Shared {
Common::PerformanceTimer& timer();
void setBlockSize(int blockSize);
int blockSize();
void scan(int n, int* odata, const int* idata);
void scanNaive(int n, int* odata, const int* idata);
void scanUnpadded(int n, int* odata, const int* idata);
int compact(int n, int* odata, const int* idata);
void printOccupancy();
}
}
