CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Chen Cheng
  * [LinkedIn](https://www.linkedin.com/in/chen-andrew-cheng-34a133229/), [GitHub](https://github.com/ischencheng)
* Tested on: Windows 11, i5-12500H @ 2.50GHz 16GB, NVIDIA GeForce RTX 2050 4GB (personal laptop)

### Description

Stream compaction removes the elements I don't want from an array and packs the rest together. Here the elements to remove are the `0`s in an array of `int`s. The main building block is scan (prefix sum). In an exclusive scan each output element is the sum of all input elements before it, for example `[1 5 0 1 2 0 3]` becomes `[0 1 6 6 7 9 9]`.

Features:

* CPU scan, and CPU compaction with and without scan
* Naive GPU scan
* Work-efficient GPU scan and compaction
* Scan with Thrust
* Command line options for the array size and the block size
* Extra credit: a faster work-efficient scan, radix sort, and shared memory scans (naive and work-efficient, with padding against bank conflicts)

### CPU Scan and Compaction

The CPU scan is one `for` loop that keeps a running sum. `compactWithoutScan` copies every non-zero element to the next free slot of the output. `compactWithScan` does it the way the GPU will do it, in three steps: map each element to 1 if I keep it and 0 if not, scan this 0/1 array, and write each kept element to the index that the scan gives (scatter).

### Naive Scan

The naive scan runs log2(n) levels. At level d every element adds the element 2^(d-1) places to its left, so after the last level each element holds the sum of everything up to and including itself. A thread can't safely read an element that another thread is writing, so I use two buffers and swap them after each level. To get an exclusive scan I shift the input right by one when I copy it to the GPU. This needs O(n log n) additions in total.

### Work-Efficient Scan and Compaction

The work-efficient scan treats the array as a balanced binary tree. In the up-sweep each level adds the left child into the right child, so the last element ends up with the total sum. Then I set the last element to 0. In the down-sweep each node gives its value to its left child, and its value plus the old left child to its right child. This needs only O(n) additions and works in place. The tree needs a power-of-two size, so I pad the array with zeros to the next power of two.

Compaction on the GPU has the same three steps as on the CPU: `kernMapToBoolean`, the work-efficient scan and `kernScatter`. The number of kept elements is the last scan value plus the last 0/1 value.

### Thrust

This is just `thrust::exclusive_scan` on two `thrust::device_vector`s. The copies to and from the GPU are outside the timer, like in my own versions.

### Performance Analysis

I added command line options, so I don't have to recompile for every test: `-pow 20` runs the tests with 2^20 elements, and `-block 256` sets the block size of all my GPU versions. Before the tests the program runs every GPU version once, so loading the kernels is not part of the timing. All numbers are from a Release build. I ran each setting 3 times and took the middle run. Below 1 ms the numbers jump around a lot between runs (often by 15%, sometimes 2×), above 1 ms they are within a few percent. The tables in this section use my first work-efficient scan; the faster one is in Extra Credit: Faster Work-Efficient Scan.

This table shows the time in ms with 2^22 elements for different block sizes.

| block size | 32 | 64 | 128 | 256 | 512 | 1024 |
| --- | --- | --- | --- | --- | --- | --- |
| naive scan | 8.65 | 7.76 | 7.80 | 7.80 | 7.86 | 7.79 |
| work-efficient scan | 13.67 | 8.13 | 4.99 | 4.99 | 5.88 | 7.65 |
| work-efficient compact | 14.96 | 8.61 | 6.20 | 6.21 | 6.90 | 8.85 |

![scan](img/scan.png)

This table shows the scan time in ms (power-of-two sizes, block size 128). The non-power-of-two sizes (2^n - 3) take almost the same time, so I only show the power-of-two ones.

| size | CPU | naive | work-efficient | thrust |
| --- | --- | --- | --- | --- |
| 2^8 | 0.0002 | 0.035 | 0.100 | 0.043 |
| 2^16 | 0.022 | 0.120 | 0.242 | 0.033 |
| 2^18 | 0.097 | 0.454 | 0.375 | 0.350 |
| 2^20 | 0.414 | 1.79 | 1.42 | 0.416 |
| 2^22 | 1.96 | 7.81 | 5.46 | 0.734 |
| 2^26 | 33.3 | 154 | 84.9 | 6.06 |

From the table, thrust is the fastest and the only one of these GPU versions that beats the CPU, 5.5× faster at 2^26. My naive and work-efficient scans are slower than the CPU at every size. The work-efficient scan does fewer additions than the naive one, but it is only 1.8× faster at 2^26, and for small arrays it is even slower than naive, because it has twice as many kernel launches.

This table shows the compaction time in ms.

| size | CPU without scan | CPU with scan | work-efficient |
| --- | --- | --- | --- |
| 2^8 | 0.0008 | 0.0008 | 0.161 |
| 2^16 | 0.147 | 0.272 | 0.334 |
| 2^18 | 0.591 | 1.12 | 0.508 |
| 2^20 | 2.36 | 4.56 | 1.73 |
| 2^22 | 9.38 | 17.4 | 6.21 |
| 2^26 | 153 | 279 | 104 |

Compaction on the GPU wins from 2^18 on, even with my slow scan, and the scan is most of its time (84.9 of 104 ms at 2^26). On the CPU, compaction without scan is 4.6× slower than the CPU scan, although both are one loop over the array. I think this is because of the `if`: about a quarter of the random elements are 0, so the CPU can't guess the branch well. With scan it is 1.8× slower again, because it makes three passes over the data.

### Answer to Questions

* **Roughly optimize the block sizes of each of your implementations for minimal run time on your GPU. (You shouldn't compare unoptimized implementations to each other!)**

  The block size table is above. Naive is about the same from 64 to 1024, and work-efficient is fastest at 128 and 256, so I use 128 for both. Thrust picks its own block size. On this GPU an SM can hold at most 16 blocks and 1536 threads. So with 32 or 64 threads per block it only gets 512 or 1024 threads, and with 1024 threads only one block fits. I think naive doesn't care much because it is limited by memory anyway, while the work-efficient scan needs many threads in flight, because most of them do nothing (see below).

* **Compare all of these GPU Scan implementations (Naive, Work-Efficient, and Thrust) to the serial CPU version of Scan. Plot a graph of the comparison (with array size on the independent axis). We wrapped up both CPU and GPU timing functions as a performance timer class for you to conveniently measure the time cost. We use `std::chrono` to provide CPU high-precision timing and CUDA event to measure the CUDA performance. For CPU, put your CPU code between `timer().startCpuTimer()` and `timer().endCpuTimer()`. For GPU, put your CUDA code between `timer().startGpuTimer()` and `timer().endGpuTimer()`. Be sure not to include any initial/final memory operations (`cudaMalloc`, `cudaMemcpy`) in your performance measurements, for comparability. Don't mix up `CpuTimer` and `GpuTimer`. To guess at what might be happening inside the Thrust implementation (e.g. allocation, memory copy), take a look at the Nsight timeline for its execution. Your analysis here doesn't have to be detailed, since you aren't even looking at the code for the implementation.**

  The graph and the table are above. In the Nsight Systems timeline, every `thrust::exclusive_scan` call first calls `cudaMalloc` for some temporary memory, then launches two kernels (`DeviceScanInitKernel` and `DeviceScanKernel`), waits for them with `cudaStreamSynchronize`, and frees the memory with `cudaFree`. At 2^20 the two kernels take only about 77 µs of the 0.44 ms. The rest is the `cudaMalloc`, the `cudaFree`, and waiting for the first kernel to start. At 2^16 the same calls take only a few µs, which explains the jump of thrust between 2^16 and 2^18. My guess is that a small temporary buffer comes from memory the driver already has, and a bigger one needs a real new allocation.

* **Write a brief explanation of the phenomena you see here. Can you find the performance bottlenecks? Is it memory I/O? Computation? Is it different for each implementation?**

  Yes, it is different for each one. For small arrays all GPU versions are limited by fixed costs: each kernel launch takes a few µs, naive has log2(n) launches and work-efficient 2·log2(n). For big arrays naive is limited by memory: in each of its log2(n) levels it reads the whole array twice and writes it once. At 2^26 that is about 21 GB in total, and this GPU can move about 112 GB/s, so roughly 190 ms. I measured 154 ms, and I think some of the reads hit the cache. The work-efficient scan moves much less data, but it still launches one thread per element at every level, and at level d only one thread out of 2^(d+1) does something. So I think it is limited by the many idle threads, not by memory or computation. Thrust is limited by memory: at 2^26 it reads and writes 268 MB each in 6.06 ms, about 89 GB/s. The CPU scan is limited by being one thread, since each addition needs the result of the previous one.

* **Paste the output of the test program into a triple-backtick block in your README. If you add your own tests (e.g. for radix sort or to test additional corner cases), be sure to mention it explicitly.**

  This is the output of `cis5650_stream_compaction_test.exe -pow 20`. The shared memory scan and radix sort tests are my own (see the extra credit sections).

  ```
  ****************
  ** SCAN TESTS **
  ****************
      [  27   2   1  21  38  28  27  37   7  20  17  25  19 ...  44   0 ]
  ==== cpu scan, power-of-two ====
     elapsed time: 0.5012ms    (std::chrono Measured)
      [   0  27  29  30  51  89 117 144 181 188 208 225 250 ... 25693932 25693976 ]
  ==== cpu scan, non-power-of-two ====
     elapsed time: 0.4761ms    (std::chrono Measured)
      [   0  27  29  30  51  89 117 144 181 188 208 225 250 ... 25693889 25693923 ]
      passed
  ==== naive scan, power-of-two ====
     elapsed time: 1.79152ms    (CUDA Measured)
      passed
  ==== naive scan, non-power-of-two ====
     elapsed time: 1.85398ms    (CUDA Measured)
      passed
  ==== work-efficient scan, power-of-two ====
     elapsed time: 1.21133ms    (CUDA Measured)
      passed
  ==== work-efficient scan, non-power-of-two ====
     elapsed time: 1.19971ms    (CUDA Measured)
      passed
  ==== thrust scan, power-of-two ====
     elapsed time: 0.671744ms    (CUDA Measured)
      passed
  ==== thrust scan, non-power-of-two ====
     elapsed time: 0.672768ms    (CUDA Measured)
      passed
  ==== shared memory naive scan, power-of-two ====
     elapsed time: 0.220384ms    (CUDA Measured)
      passed
  ==== shared memory work-efficient scan, power-of-two ====
     elapsed time: 0.216128ms    (CUDA Measured)
      passed
  ==== shared memory work-efficient scan, non-power-of-two ====
     elapsed time: 0.208832ms    (CUDA Measured)
      passed

  *****************************
  ** STREAM COMPACTION TESTS **
  *****************************
      [   0   3   1   0   0   1   2   1   2   2   2   1   0 ...   0   0 ]
  ==== cpu compact without scan, power-of-two ====
     elapsed time: 3.2987ms    (std::chrono Measured)
      [   3   1   1   2   1   2   2   2   1   3   3   2   1 ...   1   2 ]
      passed
  ==== cpu compact without scan, non-power-of-two ====
     elapsed time: 3.2212ms    (std::chrono Measured)
      [   3   1   1   2   1   2   2   2   1   3   3   2   1 ...   1   2 ]
      passed
  ==== cpu compact with scan ====
     elapsed time: 6.4644ms    (std::chrono Measured)
      [   3   1   1   2   1   2   2   2   1   3   3   2   1 ...   1   2 ]
      passed
  ==== work-efficient compact, power-of-two ====
     elapsed time: 1.33939ms    (CUDA Measured)
      passed
  ==== work-efficient compact, non-power-of-two ====
     elapsed time: 1.3527ms    (CUDA Measured)
      passed

  **********************
  ** RADIX SORT TESTS **
  **********************
      [ 484 299 461 -288 -100 -219 334 233 -438 174 122 413  60 ... -12  62 ]
  ==== cpu std::sort, power-of-two ====
     elapsed time: 47.3022ms    (std::chrono Measured)
      [ -500 -500 -500 -500 -500 -500 -500 -500 -500 -500 -500 -500 -500 ... 499 499 ]
  ==== radix sort, power-of-two ====
     elapsed time: 40.3323ms    (CUDA Measured)
      [ -500 -500 -500 -500 -500 -500 -500 -500 -500 -500 -500 -500 -500 ... 499 499 ]
      passed
  ==== cpu std::sort, non-power-of-two ====
     elapsed time: 46.73ms    (std::chrono Measured)
      [ -500 -500 -500 -500 -500 -500 -500 -500 -500 -500 -500 -500 -500 ... 499 499 ]
  ==== radix sort, non-power-of-two ====
     elapsed time: 40.0339ms    (CUDA Measured)
      [ -500 -500 -500 -500 -500 -500 -500 -500 -500 -500 -500 -500 -500 ... 499 499 ]
      passed
  ```

### Extra Credit: Faster Work-Efficient Scan

My first work-efficient scan launched one thread per element at every level, and a thread only did something if its index was a multiple of 2^(d+1). So at the deeper levels almost every thread just checked its index and quit. Now thread k works on the k-th pair of the level, so a level only launches n / 2^(d+1) threads. A whole scan now launches about 2n threads instead of 2n·log2(n).

This table shows the work-efficient scan in ms before and after this change.

| size | CPU | before | after |
| --- | --- | --- | --- |
| 2^16 | 0.022 | 0.242 | 0.212 |
| 2^18 | 0.097 | 0.375 | 0.288 |
| 2^20 | 0.414 | 1.42 | 1.02 |
| 2^22 | 1.96 | 5.46 | 3.61 |
| 2^24 | 8.08 | 20.5 | 13.8 |
| 2^26 | 33.3 | 84.9 | 54.8 |

It is about 1.5× faster for big arrays, and compaction got faster with it (104 to 74.3 ms at 2^26). The block size doesn't matter any more: 32 to 1024 all give 3.5 to 3.7 ms at 2^22. But it is still slower than the CPU. The threads of a warp touch elements that are 2^(d+1) apart, and the GPU reads memory in pieces of 32 bytes, so from the third level on most of each piece is wasted. With Nsight Compute I measured that the 44 sweep kernels at 2^22 move 319 MB, 9.5× the 33.5 MB of one read and one write of the array. So now the scan is limited by memory.

### Extra Credit: Radix Sort

Radix sort sorts the numbers one bit at a time, from the lowest bit to the highest. Each pass splits the array by the current bit: numbers with a 0 bit go to the front, numbers with a 1 bit go after them, and both groups keep their order. I map each number to 1 if its bit is 0 and scan this array with my work-efficient scan. Then a number with bit 0 goes to `scan[i]`, and a number with bit 1 goes to `i - scan[i] + totalZeros`. I flip the sign bit before looking at the bits, so negative numbers come first.

It is in `stream_compaction/radix.cu` and is called like the scans:

```cpp
StreamCompaction::Radix::sort(n, odata, idata);
```

I added my own tests to `main.cpp`: random numbers from -500 to 499 are sorted by radix sort and checked against `std::sort`, for a power-of-two and a non-power-of-two size. This is part of the output with 16 elements (`-pow 4`):

```
**********************
** RADIX SORT TESTS **
**********************
    [ -472 -58   6 -328 -150 223 -219 -309 290 309 -279 195 370 -339 129 -75 ]
==== cpu std::sort, power-of-two ====
   elapsed time: 0.0003ms    (std::chrono Measured)
    [ -472 -339 -328 -309 -279 -219 -150 -75 -58   6 129 195 223 290 309 370 ]
==== radix sort, power-of-two ====
   elapsed time: 1.99885ms    (CUDA Measured)
    [ -472 -339 -328 -309 -279 -219 -150 -75 -58   6 129 195 223 290 309 370 ]
    passed
```

This table shows radix sort vs `std::sort` in ms.

| size | std::sort | radix sort |
| --- | --- | --- |
| 2^12 | 0.161 | 4.98 |
| 2^16 | 2.32 | 7.15 |
| 2^20 | 33.6 | 38.4 |
| 2^22 | 136 | 139 |
| 2^24 | 544 | 546 |
| 2^26 | 2290 | 2217 |

Radix sort catches up with `std::sort` at about 2^22. Each of its 32 passes is mostly one scan, so 32 scans of 3.6 ms make up most of the 139 ms at 2^22.

### Extra Credit: Shared Memory Scan

My global memory scans read and write the array again at every level. In the shared memory version each block loads a tile of the array into shared memory, does all the levels there, and writes it back, so a tile costs one read and one write of global memory. Each block also saves its tile sum. I scan the tile sums the same way, and a last kernel adds them to every element of their tile. I did both versions from GPU Gems 3: the naive one (Example 39-1), where B threads scan B elements with two buffers, and the work-efficient one (Example 39-2), where B threads scan 2B elements in place.

Shared memory is split into 32 banks. In the work-efficient scan the threads of a warp read elements that are 2, 4, 8, ... apart, so many of them hit the same bank and have to wait for each other (bank conflicts). Like in Section 39.2.3, I leave one empty slot after every 32 ints, so these elements land in different banks. `-pad 0` turns this off.

This table shows the scans in ms with block size 128 (for the shared work-efficient scan, 64 to 512 are all within 1%).

| size | CPU | global work-efficient | shared naive | shared work-efficient | shared work-efficient, no padding | thrust |
| --- | --- | --- | --- | --- | --- | --- |
| 2^16 | 0.023 | 0.227 | 0.047 | 0.041 | 0.044 | 0.029 |
| 2^18 | 0.072 | 0.300 | 0.076 | 0.073 | 0.082 | 0.312 |
| 2^20 | 0.457 | 1.03 | 0.214 | 0.211 | 0.251 | 0.441 |
| 2^22 | 2.09 | 3.66 | 0.700 | 0.696 | 0.738 | 0.795 |
| 2^24 | 7.52 | 13.8 | 2.64 | 2.61 | 2.77 | 1.99 |
| 2^26 | 32.6 | 54.7 | 10.4 | 10.3 | 10.9 | 6.04 |

The shared memory scan is 5.3× faster than my global memory scan at 2^26 and 3.2× faster than the CPU, so it is my first scan that beats the CPU for big arrays. From 2^18 to 2^22 it even beats thrust, because there thrust spends most of its time on `cudaMalloc` and `cudaFree` (see the Nsight answer above). At 2^26 thrust is still 1.7× faster. With Nsight Compute I measured that my scan moves about 16 bytes per element, because it reads and writes the array twice: in the tile kernel and when it adds the tile sums. Thrust's timeline shows only one scan kernel, so I think it reads and writes the array only once.

The naive and the work-efficient versions are almost the same here. Both are limited by global memory, and the extra additions of the naive one happen in fast shared memory. The padding cuts the bank conflicts of the tile kernel from 4.6 million to 38 thousand (Nsight Compute, 2^22), but the whole scan only takes 5 to 16% less time, again because global memory is the limit.
