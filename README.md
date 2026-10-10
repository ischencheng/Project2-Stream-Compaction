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

### CPU Scan and Compaction

The CPU scan is one `for` loop that keeps a running sum. `compactWithoutScan` copies every non-zero element to the next free slot of the output. `compactWithScan` does it the way the GPU will do it, in three steps: map each element to 1 if I keep it and 0 if not, scan this 0/1 array, and write each kept element to the index that the scan gives (scatter).

### Naive Scan

The naive scan runs log2(n) levels. At level d every element adds the element 2^(d-1) places to its left, so after the last level each element holds the sum of everything up to and including itself. A thread can't safely read an element that another thread is writing, so I use two buffers and swap them after each level. To get an exclusive scan I shift the input right by one when I copy it to the GPU. This needs O(n log n) additions in total.

### Work-Efficient Scan and Compaction

The work-efficient scan treats the array as a balanced binary tree. In the up-sweep each level adds the left child into the right child, so the last element ends up with the total sum. Then I set the last element to 0. In the down-sweep each node gives its value to its left child, and its value plus the old left child to its right child. This needs only O(n) additions and works in place. The tree needs a power-of-two size, so I pad the array with zeros to the next power of two.

Compaction on the GPU has the same three steps as on the CPU: `kernMapToBoolean`, the work-efficient scan and `kernScatter`. The number of kept elements is the last scan value plus the last 0/1 value.

### Thrust

This is just `thrust::exclusive_scan` on two `thrust::device_vector`s. The copies to and from the GPU are outside the timer, like in my own versions.
