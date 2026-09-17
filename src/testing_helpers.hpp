#pragma once

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <string>
#include <ctime>

inline int testFailures = 0;

template<typename T>
int cmpArrays(int n, T *a, T *b) {
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            printf("    a[%d] = %d, b[%d] = %d\n", i, a[i], i, b[i]);
            return 1;
        }
    }
    return 0;
}

void printDesc(const char *desc) {
    printf("==== %s ====\n", desc);
}

template<typename T>
void printCmpResult(int n, T *a, T *b) {
    const bool failed = cmpArrays(n, a, b) != 0;
    testFailures += failed;
    printf("    %s \n",
            failed ? "FAIL VALUE" : "passed");
}

template<typename T>
void printCmpLenResult(int n, int expN, T *a, T *b) {
    if (n != expN) {
        printf("    expected %d elements, got %d\n", expN, n);
    }
    const bool badCount = n < 0 || n != expN;
    const bool badValue = !badCount && cmpArrays(n, a, b);
    testFailures += badCount || badValue;
    printf("    %s \n", badCount ? "FAIL COUNT" : badValue ? "FAIL VALUE" : "passed");
}

void zeroArray(int n, int *a) {
    for (int i = 0; i < n; i++) {
        a[i] = 0;
    }
}

void onesArray(int n, int *a) {
    for (int i = 0; i < n; i++) {
        a[i] = 1;
    }
}

void genArray(int n, int *a, int maxval) {
    // Deterministic sequence, seeded once, for reproducible starter output.
    static const bool seeded = [] { srand(5650); return true; }();
    (void)seeded;

    for (int i = 0; i < n; i++) {
        a[i] = rand() % maxval;
    }
}

void printArray(int n, int *a, bool abridged = false) {
    printf("    [ ");
    for (int i = 0; i < n; i++) {
        if (abridged && i + 2 == 15 && n > 16) {
            i = n - 2;
            printf("... ");
        }
        printf("%3d ", a[i]);
    }
    printf("]\n");
}

template<typename T>
void printElapsedTime(T time, std::string note = "")
{
    std::cout << "   elapsed time: " << time << "ms    " << note << std::endl;
}
