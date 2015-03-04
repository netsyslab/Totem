/**
 *  Common definitions across the modules.
 *
 *  Created on: 2011-02-28
 *  Author: Abdullah Gharaibeh
 */

#ifndef TOTEM_COMDEF_H
#define TOTEM_COMDEF_H

// system includes
#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

// TBB includes
#include "tbb/parallel_sort.h"

// Function return code types
const int SUCCESS = 0;
const int FAILURE = -1;

// Processor types. This is used as to identify the the processor on which
// a partition is processed on.
typedef enum {
  PROCESSOR_CPU = 0,
  PROCESSOR_GPU,
  PROCESSOR_MAX
} processor_type_t;

typedef struct processor_s {
  processor_type_t type; /**< Processor type (CPU or GPU). */
  uint32_t         id;   /**< Used to id GPU devices */
} processor_t;


// Stopwatch (timer) type.
typedef double stopwatch_t;

// Used to define private functions and variables.
#define PRIVATE static

// This is defined if -fopenmp flag is passed to the compiler.
#if defined(_OPENMP)
#define OMP(x) _Pragma(#x)
#include <omp.h>
#else
#define OMP(x)
static int omp_get_thread_num (void)  {return 0;}
static int omp_get_num_threads (void) {return 1;}
static int omp_get_max_threads (void) {return 1;}
#endif

// Used for bit-based space calculations.
const size_t BITS_PER_BYTE  = 8;
const size_t BYTES_PER_WORD = (sizeof(uint64_t));
const size_t BITS_PER_WORD  = (BYTES_PER_WORD * BITS_PER_BYTE);
inline size_t bits_to_bytes(size_t bits) {
  return (((bits / BITS_PER_WORD) + 1) * BYTES_PER_WORD);
}

// Commonly used communication message sizes.
const size_t MSG_SIZE_ZERO = 0;
const size_t MSG_SIZE_BYTE = BITS_PER_BYTE;
const size_t MSG_SIZE_WORD = sizeof(int) * BITS_PER_BYTE;

// A global seed value.
const int GLOBAL_SEED = 1985;

// A wrapper that asserts the success of totem function calls.
#define CALL_SAFE(func)                         \
  do {                                          \
    error_t err = func;                         \
    assert(err == SUCCESS);                     \
  } while (0)

// Simulates simple exceptions: if the statement is not correct, jump to
// label, typically an error label where you could clean up before exit.
#define CHK(stmt, label)                        \
  do {                                          \
    if (!(stmt))                                \
      goto label;                               \
  } while (0)

// Check if return value of stmt is SUCCESS, jump to label if not.
#define CHK_SUCCESS(stmt, label) CHK((stmt) == SUCCESS, label)

/**
 * Converts a string to upper case.
 * @param[in] str the string to be converted.
 */
inline void to_upper(char* str) {
  assert(str);
  while (*str != '\0') {
    char c = toupper(*str);
    *str = c;
    str++;
  }
}

/**
 * Checks if the string is a numeric number
 * @param[in] str the string to check
 * @return true if the string represents a numeric number
 */
inline bool is_numeric(char* str) {
  assert(str);
  bool numeric = true;
  while (*str != '\0' && numeric) {
    if (!isdigit(*str++)) {
      numeric = false;
    }
  }
  return numeric;
}

/**
 * A single precision atomic add. The built in __sync_add_and_fetch function
 * does not have a floating point version, hence this one.
 * @param[in] address the content is incremented by val
 * @param[in] val the value to be added to the content of address
 * @return old value stored at address
 */
inline float __sync_fetch_and_add_float(float* address, float val) {
  int* address_as_int = reinterpret_cast<int*>(address);
  int old = *address_as_int, assumed;
  do {
    assumed = old;
    float sum = val + *(reinterpret_cast<float*>(&assumed));
    old = __sync_val_compare_and_swap(
        address_as_int, assumed, *(reinterpret_cast<int*>(&sum)));
  } while (assumed != old);
  return *(reinterpret_cast<float*>(&old));
}

/**
 * A double precision atomic add. The built in __sync_add_and_fetch function
 * does not have a floating point version, hence this one.
 * @param[in] address the content is incremented by val
 * @param[in] val the value to be added to the content of address
 * @return old value stored at address
 */
inline double __sync_fetch_and_add_double(double* address, double val) {
  int64_t* address_as_int64 = reinterpret_cast<int64_t*>(address);
  int64_t old = *address_as_int64, assumed;
  do {
    assumed = old;
    double sum = val + (*(reinterpret_cast<double*>(&assumed)));
    old = __sync_val_compare_and_swap(address_as_int64, assumed,
                                      *(reinterpret_cast<int64_t*>(&sum)));
  } while (assumed != old);
  return *(reinterpret_cast<double*>(&old));
}

/**
 * Atomic min for int values. Atomically store the minimum of value at address
 * and val back at address and returns the old value at address.
 * @param[in] address stores the minimum of val and old value at address
 * @param[in] val the value to be compared with
 * @return old value stored at address
 */
inline int __sync_fetch_and_min(int* address, int val) {
  int old = *address, assumed;
  do {
    assumed = old;
    int min = (val < assumed) ? val : assumed;
    old = __sync_val_compare_and_swap(address, assumed, min);
  } while (assumed != old);
  return old;
}

inline uint32_t __sync_fetch_and_min_uint32(uint32_t* address, uint32_t val) {
  uint32_t old = *address, assumed;
  do {
    assumed = old;
    uint32_t min = (val < assumed) ? val : assumed;
    old = __sync_val_compare_and_swap(address, assumed, min);
  } while (assumed != old);
  return old;
}

/**
 * A single precision atomic min. Atomically store the minimum of value at
 * address and val back at address and returns the old value at address.
 * @param[in] address stores the minimum of val and old value at address
 * @param[in] val the value to be compared with
 * @return old value stored at address
 */
inline float __sync_fetch_and_min_float(float* address, float val) {
  int* address_as_int = reinterpret_cast<int*>(address);
  int old = *address_as_int, assumed;
  do {
    assumed = old;
    float assumed_float = *(reinterpret_cast<float*>(&assumed));
    float min = (val < assumed_float) ? val : assumed_float;
    old = __sync_val_compare_and_swap(address_as_int, assumed,
                                      *(reinterpret_cast<int*>(&min)));
  } while (assumed != old);
  return *(reinterpret_cast<float*>(&old));
}

/**
 * A double precision atomic min. Atomically store the minimum of value at
 * address and val back at address and returns the old value at address.
 * @param[in] address stores the minimum of val and old value at address
 * @param[in] val the value to be compared with
 * @return old value stored at address
 */
inline double __sync_fetch_and_min_double(double* address, double val) {
  int64_t* address_as_int64 = reinterpret_cast<int64_t*>(address);
  int64_t old = *address_as_int64, assumed;
  do {
    assumed = old;
    double assumed_double = *(reinterpret_cast<double*>(&assumed));
    double min = (val < assumed_double) ? val : assumed_double;
    old = __sync_val_compare_and_swap(address_as_int64, assumed,
                                      *(reinterpret_cast<int64_t*>(&min)));
  } while (assumed != old);
  return *(reinterpret_cast<double*>(&old));
}

/**
 * Resets the timer to current system time. Called at the moment to
 * start timing an operation.
 * @param[in] stopwatch the stopwatch handler
 */
inline void stopwatch_start(stopwatch_t* stopwatch) {
  struct timeval tval;
  gettimeofday(&tval, NULL);
  *stopwatch = (tval.tv_sec * 1000 + tval.tv_usec/1000.0);
}

/**
 * Returns the elapsed time since the stopwatch started via stopwatch_start
 * @param[in] stopwatch the stopwatch handler
 * @return elapsed time in milliseconds
 */
inline double stopwatch_elapsed(stopwatch_t* stopwatch) {
  struct timeval tval;
  gettimeofday(&tval, NULL);
  return ((tval.tv_sec * 1000 + tval.tv_usec/1000.0) - *stopwatch);
}

// Times the execution time of a function. This is typically used for debugging
// and profiling purposes.
#define STOPWATCH_FUNC(func)                                            \
  do {                                                                  \
    stopwatch_t stopwatch;                                              \
    stopwatch_start(&stopwatch);                                        \
    func;                                                               \
    printf("%s\t%f\n", #func, stopwatch_elapsed(&stopwatch));    \
    fflush(stdout);                                                     \
  } while (0)

#endif  // TOTEM_COMDEF_H
