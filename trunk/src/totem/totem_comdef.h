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
#include <float.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/time.h>
#include <unistd.h>

/**
 *  Function return code types
 */
typedef enum {
  SUCCESS = 0, /**< generic success return code */
  FAILURE = -1 /**< generic failure return code */
} error_t;

/**
 * Processor types. This is used as to identify the the processor on which
 * a partition is processed on.
 */
typedef enum {
  PROCESSOR_CPU = 0, /**< CPU processor */
  PROCESSOR_GPU,     /**< GPU processor */
  PROCESSOR_MAX      /**< Indicates the number of supported processor types */
} processor_type_t;

typedef struct processor_s {
  processor_type_t type; /**< Processor type (CPU or GPU). */
  uint32_t         id;   /**< Used to id GPU devices */
} processor_t;

/**
 * Command line options
 */
typedef struct options_s {
  char* graph_file;
  bool  weighted;
  id_t  source;
  id_t  begin_id;
  id_t  end_id;
  char* initial_rank_file;
} options_t;

/**
 * Stopwatch (timer) type
 */
typedef double stopwatch_t;

/**
 * Used to define private functions and variables
 */
#define PRIVATE static

/**
 * A wrapper that asserts the success of totem function calls
 */
#define CALL_SAFE(func)                         \
  do {                                          \
    error_t err = func;                         \
    assert(err == SUCCESS);                     \
  } while(0)

/**
 * Simulates simple exceptions: if the statement is not correct, jump to
 * label, typically an error label where you could clean up before exit.
 */
#define CHK(stmt, label)                        \
  do {                                          \
    if (!(stmt))                                \
      goto label;                               \
  } while(0)

/**
 * Check if return value of stmt is SUCCESS, jump to label if not.
 */
#define CHK_SUCCESS(stmt, label) CHK((stmt) == SUCCESS, label)

/**
 * Converts the string to upper case
 * @param[in] str the string to change to upper case
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
  int* address_as_int = (int*)address;
  int old = *address_as_int, assumed;
  do {
    assumed = old;
    float sum = (val + *((float*)&assumed));
    old = __sync_val_compare_and_swap(address_as_int, assumed, *((int*)&sum));
  } while (assumed != old);
  return *((float *)&old);
}

/**
 * A double precision atomic add. The built in __sync_add_and_fetch function
 * does not have a floating point version, hence this one.
 * @param[in] address the content is incremented by val
 * @param[in] val the value to be added to the content of address
 * @return old value stored at address
 */
inline double __sync_fetch_and_add_double(double* address, double val) {
  int64_t* address_as_int64 = (int64_t*)address;
  int64_t old = *address_as_int64, assumed;
  do {
    assumed = old;
    double sum = val + (*((double*)&assumed));
    old = __sync_val_compare_and_swap(address_as_int64, assumed, 
                                      *((int64_t*)&sum));
  } while (assumed != old);
  return *((double *)&old);
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

/**
 * A single precision atomic min. Atomically store the minimum of value at 
 * address and val back at address and returns the old value at address.
 * @param[in] address stores the minimum of val and old value at address
 * @param[in] val the value to be compared with
 * @return old value stored at address
 */
inline float __sync_fetch_and_min_float(float* address, float val) {
  int* address_as_int = (int*)address;
  int old = *address_as_int, assumed;
  do {
    assumed = old;
    float assumed_float = *((float*)&assumed);
    float min = (val < assumed_float) ? val : assumed_float;
    old = __sync_val_compare_and_swap(address_as_int, assumed, *((int*)&min));
  } while (assumed != old);
  return *((float *)&old);
}

/**
 * A double precision atomic min. Atomically store the minimum of value at 
 * address and val back at address and returns the old value at address.
 * @param[in] address stores the minimum of val and old value at address
 * @param[in] val the value to be compared with
 * @return old value stored at address
 */
inline double __sync_fetch_and_min_double(double* address, double val) {
  int64_t* address_as_int64 = (int64_t*)address;
  int64_t old = *address_as_int64, assumed;
  do {
    assumed = old;
    double assumed_double = *((double*)&assumed);
    double min = (val < assumed_double) ? val : assumed_double;
    old = __sync_val_compare_and_swap(address_as_int64, assumed, 
                                      *((int64_t*)&min));
  } while (assumed != old);
  return *((double *)&old);
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

#endif  // TOTEM_COMDEF_H
