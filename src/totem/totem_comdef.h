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
  SUCCESS = 0, /**< generic success return code. */
  FAILURE = -1 /**< generic failure return code. */
} error_t;

/**
 * Command line options
 */
typedef struct options_s {
  char* graph_file;
  bool  weighted;
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
