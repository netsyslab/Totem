/**
 *  Common definitions across the modules.
 */

#ifndef TOTEM_COMDEF_H
#define TOTEM_COMDEF_H

// system includes
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

/**
 *  Function return code types.
 */
typedef enum {
   SUCCESS = 0, /**< generic success return code. */
   FAILURE = -1 /**< generic failure return code. */
} error_t;

/**
 *  A wrapper that asserts the success of totem function calls
 */
#define CALL_SAFE(func)				\
    do {					\
	error_t err = func;			\
	assert(err = SUCCESS);			\
    }while(0)

/**
 *  Command line options.
 */
typedef struct options_s {
    char* graph_file;
    bool with_weights;
} options_t;

/**
 *  Used to define local (i.e., private) functions and variables
 */
#define PRIVATE static

#endif  // TOTEM_COMDEF_H

