/**
 *  CUDA Utilities
 *
 *  Created on: 2011-10-07
 *  Author: Greg Redekop
 */

#ifndef TOTEM_UTIL_H
#define TOTEM_UTIL_H

// totem includes
#include "totem_comdef.h"

/**
 * This defines the minimum required CUDA architecture version, denoted
 * by major_version.minor_version
 */
#define REQ_CUDAVERSION_MAJOR     2
#define REQ_CUDAVERSION_MINOR     0


/**
 * Ensure the device supports the minimum CUDA architecture requirements
 * @return generic success or failure
 */
error_t check_cuda_version();

/**
 * Returns the number of available CUDA-enabled GPUs
 * @return number of CUDA-enabled GPUs
 */
int get_gpu_count();

/**
 * Compares two vertex ids, used by the qsort function
 * @return 0 if equal; 1, if *a > *b; and -1, if *b > *a
 */
int compare_ids(const void *a, const void *b);

#endif  // TOTEM_UTIL_H

