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
// cuda includes
#include <cuda_runtime_api.h>

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

#endif  // TOTEM_UTIL_H

