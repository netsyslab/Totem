/**
 *  Defines a set of convenience kernels
 *
 *  Created on: 2011-03-07
 *  Author: Abdullah Gharaibeh
 */

#ifndef TOTEM_COMKERNEL_CUH
#define TOTEM_COMKERNEL_CUH

// totem includes
#include "totem_comdef.h"

/**
 * a templatized version of memset for device buffers, the assumption is that
 * the caller will dispatch a number of threads at least equal to "size"
 * @param[in] buffer the buffer to be set
 * @param[in] value the value the buffer elements are set to
 * @param[in] size number of elements in the buffer
 */
template<typename T>
__global__ void memset_device(T* buffer, T value, uint32_t size) {

  uint32_t index = THREAD_GLOBAL_INDEX;
  
  if (index >= size) {
    return;
  }

  buffer[index] = value;
}

#endif  // TOTEM_COMKERNEL_CUH
