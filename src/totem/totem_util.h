/**
 *  Utility functions.
 *
 *  Created on: 2011-10-07
 *  Author: Greg Redekop
 */

#ifndef TOTEM_UTIL_H
#define TOTEM_UTIL_H

// totem includes
#include "totem_comdef.h"
#include "totem_graph.h"

/**
 * This defines the minimum required CUDA architecture version, denoted
 * by major_version.minor_version.
 */
#define REQ_CUDAVERSION_MAJOR     2
#define REQ_CUDAVERSION_MINOR     0

/**
 * Swap the value of two elements.
 *
 * @param[in] a First element of the swap
 * @param[in] b Second element of the swap
 */
template<typename T>
inline void swap(T* a, T* b) {
  T temp = *a;
  *a = *b;
  *b = temp;
}

/**
 * Reverses an array.
 *
 * @param[in] array Array to be reversed
 * @param[in] len Length of the array
 */
template<typename T>
inline void reverse(T* array, size_t len) {
  OMP(omp parallel for)
  for (int64_t start = 0; start < len / 2; start++) {
    int64_t end = len - 1 - start;
    swap(&array[start], &array[end]);
  }
}

/**
 * Ensure the device supports the minimum CUDA architecture requirements.
 * @return generic success or failure
 */
error_t check_cuda_version();

/**
 * Returns the number of available CUDA-enabled GPUs.
 * @return number of CUDA-enabled GPUs
 */
int get_gpu_count();

/**
 * Returns the total amount of global memory available on device 0 in bytes.
 * @return amount of global memory in bytes
 */
size_t get_gpu_device_memory();

/**
 * Compares two vertex ids, used by the qsort function.
 * @return 0 if equal; 1, if *a > *b; and -1, if *b > *a
 */
int compare_ids_asc(const void* a, const void* b);
int compare_ids_dsc(const void* a, const void* b);

/**
 * Compares two vertex ids, used by the tbb parallel qsort function.
 * @return true if v1 should appear before v2, false otherwise
 */
bool compare_ids_tbb(const vid_t& v1, const vid_t& v2);

/**
 * Returns the index of the most significant set bit in a word.
 * @param[in] word the word to operate on
 * @param[out] the most significant set bit, -1 if word is 0
 */
int get_mssb(uint32_t word);

#endif  // TOTEM_UTIL_H
