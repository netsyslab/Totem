/**
 * Implements the sorting functions for Binary Radixsort
 *
 *  Created on: 2014-08-02
 *  Author: Daniel Lucas dos Santos Borges
 */
#include "totem_radix_sort.h"

/**
 * Partition the array of vdegree_t elements based on binary
 * radix-sort strategy. 
 * 
 * @param[in] array Reference to the array to be partitioned
 * @param[in] start Starting index of the sub-array
 * @param[in] end Ending index of the sub-array
 * @param[in] mask Bit-mask used to compare elements via bit-wise operations
 * @param[in] bin0_ret Reference to the limit of the 0's bin 
 * @param[in] bin1_ret Reference to the limit of the 1's bin
 */
PRIVATE void radix_sort_partition(vdegree_t* array, int64_t start, int64_t end,
                                  uint64_t mask, int64_t* bin0_ret,
                                  int64_t* bin1_ret) {
  int64_t bin0 = start;
  int64_t bin1 = end;
  while (bin0 != bin1) {
    if ((array[bin0].degree & mask) == mask) {
      swap(&array[bin1], &array[bin0]);
      bin1--;
    } else {
      bin0++;
    }
  }
  if ((array[bin0].degree & mask) == mask) {
    bin0--;
  } else {
    bin1++;
  }

  *bin0_ret = bin0;
  *bin1_ret = bin1;
}

/**
 * Partition the array of vdegree_t elements based on binary
 * radix-sort strategy. 
 * 
 * @param[in] array Reference to the array to be partitioned
 * @param[in] start Starting index of the sub-array
 * @param[in] end Ending index of the sub-array
 * @param[in] mask Bit-mask used to compare elements via bit-wise operations
 * @param[in] bin0_ret Reference to the limit of the 0's bin 
 * @param[in] bin1_ret Reference to the limit of the 1's bin
 */
PRIVATE void radix_sort_partition(vid_t* array, int64_t start, int64_t end,
                                  uint64_t mask, int64_t* bin0_ret,
                                  int64_t* bin1_ret) {
  int64_t bin0 = start;
  int64_t bin1 = end;
  while (bin0 != bin1) {
    if ((array[bin0] & mask) == mask) {
      swap(&array[bin1], &array[bin0]);
      bin1--;
    } else {
      bin0++;
    }
  }
  if ((array[bin0] & mask) == mask) {
    bin0--;
  } else {
    bin1++;
  }
  *bin0_ret = bin0;
  *bin1_ret = bin1;
}

/**
 * Approximately sorts an array of vid_t elements using the 
 * binary radix-sort algorithm.
 * This function is first called by the parallel_radix_sort recursion wrapper
 *
 * @param[in] array Reference to the array to be sorted
 * @param[in] start Starting index of the sub-array
 * @param[in] end Ending index of the sub-array
 * @param[in] depth Current recursion level
 * @param[in] max_depth Maximum level the recusion can reach, also represents 
 *                      how many bit are being used as precision.
 * @param[in] mask Bit-mask used to compare elements via bit-wise operations 
 */

PRIVATE void binary_radix_sort(vid_t* array, int64_t start, int64_t end,
                               int64_t depth, int max_depth, uint64_t mask) {
  if (start < end && (depth <= max_depth) && (mask != 0)) {
    int64_t bin0, bin1;
    radix_sort_partition(array, start, end, mask, &bin0, &bin1);
    binary_radix_sort(array, start, bin0, depth + 1, max_depth, mask >> 1);
    binary_radix_sort(array, bin1, end, depth + 1, max_depth, mask >> 1);
  }
}

/**
 * Approximately sorts an array of vid_t elements using the 
 * parallel version of binary radix-sort algorithm.
 * This function is first called by the parallel_radix_sort recursion wrapper
 *
 * @param[in] array Reference to the array to be sorted
 * @param[in] start Starting index of the sub-array
 * @param[in] end Ending index of the sub-array
 * @param[in] depth Current recursion level
 * @param[in] max_depth Maximum level the recusion can reach, also represents 
 *                      how many bit are being used as precision.
 * @param[in] mask Bit-mask used to compare elements via bit-wise operations 
 */

PRIVATE void parallel_binary_radix_sort(vid_t* array, int64_t start,
                                        int64_t end, int depth, int max_depth,
                                        uint64_t mask) {
  if (start < end && (depth <= max_depth) && (mask != 0)) {
    int64_t bin0, bin1;
    radix_sort_partition(array, start, end, mask, &bin0, &bin1);
    OMP(omp parallel sections) {
      OMP(omp section)
      binary_radix_sort(array, start, bin0, depth + 1, max_depth, mask >> 1);

      OMP(omp section)
      binary_radix_sort(array, bin1, end, depth + 1, max_depth, mask >> 1);
    }
  }
}

/**
 * Approximately sorts an array of vdegree_t elements using the 
 * binary radix-sort algorithm.
 * This function is first called by the parallel_radix_sort recursion wrapper
 *
 * @param[in] array Reference to the array to be sorted
 * @param[in] start Starting index of the sub-array
 * @param[in] end Ending index of the sub-array
 * @param[in] depth Current recursion level
 * @param[in] max_depth Maximum level the recusion can reach, also represents 
 *                      how many bit are being used as precision.
 * @param[in] mask Bit-mask used to compare elements via bit-wise operations 
 */
PRIVATE void binary_radix_sort(vdegree_t* array, int64_t start, int64_t end,
                               int64_t depth, int max_depth, uint64_t mask) {
  if (start < end && (depth <= max_depth) && (mask != 0)) {
    int64_t bin0, bin1;
    radix_sort_partition(array, start, end, mask, &bin0, &bin1);
    binary_radix_sort(array, start, bin0, depth + 1, max_depth, mask >> 1);
    binary_radix_sort(array, bin1, end, depth + 1, max_depth, mask >> 1);
  }
}

/**
 * Approximately sorts an array of vdegree_t elements using the 
 * parallel version of binary radix-sort algorithm.
 * This function is first called by the parallel_radix_sort recursion wrapper
 *
 * @param[in] array Reference to the array to be sorted
 * @param[in] start Starting index of the sub-array
 * @param[in] end Ending index of the sub-array
 * @param[in] depth Current recursion level
 * @param[in] max_depth Maximum level the recusion can reach, also represents 
 *                      how many bit are being used as precision.
 * @param[in] mask Bit-mask used to compare elements via bit-wise operations 
 */

PRIVATE void parallel_binary_radix_sort(vdegree_t* array, int64_t start,
                                        int64_t end, int depth,
                                        int max_depth, uint64_t mask) {
  if (start < end && (depth <= max_depth) && (mask != 0)) {
    int64_t bin0, bin1;
    radix_sort_partition(array, start, end, mask, &bin0, &bin1);
    OMP(omp parallel sections) {
      OMP(omp section)
      binary_radix_sort(array, start, bin0, depth + 1, max_depth, mask >> 1);

      OMP(omp section)
      binary_radix_sort(array, bin1, end, depth + 1, max_depth, mask >> 1);
    }
  }
}

void parallel_radix_sort(vid_t* array, size_t num, size_t size,
                         bool order, int sorting_precision) {
  uint64_t mask = (uint64_t)1 << (size * 8 - 2);
  parallel_binary_radix_sort(array, 0, num-1, 0, sorting_precision,
                             mask);
  if (!order) {
    reverse(array, num);
  }
}

void parallel_radix_sort(vdegree_t* array, size_t num, size_t size,
                         bool order, int sorting_precision) {
  uint64_t mask = static_cast<uint64_t>(1) << (size * 8 - 2);
  parallel_binary_radix_sort(array, 0, num-1, 0, sorting_precision,
                            mask);
  if (!order) {
    reverse(array, num);
  }
}

