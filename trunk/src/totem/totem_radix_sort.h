/**
 * Headers for the binary radixsort functions.
 *
 *  Created on: 2014-08-02
 *  Author: Daniel Lucas dos Santos Borges
 */
#include "totem_graph.h"
#include "totem_util.h"
/**
 * Sorts an array using the binary radix-sort algorithm, 
 * which can be a full or approximate sorting
 *
 * @param[in] array Reference to the array to be sorted
 * @param[in] num Number of elements in the array
 * @param[in] size Size of element's type
 * @param[in] order Boolean to specify if the sorting is in 
 *                  ascending or descending order:
 *                  true for ascending, false for descending
 * @param[in] sorting_precision Indicates the number of bits to use 
 *                             during sorting, starting from the Most 
 *                             Significant bit
 */

void parallel_radix_sort(vdegree_t* array, size_t num, size_t size,
                         bool order,
                         int sorting_precision = sizeof(vid_t) * 8);
void parallel_radix_sort(vid_t* array, size_t num, size_t size,
                         bool order,
                         int sorting_precision = sizeof(vid_t) * 8);

