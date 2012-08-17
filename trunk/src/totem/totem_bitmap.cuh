/**
 * A thread-safe bitmap implementation 
 *
 *  Created on: 2012-07-06
 *  Author: Abdullah Gharaibeh
 */

#ifndef TOTEM_BITMAP_H
#define TOTEM_BITMAP_H

// totem includes
#include "totem_comdef.h"
#include "totem_comkernel.cuh"
#include "totem_graph.h"

/**
 * The bitmap data structure. A flat array of words.
 */
typedef uint32_t bitmap_word_t;
typedef bitmap_word_t* bitmap_t;

/**
 * Common bitmap helper constants and functions
 */
const size_t BITMAP_BITS_PER_WORD = (sizeof(bitmap_word_t) * BITS_PER_BYTE);

__device__ __host__ inline bitmap_word_t bitmap_bit_mask(vid_t bit) {
  return (((bitmap_word_t)1) << (bit % BITMAP_BITS_PER_WORD));
}

__device__ __host__ inline size_t bitmap_bits_to_bytes(vid_t len) {
  return ((len / BITMAP_BITS_PER_WORD + 1) * sizeof(bitmap_word_t));
}

/**
 * Allocates and initializes a bitmap data structure
 * @param[in]  len  the length of the bitmap
 * @return an initialized bitmap
*/
inline bitmap_t bitmap_init_cpu(size_t len) {
  bitmap_t map = (bitmap_t)calloc(bitmap_bits_to_bytes(len), 1);
  assert(map);
  return map;
}

inline bitmap_t bitmap_init_gpu(size_t len) {
  bitmap_t map = NULL;
  CALL_CU_SAFE(cudaMalloc((void**)&(map), bitmap_bits_to_bytes(len)));
  CALL_CU_SAFE(cudaMemset(map, 0, bitmap_bits_to_bytes(len)));
  return map;
}

/**
 * Frees the state allocated by the bitmap
 * @param[in] bitmap bitmap to be finalized
*/
inline void bitmap_finalize_cpu(bitmap_t map) {
  assert(map);
  free(map);
}

inline void bitmap_finalize_gpu(bitmap_t map) {
  assert(map);
  cudaFree(map);
}

/**
 * Clears all the bits 
 * @param[in] bitmap bitmap to be reset
*/
inline void bitmap_reset_cpu(bitmap_t map, size_t len) {
  memset(map, 0, bitmap_bits_to_bytes(len));
}

inline void bitmap_reset_gpu(bitmap_t map, size_t len) {
  CALL_CU_SAFE(cudaMemset(map, 0, bitmap_bits_to_bytes(len)));
}

/**
 * Atomically sets a bit to 1 in the bitmap
 * @param[in] bitmap the bitmap to be manipulated
 * @param[in] bit    the bit to be set
 * @return true if the bit is set, false if it was already set
*/
inline bool bitmap_set_cpu(bitmap_t map, vid_t bit) {
  bitmap_word_t mask = bitmap_bit_mask(bit);
  return !(__sync_fetch_and_or(&(map[bit / BITMAP_BITS_PER_WORD]), 
                               mask) & mask);
}

__device__ inline bool bitmap_set_gpu(bitmap_t map, vid_t bit) {
  bitmap_word_t mask = bitmap_bit_mask(bit);
  return !(atomicOr(&(map[bit / BITMAP_BITS_PER_WORD]), mask) & mask);
}

/**
 * Atomically unsets a bit in the bitmap
 * @param[in] bitmap the bitmap to be manipulated
 * @param[in] bit    the bit to unset
 * @return true if the bit is unset, false if it was already unset
*/
inline bool bitmap_unset_cpu(bitmap_t map, vid_t bit) {
  bitmap_word_t mask = bitmap_bit_mask(bit);
  return (__sync_fetch_and_and(&(map[bit / BITMAP_BITS_PER_WORD]), 
                               ~mask) & mask);
}

__device__ inline bool bitmap_unset_gpu(bitmap_t map, vid_t bit) {
  bitmap_word_t mask = bitmap_bit_mask(bit);
  return (atomicAnd(&(map[bit / BITMAP_BITS_PER_WORD]), ~mask) & mask);
}

/**
 * Checks if a bit is set
 * @param[in] bitmap the bitmap to be manipulated
 * @param[in] bit    the bit to be checked
 * @return true if the bit is set, false if not
*/
__host__ __device__ inline bool bitmap_is_set(bitmap_t map, vid_t bit) {
  return (map[bit / BITMAP_BITS_PER_WORD] & bitmap_bit_mask(bit));
}

#endif // TOTEM_BITMAP_H
