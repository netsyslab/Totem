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
#include "totem_mem.h"

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

__device__ __host__ inline size_t bitmap_bits_to_words(vid_t len) {
  return (len / BITMAP_BITS_PER_WORD + 1);
}

/**
 * Allocates and initializes a bitmap data structure
 * @param[in]  len  the length of the bitmap
 * @return an initialized bitmap
*/
inline bitmap_t bitmap_init_cpu(size_t len) {
  bitmap_t map = reinterpret_cast<bitmap_t>(
      calloc(bitmap_bits_to_bytes(len), 1));
  assert(map);
  return map;
}

inline bitmap_t bitmap_init_gpu(size_t len) {
  bitmap_t map = NULL;
  CALL_CU_SAFE(cudaMalloc(reinterpret_cast<void**>(&map),
                          bitmap_bits_to_bytes(len)));
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

inline void bitmap_reset_gpu(bitmap_t map, size_t len,
                             cudaStream_t stream = 0) {
  CALL_CU_SAFE(cudaMemsetAsync(map, 0, bitmap_bits_to_bytes(len), stream));
}

/**
 * Atomically sets a bit to 1 in the bitmap
 * @param[in] bitmap the bitmap to be manipulated
 * @param[in] bit    the bit to be set
 * @return true if the bit is set, false if it was already set
*/
inline bool bitmap_set_cpu(bitmap_t map, vid_t bit) {
  vid_t word = bit / BITMAP_BITS_PER_WORD;
  bitmap_word_t mask = (bitmap_word_t)1 << (bit - word * BITMAP_BITS_PER_WORD);
  return !(__sync_fetch_and_or(&(map[word]), mask) & mask);
}

__device__ inline bool bitmap_set_gpu(bitmap_t map, vid_t bit) {
  vid_t word = bit / BITMAP_BITS_PER_WORD;
  bitmap_word_t mask = (bitmap_word_t)1 << (bit - word * BITMAP_BITS_PER_WORD);
  return !(atomicOr(&(map[word]), mask) & mask);
}

/**
 * Atomically unsets a bit in the bitmap
 * @param[in] bitmap the bitmap to be manipulated
 * @param[in] bit    the bit to unset
 * @return true if the bit is unset, false if it was already unset
*/
inline bool bitmap_unset_cpu(bitmap_t map, vid_t bit) {
  vid_t word = bit / BITMAP_BITS_PER_WORD;
  bitmap_word_t mask = (bitmap_word_t)1 << (bit - word * BITMAP_BITS_PER_WORD);
  return (__sync_fetch_and_and(&(map[word]), ~mask) & mask);
}

__device__ inline bool bitmap_unset_gpu(bitmap_t map, vid_t bit) {
  vid_t word = bit / BITMAP_BITS_PER_WORD;
  bitmap_word_t mask = (bitmap_word_t)1 << (bit - word * BITMAP_BITS_PER_WORD);
  return (atomicAnd(&(map[word]), ~mask) & mask);
}

/**
 * Checks if a bit is set
 * @param[in] bitmap the bitmap to be checked
 * @param[in] bit    the bit to be checked
 * @return true if the bit is set, false if not
*/
__host__ __device__ inline bool bitmap_is_set(bitmap_t map, vid_t bit) {
  vid_t word = bit / BITMAP_BITS_PER_WORD;
  bitmap_word_t mask = (bitmap_word_t)1 << (bit - word * BITMAP_BITS_PER_WORD);
  return (map[word] & mask);
}

/**
 * Checks if a bit is set in a word
 * @param[in] word the word to be checked
 * @param[in] bit  the bit to be checked
 * @return true if the bit is set, false if not
*/
__host__ __device__ inline bool bitmap_is_set(bitmap_word_t word, vid_t bit) {
  return (word & (((bitmap_word_t)1) << bit));
}

/**
 * Counts the number of set bits in the bitmap
 * @param[in] bitmap the bitmap to be counted
 * @param[in] len the length of the bitmap
 * @param[in] count_d a reference to a sizeof(vid_t) worth of space
 *                    on device memory used to compute the number
 *                    of set bits in the diff bitmap.
 * @param[in] stream the stream within which this computation will be launched
 *
 * @return the number of set bits
*/
vid_t bitmap_count_cpu(bitmap_t bitmap, size_t len);
vid_t bitmap_count_gpu(bitmap_t bitmap, size_t len, vid_t* count_d = NULL,
                       cudaStream_t stream = 0);
void bitmap_count_gpu(bitmap_t bitmap, size_t len, vid_t* count_h,
                      vid_t* count_d, cudaStream_t stream = 0);

/**
 * Diffs the two bitmaps and stores the result back in "diff"
 * @param[in] cur the current visited state bitmap
 * @param[in/out] diff at entry, represents the bitmap of last visited round,
 *                when the function returns, it will represent the diff
 *                between the current and last visited bitmaps.
 * @param[in] len the length of the bitmaps
 * @param[in] stream the stream within which this computation will be launched
*/
void bitmap_diff_cpu(bitmap_t cur, bitmap_t diff, size_t len);
void bitmap_diff_gpu(bitmap_t cur, bitmap_t diff, size_t len,
                     cudaStream_t stream = 0);

/**
 * Copy bitmap
 * @param[in] src the source bitmap
 * @param[out] dst the destination bitmap
 * @param[in] len the length of the bitmaps
 * @param[in] stream the stream within which this computation will be launched
*/
void bitmap_copy_cpu(bitmap_t src, bitmap_t dst, size_t len);
void bitmap_copy_gpu(bitmap_t src, bitmap_t dst, size_t len,
                     cudaStream_t stream = 0);

/**
 * A multi-purpose function that is motivated by traversal-based algorithms
 * to create the frontier bitmap. This function does two things: first, cur and
 * diff are xor-ed and the result is stored back in diff (i.e., diffing cur and
 * the old value in diff); second, copy cur bitmap to "copy".
 * @param[in] cur the current visited state bitmap
 * @param[in/out] diff at entry, represents the bitmap of last visited round,
 *                when the function returns, it will represent the diff
 *                between the current and last visited bitmaps (i.e., the
 *                frontier).
 * @param[out] copy used to backup the cur bitmap
 * @param[in] len the length of the bitmaps
 * @param[in] stream the stream within which this computation will be launched
 *
*/
void bitmap_diff_copy_gpu(bitmap_t cur, bitmap_t diff, bitmap_t copy,
                           size_t len, cudaStream_t stream = 0);
void bitmap_diff_copy_cpu(bitmap_t cur, bitmap_t diff, bitmap_t copy,
                           size_t len);

/**
 * A multi-purpose function that is motivated by traversal-based algorithms
 * to create the frontier bitmap and get the number of vertices in the frontier.
 * This function does three things: first, cur and diff are xor-ed and the
 * result is stored back in diff (i.e., diffing cur and the old value in diff);
 * second, count the number of set bits in diff after the diffing; finally,
 * copy cur bitmap to "copy".
 * @param[in] cur the current visited state bitmap
 * @param[in/out] diff at entry, represents the bitmap of last visited round,
 *                when the function returns, it will represent the diff
 *                between the current and last visited bitmaps (i.e., the
 *                frontier).
 * @param[out] copy used to backup the cur bitmap
 * @param[in] len the length of the bitmaps
 * @param[in] count_d a reference to a sizeof(vid_t) worth of space
 *                    on device memory used to compute the number
 *                    of set bits in the diff bitmap.
 * @param[in] stream the stream within which this computation will be launched
 *
 * @return the number of set bits in diff
*/
vid_t bitmap_diff_copy_count_gpu(bitmap_t cur, bitmap_t diff, bitmap_t copy,
                                 size_t len, vid_t* count_d = NULL,
                                 cudaStream_t stream = 0);
vid_t bitmap_diff_copy_count_cpu(bitmap_t cur, bitmap_t diff, bitmap_t copy,
                                 size_t len);


#endif  // TOTEM_BITMAP_H
