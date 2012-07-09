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
#include "totem_graph.h"

/**
 * The bitmap data structure. A flat array of 64-bit words.
 */
typedef uint64_t bitmap_word_t;
typedef bitmap_word_t* bitmap_t;
#define BITS_PER_WORD (sizeof(bitmap_word_t) * 8)

/**
 * Allocates and initializes a bitmap data structure
 *
 * @param[in]  len  the length of the bitmap
 * @return an initialized bitmap
*/
inline bitmap_t bitmap_init(id_t len) {
  bitmap_t map = (bitmap_t)calloc((len / BITS_PER_WORD) + 1, 
                                  sizeof(bitmap_word_t));
  assert(map);
  return map;
}

/**
 * Frees the state allocated by the bitmap
 *
 * @param[in] bitmap bitmap to be finalized
*/
inline void bitmap_finalize(bitmap_t map) {
  assert(map);
  free(map);
}

/**
 * Atomically sets a bit to 1 in the bitmap
 *
 * @param[in] bitmap the bitmap to be manipulated
 * @param[in] bit    the bit to be set
 * @return true if the bit is set, false if it was already set
*/
inline bool bitmap_set(bitmap_t map, id_t bit) {
  bitmap_word_t mask = ((bitmap_word_t)1) << (bit % BITS_PER_WORD);
  return !(__sync_fetch_and_or(&(map[bit / BITS_PER_WORD]), mask) & mask);
}

/**
 * Checks if a bit is set.
 *
 * @param[in] bitmap the bitmap to be manipulated
 * @param[in] bit    the bit to be checked
 * @return true if the bit is set, false if not
*/
inline bool bitmap_is_set(bitmap_t map, id_t bit) {
  bitmap_word_t mask = ((bitmap_word_t)1) << (bit % BITS_PER_WORD);
  return (map[bit / BITS_PER_WORD] & mask);
}

#endif // TOTEM_BITMAP_H
