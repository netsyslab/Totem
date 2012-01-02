/**
 * Defines a hash table interface. Mainly the data-structure the insert and 
 * the retrieve operations. Based on the hash table described in [Alcantara09].
 * D.A. Alcantara et al., "Real-time parallel hashing on the GPU," in
 * ACM Transactions on Graphics (TOG).
 * 
 * Note that this implementation does not support inserting individual items
 * to the table, rather it allows for only building the hash table at once
 * (i.e., all (key,value) pairs to be inserted in the table should be known at 
 * initialization time).

 *  Created on: 2011-12-30
 *  Author: Abdullah Gharaibeh
 */
#ifndef TOTEM_HASH_TABLE_H
#define TOTEM_HASH_TABLE_H

// totem includes
#include "totem_comdef.h"
#include "totem_mem.h"

#define HT_SPACE_EXPANSION_RATIO 1.25
#define HT_PRIME_NUMBER          334214459
#define HT_KEY_EMPTY             0
#define HT_MAX_ITERATIONS        100

/** 
 * Macros to manage hash table entries. An entry is a 64-bit that combines a 
 * key (32 higher-order bits) and the index (32 lower-order bits) of the 
 * corresponding value in the values array
*/
#define HT_MAKE_ENTRY(key, value_index) (((uint64_t)key << 32) + (value_index))
#define HT_GET_KEY(entry) ((uint32_t)((entry) >> 32))
#define HT_GET_VALUE_INDEX(entry) ((uint32_t)((entry) & ((uint32_t)-1)))

/** 
 * Weak hash functions following the form (A*KEY + B) mod P % SIZE, where
 * A and B are random numbers, P is a prime number and SIZE is the table size
*/
#define HT_FUNC1(ht, k) (((9600*(k) + 11517) % HT_PRIME_NUMBER) % (ht)->size)
#define HT_FUNC2(ht, k) (((16726*(k) + 6274) % HT_PRIME_NUMBER) % (ht)->size)
#define HT_FUNC3(ht, k) (((8334*(k) + 19108) % HT_PRIME_NUMBER) % (ht)->size)
#define HT_FUNC4(ht, k) (((23720*(k) + 7860) % HT_PRIME_NUMBER) % (ht)->size)

#define HT_GET_VALUE(_ht, _key, _vptr)                                  \
  do {                                                                  \
    uint32_t _location_1 = HT_FUNC1((_ht), (_key));                     \
    uint32_t _location_2 = HT_FUNC2((_ht), (_key));                     \
    uint32_t _location_3 = HT_FUNC3((_ht), (_key));                     \
    uint32_t _location_4 = HT_FUNC4((_ht), (_key));                     \
    uint64_t _entry;                                                    \
    if (HT_GET_KEY(_entry = (_ht)->keys[_location_1]) != (_key))        \
      if (HT_GET_KEY(_entry = (_ht)->keys[_location_2]) != (_key))      \
        if (HT_GET_KEY(_entry = (_ht)->keys[_location_3]) != (_key))    \
          if (HT_GET_KEY(_entry = (_ht)->keys[_location_4]) != (_key)) { \
            value_ptr = NULL;                                           \
            break;                                                      \
          }                                                             \
    T* _values = (T*)(_ht)->values;                                     \
    _vptr = &_values[HT_GET_VALUE_INDEX(_entry)];                       \
  } while(0)

/**
 * Defines a hash table data structure. Note that users are not supposed to 
 * directly manipulate the state.
 * TODO(abdullah): define the struct as a template as well
 */
typedef struct hash_table_s {
  uint32_t  size; /**< the size of the table */
  uint64_t* keys; /**< an entry in the array encodes two things: a key 
                     (higher-order 32-bits) and the index (lower-order 
                     32-bits) of the corresponding item */
  void*     values; /**< the values array */
} hash_table_t;

/**
 * Frees the state allocated for the hash table
 * @param[in] graph a reference to the hash table
 * @return generic success or failure
 */
inline error_t hash_table_finalize_cpu(hash_table_t* hash_table) {
  assert(hash_table && hash_table->keys);
  free(hash_table->keys);
  free(hash_table);
  return SUCCESS;
}

/**
 * Initializes a hash table. It basically allocates the state and builds the 
 * hash table by inserting the key-value entries in the table.
 * @param[in] values an array of values to be inserted in the table
 * @param[in] keys   the corresponding keys of the values
 * @param[in] count  the numbero f values
 * @param[out] hash_table_ret a reference to the created hash table
 * @return generic success or failure
 */
template<typename T>
error_t hash_table_initialize_cpu(T* values, uint32_t* keys, int count,
                                  hash_table_t** hash_table_ret) {
  if (count <= 0 || !values) return FAILURE;

  // allocate hash table state
  hash_table_t* hash_table = (hash_table_t*)calloc(1, sizeof(hash_table_t));
  hash_table->size = count * HT_SPACE_EXPANSION_RATIO;
  hash_table->keys = (uint64_t*)calloc(hash_table->size, sizeof(uint64_t));
  hash_table->values = (void*)values;

  // build the hash table
  for (int index = 0; index < count; index++) {
    uint32_t key   = keys[index];
    uint64_t entry = HT_MAKE_ENTRY(key, index);
    uint32_t location = HT_FUNC1(hash_table, key);
    int its = 0;
    for (; its < HT_MAX_ITERATIONS; its++) {
      // Insert the new item
      entry = __sync_lock_test_and_set(&(hash_table->keys[location]), entry);
      key = HT_GET_KEY(entry);
      if (key == HT_KEY_EMPTY) break;
      
      // if an item was evicted, reinsert it again
      uint32_t location_1 = HT_FUNC1(hash_table, key);
      uint32_t location_2 = HT_FUNC2(hash_table, key);
      uint32_t location_3 = HT_FUNC3(hash_table, key);
      uint32_t location_4 = HT_FUNC4(hash_table, key);
      if (location == location_1) location = location_2;
      else if (location == location_2) location = location_3;
      else if (location == location_3) location = location_4;
      else location = location_1;
    }
    // the eviction chain was too long, can't build the hash table
    if (its == HT_MAX_ITERATIONS) {
      hash_table_finalize_cpu(hash_table);
      return FAILURE;
    }
  }
  *hash_table_ret = hash_table;
  return SUCCESS;
}

/**
 * Retrieves a reference to an item of the passed key.
 * @param[in] hash_table a reference to the hash table
 * @param[in] key the key to look up
 * @param[in] value_ptr a reference to the looked up entry
 * @return SUCCESS if found, FAILURE otherwise
 */
template<typename T>
error_t hash_table_get_cpu(hash_table_t* hash_table, uint32_t key, 
                           T** value_ptr) {
  HT_GET_VALUE(hash_table, key, *value_ptr);
  if (!(*value_ptr)) return FAILURE;
  return SUCCESS;
}

/**
 * Retrieves a group of values from the table. This function is offered 
 * mainly to facilitate easy unit testing of the retrieve functionality.
 * @param[in] hash_table a reference to the hash table
 * @param[in] keys a reference to the group of keys to be looked up
 * @param[in] count number of keys
 * @param[out] values_ret a reference the list of looked up values
 * @return SUCCESS if all are found, FAILURE otherwise
 */
template<typename T>
error_t hash_table_get_cpu(hash_table_t* hash_table, uint32_t* keys, int count,
                           T** values_ret) {
  T* values = (T*)mem_alloc(count * sizeof(T));
  for (int k = 0; k < count; k++) {
    T* value = NULL;
    if (hash_table_get_cpu(hash_table, keys[k], &value) != SUCCESS) {
      free(values);
      return FAILURE;
    }
    memcpy(&values[k], value, sizeof(T));
  }
  *values_ret = values;
  return SUCCESS;
}

#endif  // TOTEM_HASH_TABLE_H
