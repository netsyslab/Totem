/**
 * Defines a hash table interface. Mainly the data-structure the build and
 * the retrieve operations. Based on the hash table described in [Alcantara09].
 * D.A. Alcantara et al., "Real-time parallel hashing on the GPU," in
 * ACM Transactions on Graphics (TOG).
 *
 * Two important notes:
 * 1) Only one type of value is supported: int32_t. To use other types, the
 * integer value can be used as an index in another array of the other type,
 * which we anticipate to be the common use-case.
 *
 * 2) this implementation does not support inserting individual items
 * to the table for the gpu version, it allows for only building the hash table
 * at once (i.e., all (key,value) pairs to be inserted in the table should be
 * known at initialization time). But allows for individual retrieval of items.
 *
 *  Created on: 2011-12-30
 *  Author: Abdullah Gharaibeh
 */
#ifndef TOTEM_HASH_TABLE_H
#define TOTEM_HASH_TABLE_H

// totem includes
#include "totem_comdef.h"

/**
 * Macros to manage hash table entries. An entry is a 64-bit that combines a
 * key (32 higher-order bits) and the value (32 lower-order bits)
*/
#define HT_MAKE_ENTRY(_key, _value) (((uint64_t)_key << 32) + (_value))
#define HT_GET_KEY(_entry) ((uint32_t)((_entry) >> 32))
#define HT_GET_VALUE(_entry) ((uint32_t)((_entry) & ((uint32_t)-1)))

/**
 * A prime number used in the hash functions
 */
#define HT_PRIME_NUMBER          334214459

/**
 * A special key used to indicate an empty spot
 */
#define HT_KEY_EMPTY             ((uint32_t)(-1))

/**
 * Weak hash functions following the form (OP(A,KEY) + B) mod P mod SIZE, where
 * OP is either a multiply or an XOR operation. A and B are random numbers, P is
 * a prime number and SIZE is the table size
*/
#define HT_FUNC1(ht, k) (((9600^(k)) + 11517) % HT_PRIME_NUMBER % (ht)->size)
#define HT_FUNC2(ht, k) (((16726*(k)) + 6274) % HT_PRIME_NUMBER % (ht)->size)
#define HT_FUNC3(ht, k) (((8334^(k)) + 19108) % HT_PRIME_NUMBER % (ht)->size)
#define HT_FUNC4(ht, k) (((23720*(k)) + 7860) % HT_PRIME_NUMBER % (ht)->size)

/**
 * Macro to look up a value index from the hash table. This macro can be used
 * for both, the CPU and the GPU-hosted tables
 */
#define HT_LOOKUP(_ht, _key, _value)                                    \
  do {                                                                  \
    uint32_t _location_1 = HT_FUNC1((_ht), (_key));                     \
    uint32_t _location_2 = HT_FUNC2((_ht), (_key));                     \
    uint32_t _location_3 = HT_FUNC3((_ht), (_key));                     \
    uint32_t _location_4 = HT_FUNC4((_ht), (_key));                     \
    uint64_t _entry;                                                    \
    if (HT_GET_KEY(_entry = (_ht)->entries[_location_1]) != (_key))     \
      if (HT_GET_KEY(_entry = (_ht)->entries[_location_2]) != (_key))   \
        if (HT_GET_KEY(_entry = (_ht)->entries[_location_3]) != (_key)) \
          if (HT_GET_KEY(_entry = (_ht)->entries[_location_4]) != (_key)) { \
            _value = -1;                                                \
            break;                                                      \
          }                                                             \
    _value = HT_GET_VALUE(_entry);                                      \
  } while(0)

/**
 * Macro to check if a key exists
 */
#define HT_CHECK(_ht, _key, _found)                                     \
  do {                                                                  \
    uint32_t _location_1 = HT_FUNC1((_ht), (_key));                     \
    if (HT_GET_KEY((_ht)->entries[_location_1]) != (_key)) {            \
      uint32_t _location_2 = HT_FUNC2((_ht), (_key));                   \
      if (HT_GET_KEY((_ht)->entries[_location_2]) != (_key)) {          \
        uint32_t _location_3 = HT_FUNC3((_ht), (_key));                 \
        if (HT_GET_KEY((_ht)->entries[_location_3]) != (_key)) {        \
          uint32_t _location_4 = HT_FUNC4((_ht), (_key));               \
          if (HT_GET_KEY((_ht)->entries[_location_4]) != (_key)) {      \
            _found = false;                                             \
            break;                                                      \
          }                                                             \
        }                                                               \
      }                                                                 \
    }                                                                   \
    _found = true;                                                      \
  } while(0)

/**
 * Defines a hash table data structure. Note that users are not supposed to
 * directly manipulate the state.
 */
typedef struct hash_table_s {
  uint32_t  size;       /**< the size of the table */
  uint64_t* entries;    /**< an entry in the array encodes two things: a key
                           (higher-order 32-bits) and the value (lower-order
                           32-bits) */
  bool      allocated;  /**< indicates whether the hash_table is allocated
                           by the interface or not */
} hash_table_t;

/**
 * Initializes a hash table. The function allocates the hash_table struct.
 * @param[in] count  the number of values
 * @param[out] hash_table a reference to the created hash table
 * @return generic success or failure
 */
error_t hash_table_initialize_cpu(uint32_t count, hash_table_t** hash_table);

/**
 * Initializes a hash table. The hash table struct is allocated by the caller
 * @param[in] count  the number of values
 * @param[out] hash_table a reference to the created hash table
 * @return generic success or failure
 */
error_t hash_table_initialize_cpu(uint32_t count, hash_table_t* hash_table);

/**
 * Overloaded initialization. It allocates the state and builds the hash table
 * by inserting the key-value entries in the table. The assumption is that the
 * value of a key is the key's index in the keys array.
 * @param[in] values an array of values to be inserted in the table
 * @param[in] keys   the corresponding keys of the values
 * @param[in] count  the numbero f values
 * @param[out] hash_table_ret a reference to the created hash table
 * @return generic success or failure
 */
error_t hash_table_initialize_cpu(uint32_t* keys, uint32_t count,
                                  hash_table_t** hash_table);

/**
 * Frees the state allocated for the hash table
 * @param[in] hash_table a reference to the hash table
 * @return generic success or failure
 */
error_t hash_table_finalize_cpu(hash_table_t* hash_table);

/**
 * Inserts a (key, value) to the hash table
 * @param[in] graph a reference to the hash table
 * @param[in] key key to be inserted
 * @param[in] index a value's index to accompany the key
 * @return generic success or failure
 */
error_t hash_table_put_cpu(hash_table_t* hash_table, uint32_t key, int value);

/**
 * Retrieves the value for the corresponding key.
 * @param[in] hash_table a reference to the hash table
 * @param[in] key the key to look up
 * @param[out] value a reference to the looked up entry, or NULL if the key
               is not found in the table.
 * @return SUCCESS if found, FAILURE otherwise
 */
error_t hash_table_get_cpu(hash_table_t* hash_table, uint32_t key, int* value);

/**
 * Retrieves a list of values from the table.
 * @param[in] hash_table a reference to the hash table
 * @param[in] keys a reference to the group of keys to be looked up
 * @param[in] count number of keys
 * @param[out] values the list of values retrieved (allocated via mem_alloc)
 * @return SUCCESS if all are found, FAILURE otherwise
 */
error_t hash_table_get_cpu(hash_table_t* hash_table, uint32_t* keys,
                           uint32_t count, int** values);

/**
 * Retrieves the list of keys in the hash table
 * @param[in] hash_table a reference to the hash table
 * @param[out] keys a reference to the list of returned keys
 * @param[out] count number of keys
 * @return SUCCESS if all are found, FAILURE otherwise
 */
error_t hash_table_get_keys_cpu(hash_table_t* hash_table, uint32_t** keys,
                                uint32_t* count);

/**
 * Initializes a hash table on the gpu. It allocates the state on the cpu and
 * moves it to the gpu
 * @param[in] values an array of values to be inserted in the table
 * @param[in] keys   the corresponding keys of the values
 * @param[in] count  the numbero f values
 * @param[out] hash_table_ret a reference to the created hash table
 * @return generic success or failure
 */
error_t hash_table_initialize_gpu(uint32_t* keys, uint32_t count,
                                  hash_table_t** hash_table);

/**
 * Initializes a hash table on the gpu from a hash table that already exists
 * on the host. The structure of the newly created hash table is allocated
 * by the function.
 * @param[in] hash_table a reference to the host hash table
 * @param[out] hash_table_d a reference to the created hash table on the gpu
 * @return generic success or failure
 */
error_t hash_table_initialize_gpu(hash_table_t* hash_table,
                                  hash_table_t** hash_table_d);

/**
 * Initializes a hash table on the gpu from a hash table that already exists
 * on the host. The structure of the newly created hash table is allocated
 * by the caller.
 * @param[in] hash_table a reference to the host hash table
 * @param[out] hash_table_d a reference to the created hash table on the gpu
 * @return generic success or failure
 */
error_t hash_table_initialize_gpu(hash_table_t* hash_table,
                                  hash_table_t* hash_table_d);

/**
 * Frees the state allocated for the gpu-based hash table
 * @param[in] hash_table a reference to the hash table
 * @return generic success or failure
 */
error_t hash_table_finalize_gpu(hash_table_t* hash_table);

/**
 * Retrieves a group of values from the gpu-based table.
 * @param[in] hash_table a reference to the hash table
 * @param[in] keys the list of keys to be looked up
 * @param[in] count number of keys
 * @param[out] values the list of retrieved values (allocated via mem_alloc)
 * @return SUCCESS if all are found, FAILURE otherwise
 */
error_t hash_table_get_gpu(hash_table_t* hash_table, uint32_t* keys,
                           uint32_t count, int** values);

#endif  // TOTEM_HASH_TABLE_H
