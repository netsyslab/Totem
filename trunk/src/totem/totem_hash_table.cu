/**
 * Defines a hash table interface. Mainly the data-structure the insert and the
 * retrieve operations. Based on the hash table described in [Alcantara09].
 * D.A. Alcantara et al., "Real-time parallel hashing on the GPU," in ACM
 * Transactions on Graphics (TOG).
 *
 * Note that this implementation does not support inserting individual items to
 * the table, rather it allows for only building the hash table at once (i.e.,
 * all (key,value) pairs to be inserted in the table should be known at
 * initialization time).

 *  Created on: 2011-12-30
 *  Author: Abdullah Gharaibeh
 */

// totem includes
#include "totem_comdef.h"
#include "totem_comkernel.cuh"
#include "totem_hash_table.h"
#include "totem_mem.h"
#include "totem_util.h"

/**
 * Maximum length of the eviction (collision) chain during the key insertion
 * operation.
 */
#define HT_MAX_ITERATIONS 1000

/**
 * Stores prime numbers that are used to determine the actual size of the array
 * used to implement the hash table.
 */
PRIVATE uint32_t primes[] = {
  1,          /* 1 item */
  7,          /* 2 items */
  23,         /* 4 items */
  29,         /* 8 items */
  53,         /* 16 items */
  97,         /* 32 items */
  193,        /* 64 items */
  389,        /* 128 items */
  769,        /* 256 items */
  1543,       /* 512 items */
  3079,       /* 1024 items */
  6151,       /* 2048 items */
  12289,      /* 4096 items */
  24593,      /* 8192 items */
  49157,      /* 16K items */
  98317,      /* 32K items */
  196613,     /* 64K items */
  393241,     /* 128K items */
  786433,     /* 256K items */
  1572869,    /* 512K items */
  3145739,    /* 1M items */
  6291469,    /* 2M items */
  12582917,   /* 4M items */
  25165843,   /* 8M items */
  50331653,   /* 16M items */
  100663319,  /* 32M items */
  201326611,  /* 64M items */
  402653189,  /* 128M items */
  805306457,  /* 256M items */
  1610612741, /* 512M items */
  3143102429  /* 1B items */
};

/**
 * Returns the size of the array that should be allocated to store the hash
 * table.
 * @param[in] count maximum number of elements to be stored in the hash table
 * @param[out] the size of the array in words
 */
PRIVATE uint32_t get_size(uint32_t count) {
  if (count == 0 || get_mssb(count) > 30) return 0;
  int mssb = get_mssb(count);
  int index = 0;
  if (mssb > 0) {
    // Test if the bit to the right of the most significant one is set. If so,
    // then "count" is in the second half of the range covered by the entry
    // indexed by "mssb", and hence the next bracket is chosen instead to
    // double the table size.
    index = (count & (1 << (mssb - 1))) ? mssb + 1 : mssb;
  }
  return primes[index];
}

error_t hash_table_initialize_cpu(uint32_t count, hash_table_t* hash_table) {
  hash_table->size = get_size(count);
  if (hash_table->size == 0) return FAILURE;
  hash_table->allocated = false;
  hash_table->entries = (uint64_t*)calloc(hash_table->size, sizeof(uint64_t));
  memset(hash_table->entries, -1, hash_table->size * sizeof(uint64_t));
  return SUCCESS;
}

error_t hash_table_initialize_cpu(uint32_t count, hash_table_t** hash_table) {
  *hash_table = (hash_table_t*)calloc(1, sizeof(hash_table_t));
  error_t rc = hash_table_initialize_cpu(count, *hash_table);
  (*hash_table)->allocated = true;
  return rc;
}

error_t hash_table_initialize_cpu(uint32_t* keys, uint32_t count,
                                  hash_table_t** hash_table) {
  if (count <= 0 || !keys) return FAILURE;
  CHK_SUCCESS(hash_table_initialize_cpu(count, hash_table), err);

  // Build the hash table: insert all the keys, where the value corresponding
  // to each key is its index in keys array.
  for (int index = 0; index < count; index++) {
    uint32_t key = keys[index];
    CHK_SUCCESS(hash_table_put_cpu(*hash_table, key, index), err_free_state);
  }
  return SUCCESS;

 err_free_state:
  hash_table_finalize_cpu(*hash_table);
 err:
  return FAILURE;
}

error_t hash_table_finalize_cpu(hash_table_t* hash_table) {
  assert(hash_table);
  assert(hash_table->entries);
  free(hash_table->entries);
  if (hash_table->allocated) {
    free(hash_table);
  } else {
    memset(hash_table, 0, sizeof(hash_table_t));
  }
  return SUCCESS;
}

error_t hash_table_put_cpu(hash_table_t* hash_table, uint32_t key, int value) {
  uint64_t entry = HT_MAKE_ENTRY(key, value);
  uint32_t location = HT_FUNC1(hash_table, key);
  int its = 0;

  for (; its < HT_MAX_ITERATIONS; its++) {
    // Insert the new item.
    entry = __sync_lock_test_and_set(&(hash_table->entries[location]), entry);
    key = HT_GET_KEY(entry);
    if (key == HT_KEY_EMPTY) break;

    // If an item was evicted, reinsert it again.
    uint32_t location_1 = HT_FUNC1(hash_table, key);
    uint32_t location_2 = HT_FUNC2(hash_table, key);
    uint32_t location_3 = HT_FUNC3(hash_table, key);
    uint32_t location_4 = HT_FUNC4(hash_table, key);
    if (location == location_1) location = location_2;
    else if (location == location_2) location = location_3;
    else if (location == location_3) location = location_4;
    else location = location_1;
  }
  // The eviction chain was too long, can't build the hash table.
  if (its == HT_MAX_ITERATIONS) {
    return FAILURE;
  }
  return SUCCESS;
}

error_t hash_table_get_cpu(hash_table_t* hash_table, uint32_t key, int* value) {
  assert(value);
  HT_LOOKUP(hash_table, key, (*value));
  if (*value == -1) return FAILURE;
  return SUCCESS;
}

error_t hash_table_get_cpu(hash_table_t* hash_table, uint32_t* keys,
                           uint32_t count, int** values) {
  CALL_SAFE(totem_malloc(count * sizeof(int), TOTEM_MEM_HOST, (void**)values));
  for (int k = 0; k < count; k++) {
    CHK_SUCCESS(hash_table_get_cpu(hash_table, keys[k], &((*values)[k])), err);
  }
  return SUCCESS;
 err:
  free(*values);
  return FAILURE;
}

error_t hash_table_get_keys_cpu(hash_table_t* hash_table, uint32_t** keys,
                                uint32_t* count) {
  assert(hash_table && count);
  *count = 0;
  if (hash_table->size == 0) return SUCCESS;

  *keys = (uint32_t*)calloc(hash_table->size, sizeof(uint32_t));
  if (!(*keys)) return FAILURE;
  for (int e = 0; e < hash_table->size; e++) {
    uint32_t key = HT_GET_KEY((hash_table->entries[e]));
    if (key != HT_KEY_EMPTY) {
      (*keys)[*count] = key;
      (*count)++;
    }
  }
  if (*count == 0) {
    free(*keys);
    *keys = NULL;
  }
  return SUCCESS;
}

/**
 * Retrieves a list of values for the corresponding keys.
 * @param[in] hash_table hash table
 * @param[in] keys the set of keys to look up
 * @param[in] count number of keys
 * @param[out] values the list of retrieved values
 */
PRIVATE __global__ void get_kernel(hash_table_t hash_table, uint32_t* keys,
                                   uint32_t count, int* values) {
  uint32_t index = THREAD_GLOBAL_INDEX;
  if (index >= count) return;
  int value;
  HT_LOOKUP(&hash_table, keys[index], value);
  values[index] = value;
}

error_t hash_table_get_gpu(hash_table_t* hash_table, uint32_t* keys,
                           uint32_t count, int** values) {
  uint32_t* keys_d = NULL;
  CHK_CU_SUCCESS(cudaMalloc((void**)&(keys_d), count * sizeof(uint32_t)), err);
  CHK_CU_SUCCESS(cudaMemcpy(keys_d, keys, count * sizeof(uint32_t),
                            cudaMemcpyHostToDevice), err_free_keys_d);
  int* values_d;
  CHK_CU_SUCCESS(cudaMalloc((void**)&(values_d), count * sizeof(int)),
                 err_free_keys_d);

  {
  dim3 blocks, threads_per_block;
  KERNEL_CONFIGURE(count, blocks, threads_per_block);
  get_kernel<<<blocks, threads_per_block>>>(*hash_table, keys_d, count,
                                            values_d);
  CHK_CU_SUCCESS(cudaGetLastError(), err_free_values_d);
  CALL_SAFE(totem_malloc(count * sizeof(int), TOTEM_MEM_HOST, (void**)values));
  CHK_CU_SUCCESS(cudaMemcpy(*values, values_d, count * sizeof(int),
                            cudaMemcpyDeviceToHost), err_free_values);
  }
  CALL_CU_SAFE(cudaFree(values_d));
  CALL_CU_SAFE(cudaFree(keys_d));
  return SUCCESS;

 err_free_values:
  totem_free(*values, TOTEM_MEM_HOST);
 err_free_values_d:
  CALL_CU_SAFE(cudaFree(values_d));
 err_free_keys_d:
  CALL_CU_SAFE(cudaFree(keys_d));
 err:
  return FAILURE;
}

error_t hash_table_finalize_gpu(hash_table_t* hash_table) {
  assert(hash_table);
  assert(hash_table->entries);
  CALL_CU_SAFE(cudaFree(hash_table->entries));
  if (hash_table->allocated) {
    free(hash_table);
  } else {
    memset(hash_table, 0, sizeof(hash_table_t));
  }
  return SUCCESS;
}

error_t hash_table_initialize_gpu(hash_table_t* hash_table,
                                  hash_table_t* hash_table_d) {
  hash_table_d->size = hash_table->size;
  hash_table_d->allocated = false;
  CHK_CU_SUCCESS(cudaMalloc((void**)&(hash_table_d->entries),
                            hash_table_d->size * sizeof(uint64_t)), err);
  CHK_CU_SUCCESS(cudaMemcpy(hash_table_d->entries, hash_table->entries,
                            hash_table_d->size * sizeof(uint64_t),
                            cudaMemcpyHostToDevice), err_free_entries);
  return SUCCESS;

 err_free_entries:
  CALL_CU_SAFE(cudaFree(hash_table_d->entries));
 err:
  return FAILURE;
}

error_t hash_table_initialize_gpu(hash_table_t* hash_table,
                                  hash_table_t** hash_table_d) {
  *hash_table_d = (hash_table_t*)calloc(1, sizeof(hash_table_t));
  CHK(*hash_table_d, err);
  error_t rc;
  rc = hash_table_initialize_gpu(hash_table, *hash_table_d);
  (*hash_table_d)->allocated = true;
  return rc;
 err:
  return FAILURE;
}

error_t hash_table_initialize_gpu(uint32_t* keys, uint32_t count,
                                  hash_table_t** hash_table) {
  hash_table_t* hash_table_h;
  CHK_SUCCESS(hash_table_initialize_cpu(keys, count, &hash_table_h), err);
  CHK_SUCCESS(hash_table_initialize_gpu(hash_table_h, hash_table),
              err_free_host_state);
  hash_table_finalize_cpu(hash_table_h);
  return SUCCESS;

 err_free_host_state:
  hash_table_finalize_cpu(hash_table_h);
 err:
  return FAILURE;
}
