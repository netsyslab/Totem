/* TODO(lauro,abdullah,elizeu): Add license.
 *
 * Contains unit tests for hash table data structure
 *
 *  Created on: 2011-12-30
 *      Author: Abdullah Gharaibeh
 */

// totem includes
#include "totem_common_unittest.h"
#include "totem_hash_table.h"

#if GTEST_HAS_PARAM_TEST

using ::testing::TestWithParam;
using ::testing::Values;

// The following implementation relies on TestWithParam<HashTableFunctions>
// to test the two versions of hash table implemented: CPU and GPU.
// Details on how to use TestWithParam<T> can be found at:
// http://code.google.com/p/googletest/source/browse/trunk/samples/sample7_unittest.cco

/**
 * Defines the list of related functions to test. This is to allow easy testing
 * of both cpu and gpu versions
 */
typedef struct {
  error_t(*initialize)(int*, uint32_t*, int, hash_table_t**);
  error_t(*finalize)(hash_table_t*);
  error_t(*get)(hash_table_t*, uint32_t*, int, int**);
} HashTableFunctions;

HashTableFunctions hash_table_cpu = {
  hash_table_initialize_cpu,
  hash_table_finalize_cpu,
  hash_table_get_cpu
};

class HashTableTest : public TestWithParam<HashTableFunctions*> {
 public:
  virtual void SetUp() {
    hash_table_funcs = GetParam();
  }
 protected:
   HashTableFunctions* hash_table_funcs;
};

TEST_P(HashTableTest, BuildAndRetrieveRandom) {
  // create random keys and correponding values
  srand(13);
  int count = 100000;
  uint32_t* keys = (uint32_t*)calloc(count, sizeof(int));
  int* values = (int*)mem_alloc(count * sizeof(int));
  for (int k = 0; k < count; k++) {
    // generate unique random keys
    keys[k] = (uint32_t)((rand()) << 24 | k);
    values[k] = rand();
  }

  // build the hash table
  hash_table_t* hash_table;
  EXPECT_EQ(SUCCESS, hash_table_funcs->initialize(values, keys, count, 
                                                  &hash_table));

  // retrieve the values and check
  int* retrieved_values = NULL;
  CALL_SAFE(hash_table_funcs->get(hash_table, keys, count, &retrieved_values));
  for (int k = 0; k < count; k++) {
    EXPECT_EQ(values[k], retrieved_values[k]);
  }

  // clean up
  EXPECT_EQ(SUCCESS, hash_table_funcs->finalize(hash_table));
  mem_free(retrieved_values);
  mem_free(values);
  free(keys);
}


// From Google documentation:
// In order to run value-parameterized tests, we need to instantiate them,
// or bind them to a list of values which will be used as test parameters.
//
// Values() receives a list of parameters and the framework will execute the
// whole set of tests HashTableTest for each element of Values()
INSTANTIATE_TEST_CASE_P(HashTableGPUAndCPUTest, HashTableTest, 
                        Values(&hash_table_cpu));

#else

// From Google documentation:
// Google Test may not support value-parameterized tests with some
// compilers. This dummy test keeps gtest_main linked in.
TEST_P(DummyTest, ValueParameterizedTestsAreNotSupportedOnThisPlatform) {}

#endif  // GTEST_HAS_PARAM_TEST
