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
  error_t(*initialize)(uint32_t*, uint32_t, hash_table_t**);
  error_t(*finalize)(hash_table_t*);
  error_t(*get)(hash_table_t*, uint32_t*, uint32_t, int**);
} HashTableFunctions;

HashTableFunctions hash_table_cpu = {
  hash_table_initialize_cpu,
  hash_table_finalize_cpu,
  hash_table_get_cpu
};

HashTableFunctions hash_table_gpu = {
  hash_table_initialize_gpu,
  hash_table_finalize_gpu,
  hash_table_get_gpu
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
  uint32_t count = 100000;
  uint32_t* keys = (uint32_t*)calloc(count, sizeof(int));
  for (uint32_t k = 0; k < count; k++) {
    // generate unique random keys
    keys[k] = (uint32_t)((rand()) << 24 | k);
  }

  // build the hash table
  hash_table_t* hash_table;
  EXPECT_EQ(SUCCESS, hash_table_funcs->initialize(keys, count, &hash_table));

  // retrieve the values and check
  int* values = NULL;
  CALL_SAFE(hash_table_funcs->get(hash_table, keys, count, &values));
  for (uint32_t k = 0; k < count; k++) {
    EXPECT_EQ(k, (uint32_t)values[k]);
  }

  // clean up
  EXPECT_EQ(SUCCESS, hash_table_funcs->finalize(hash_table));
  mem_free(values);
  free(keys);
}

class HashTableCPUTests : public ::testing::Test {
};

TEST_F(HashTableCPUTests, PutGet) {
  srand(13);
  uint32_t count = 100;
  hash_table_t* hash_table;
  EXPECT_EQ(SUCCESS, hash_table_initialize_cpu(count, &hash_table));

  uint32_t* keys = (uint32_t*)calloc(count, sizeof(uint32_t));
  for (uint32_t k = 0; k < count; k++) {
    // just to increase the probability that we get unique keys!
    keys[k] = rand() * rand();
    int value;
    EXPECT_EQ(FAILURE, hash_table_get_cpu(hash_table, keys[k], &value));
    EXPECT_EQ(SUCCESS, hash_table_put_cpu(hash_table, keys[k], k));
    EXPECT_EQ(SUCCESS, hash_table_get_cpu(hash_table, keys[k], &value));
    EXPECT_EQ(k, (uint32_t)value);
  }

  uint32_t  count_got;
  uint32_t* keys_got;
  EXPECT_EQ(SUCCESS, hash_table_get_keys_cpu(hash_table, &keys_got, 
                                             &count_got));
  EXPECT_EQ(count, count_got);
  for (uint32_t k = 0; k < count; k++) {
    bool found = false;
    for (uint32_t j = 0; j < count; j++) {
      if (keys_got[k] == keys[j]) {
        found = true;
        break;
      }
    }
    EXPECT_EQ(true, found);
  }
  EXPECT_EQ(SUCCESS, hash_table_finalize_cpu(hash_table));
}

// From Google documentation:
// In order to run value-parameterized tests, we need to instantiate them,
// or bind them to a list of values which will be used as test parameters.
//
// Values() receives a list of parameters and the framework will execute the
// whole set of tests HashTableTest for each element of Values()
INSTANTIATE_TEST_CASE_P(HashTableGPUAndCPUTest, HashTableTest, 
                        Values(&hash_table_cpu, &hash_table_gpu));

#else

// From Google documentation:
// Google Test may not support value-parameterized tests with some
// compilers. This dummy test keeps gtest_main linked in.
TEST_P(DummyTest, ValueParameterizedTestsAreNotSupportedOnThisPlatform) {}

#endif  // GTEST_HAS_PARAM_TEST
