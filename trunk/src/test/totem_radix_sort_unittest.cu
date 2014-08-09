/*
 * Contains unit tests for binary radix-sort algorithm
 *
 *  Created on: 2014-08-03
 *      Author: Daniel Lucas dos Santos Borges
 */

// totem includes
#include "totem_common_unittest.h"
#include "totem_radix_sort.h"

#if GTEST_HAS_PARAM_TEST

using ::testing::TestWithParam;
using ::testing::Values;

class RadixsortTest : public ::testing::Test{
 public:
  virtual void SetUp() {
  }
};

TEST_F(RadixsortTest, AscendingSortTest) {
  unsigned int seed = 13;
  const int kSize = 10000;
  vid_t array[kSize];
  for (int i = 0; i < kSize; i++) {
    array[i] = rand_r(&seed) % 100000;
  }
  // sorting in ascending order
  parallel_radix_sort(array, kSize, sizeof(vid_t),
          true /* indicates ascending sorting */);

  // verifying if the array is really sorted
  for (int i = 1; i < kSize; i++) {
    EXPECT_LE(array[i - 1], array[i]);
  }
}

TEST_F(RadixsortTest, DescendingSortTest) {
  unsigned int seed = 13;
  const int kSize = 10000;
  vid_t array[kSize];
  for (int i = 0; i < kSize; i++) {
    array[i] = rand_r(&seed) % 100000;
  }
  // sorting in descending order
  parallel_radix_sort(array, kSize, sizeof(vid_t),
          false /* indicates descending sorting */);

  // verifying if the array is really sorted
  for (int i = 1; i < kSize; i++) {
    EXPECT_GE(array[i - 1], array[i]);
  }
}

#else

// From Google documentation:
// Google Test may not support value-parameterized tests with some
// compilers. This dummy test keeps gtest_main linked in.
TEST_P(DummyTest, ValueParameterizedTestsAreNotSupportedOnThisPlatform) {}

#endif  // GTEST_HAS_PARAM_TEST

