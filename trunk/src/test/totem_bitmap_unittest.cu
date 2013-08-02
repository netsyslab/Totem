/* TODO(lauro,abdullah,elizeu): Add license.
 *
 * Contains unit tests for totem bitmap functions.
 *
 *  Created on: 2012-08-06
 *      Author: Abdullah Gharaibeh
 */

// totem includes
#include "totem_common_unittest.h"
#include "totem_bitmap.cuh"

#if GTEST_HAS_PARAM_TEST

using ::testing::TestWithParam;
using ::testing::Values;

class BitmapTest : public TestWithParam<int> {
 public:
  virtual void SetUp() {
    // Ensure the minimum CUDA architecture is supported
    CUDA_CHECK_VERSION();
    seed_ = GetParam();
    bitmap_ = NULL;
    bits_not_set_count_ = 0;
  }
  
  void BitmapGetNotSet() {
    bits_not_set_count_ = 0;
    for (vid_t i = 0; i < bitmap_len_; i++) {
      // Search for the bit in the bits_set array
      vid_t j = 0;
      for (; j < bits_total_set_count_; j++) {
        if (bits_set_[j] == i) {
          break;
        }
      }
      if (j == bits_total_set_count_) {
        // This bit is not set, add it to the bits_not_set array
        bits_not_set_[bits_not_set_count_++] = i;
      }
    }
  }
  
  void BitmapInitTest() {
    // Randomly set a group of bits
    srand (seed_);
    for (vid_t i = 0; i < bits_total_set_count_; i++) {
      vid_t index;
      bool try_again = false;
      do {
        try_again = false;
        index = rand() % bitmap_len_;
        for (int k = 0; k < i; k++) {
          if (index == bits_set_[k]) {
            try_again = true;
          }
        }         
      } while(try_again);
      bits_set_[i] = index;
    }   
    // Get the bits that are not set
    BitmapGetNotSet();
  }
  
  void BitmapSetVectorCPU(bitmap_t bitmap, vid_t* bits, vid_t count) {
    OMP(omp parallel for)
    for (vid_t i = 0; i < count; i++) {
      bitmap_set_cpu(bitmap, bits[i]);
      // Try to unset again
      EXPECT_FALSE(bitmap_set_cpu(bitmap, bits[i]));
    }
  }
  
  void BitmapUnsetVectorCPU(bitmap_t bitmap, vid_t* bits, vid_t count) {
    OMP(omp parallel for)
    for (vid_t i = 0; i < count; i++) {
      bitmap_unset_cpu(bitmap, bits[i]);
      // Try to unset again
      EXPECT_FALSE(bitmap_unset_cpu(bitmap, bits[i]));
    }
  }
  
  void BitmapVerifyCPU(bitmap_t bitmap, vid_t* is_set, vid_t is_set_count, 
                       vid_t* is_not_set, vid_t is_not_set_count) {
    OMP(omp parallel for)
    for (vid_t i = 0; i < is_set_count; i++) {
      EXPECT_TRUE(bitmap_is_set(bitmap, is_set[i]));
    }
    OMP(omp parallel for)
    for (vid_t i = 0; i < is_not_set_count; i++) {
      EXPECT_FALSE(bitmap_is_set(bitmap, is_not_set[i]));
    }
  }
  
 protected:
  // Initialize the test
  int seed_;
  bitmap_t bitmap_;
  bitmap_t bitmap_copy_;
  bitmap_t bitmap_diff_;
  static const vid_t bitmap_len_ = 1006;
  static const vid_t bits_set_count_ = bitmap_len_ / 2;
  static const vid_t bits_add_set_count_ = bitmap_len_ / 4;
  static const vid_t bits_total_set_count_ = 
    bits_set_count_ + bits_add_set_count_;
  vid_t bits_set_[bits_total_set_count_];
  vid_t bits_not_set_[bitmap_len_];
  vid_t bits_not_set_count_;
};

TEST_P(BitmapTest, BitmapCPU) {

  // Create the bitmap and initialize the test
  bitmap_ = bitmap_init_cpu(bitmap_len_);
  bitmap_copy_ = bitmap_init_cpu(bitmap_len_);
  BitmapInitTest();

  // First set the bits in the bits_set array and verify the bitmap
  BitmapSetVectorCPU(bitmap_, bits_set_, bits_set_count_);
  BitmapVerifyCPU(bitmap_, bits_set_, bits_set_count_, bits_not_set_, 
                  bits_not_set_count_);
  vid_t count = bitmap_count_cpu(bitmap_, bitmap_len_);
  EXPECT_EQ((int)bits_set_count_, (int)count);

  // Copy the bitmap and verify the copied-bitmap
  bitmap_copy_cpu(bitmap_, bitmap_copy_, bitmap_len_);
  BitmapVerifyCPU(bitmap_copy_, bits_set_, bits_set_count_, bits_not_set_, 
                  bits_not_set_count_);
  count = bitmap_count_cpu(bitmap_, bitmap_len_);
  EXPECT_EQ((int)bits_set_count_, (int)count);

  // Add the rest of bits to be set to the copied bitmap
  BitmapSetVectorCPU(bitmap_copy_, bits_set_, bits_total_set_count_);
  BitmapVerifyCPU(bitmap_copy_, bits_set_, bits_total_set_count_, 
                  bits_not_set_, bits_not_set_count_);
  count = bitmap_count_cpu(bitmap_copy_, bitmap_len_);
  EXPECT_EQ((int)(bits_total_set_count_), (int)count);

  // Diff the bitmap and its copy
  bitmap_diff_cpu(bitmap_, bitmap_copy_, bitmap_len_);
  BitmapVerifyCPU(bitmap_copy_, &bits_set_[bits_set_count_], 
                  bits_add_set_count_, bits_not_set_, bits_not_set_count_);
  count = bitmap_count_cpu(bitmap_copy_, bitmap_len_);
  EXPECT_EQ((int)(bits_add_set_count_), (int)count);

  // Second revert the set and unset bits, and verify the bitmap
  BitmapSetVectorCPU(bitmap_, bits_not_set_, bits_not_set_count_);
  BitmapUnsetVectorCPU(bitmap_, bits_set_, bits_set_count_);
  BitmapVerifyCPU(bitmap_, bits_not_set_, bits_not_set_count_, bits_set_, 
                  bits_set_count_);
  count = bitmap_count_cpu(bitmap_, bitmap_len_);
  EXPECT_EQ((int)bits_not_set_count_, (int)count);

  // Copy the bitmap and verify the copied-bitmap
  bitmap_copy_cpu(bitmap_, bitmap_copy_, bitmap_len_);
  BitmapVerifyCPU(bitmap_copy_, bits_not_set_, bits_not_set_count_, bits_set_, 
                  bits_set_count_);
  count = bitmap_count_cpu(bitmap_copy_, bitmap_len_);
  EXPECT_EQ((int)bits_not_set_count_, (int)count);

  bitmap_finalize_cpu(bitmap_);
  bitmap_finalize_cpu(bitmap_copy_);
}

__global__ void BitmapSetVectorGPU(vid_t* bitmap, vid_t* bits, vid_t count) {
  if (THREAD_GLOBAL_INDEX < count) {
    bitmap_set_gpu(bitmap, bits[THREAD_GLOBAL_INDEX]);
    KERNEL_EXPECT_TRUE(!bitmap_set_gpu(bitmap, bits[THREAD_GLOBAL_INDEX]));
  }
}

__global__ void BitmapUnsetVectorGPU(vid_t* bitmap, vid_t* bits, vid_t count) {
  if (THREAD_GLOBAL_INDEX < count) {
    bitmap_unset_gpu(bitmap, bits[THREAD_GLOBAL_INDEX]);
    KERNEL_EXPECT_TRUE(!bitmap_unset_gpu(bitmap, bits[THREAD_GLOBAL_INDEX]));
  }
}

__global__ void BitmapVerifyGPU(vid_t* bitmap, vid_t* bits_set, 
                                vid_t bits_set_count, vid_t* bits_not_set, 
                                vid_t bits_not_set_count) {
  vid_t index = THREAD_GLOBAL_INDEX;
  if (index < bits_set_count) {
    KERNEL_EXPECT_TRUE(bitmap_is_set(bitmap, bits_set[index]));
  }
  if (index < bits_not_set_count) {
    KERNEL_EXPECT_TRUE(!bitmap_is_set(bitmap, bits_not_set[index]));
  }
}

TEST_P(BitmapTest, BitmapGPU) {
  // Create the bitmap and initialize the test
  bitmap_ = bitmap_init_gpu(bitmap_len_);
  bitmap_copy_ = bitmap_init_gpu(bitmap_len_);
  BitmapInitTest();

  // Move state to GPU
  vid_t* bits_set_d = NULL;
  CALL_CU_SAFE(cudaMalloc(&bits_set_d, bits_total_set_count_ * sizeof(vid_t)));
  CALL_CU_SAFE(cudaMemcpy(bits_set_d, bits_set_, 
                          bits_total_set_count_ * sizeof(vid_t),
                          cudaMemcpyDefault));
  vid_t* bits_not_set_d = NULL;
  CALL_CU_SAFE(cudaMalloc(&bits_not_set_d, 
                          bits_not_set_count_ * sizeof(vid_t)));
  CALL_CU_SAFE(cudaMemcpy(bits_not_set_d, bits_not_set_,
                          bits_not_set_count_ * sizeof(vid_t),
                          cudaMemcpyDefault));

  // First set the bits in the bits_set array and verify the bitmap
  int threads = DEFAULT_THREADS_PER_BLOCK;
  dim3 blocks;
  kernel_configure(bitmap_len_, blocks);
  BitmapSetVectorGPU<<<blocks, threads>>>(bitmap_, bits_set_d, bits_set_count_);
  BitmapVerifyGPU<<<blocks, threads>>>(bitmap_, bits_set_d, bits_set_count_,
                                       bits_not_set_d, bits_not_set_count_);
  vid_t count = bitmap_count_gpu(bitmap_, bitmap_len_);
  EXPECT_EQ((int)bits_set_count_, (int)count);

  // Copy the bitmap and verify the copied-bitmap
  bitmap_copy_gpu(bitmap_, bitmap_copy_, bitmap_len_);
  BitmapVerifyGPU<<<blocks, threads>>>(bitmap_copy_, bits_set_d, 
                                       bits_set_count_, bits_not_set_d,
                                       bits_not_set_count_);
  count = bitmap_count_gpu(bitmap_copy_, bitmap_len_);
  EXPECT_EQ((int)bits_set_count_, (int)count);

  // Add the rest of bits to be set to the copied bitmap
  BitmapSetVectorGPU<<<blocks, threads>>>
    (bitmap_copy_, bits_set_d, bits_total_set_count_);
  BitmapVerifyGPU<<<blocks, threads>>>
    (bitmap_copy_, bits_set_d, bits_total_set_count_, bits_not_set_d,
     bits_not_set_count_);
  count = bitmap_count_gpu(bitmap_copy_, bitmap_len_);
  EXPECT_EQ((int)(bits_total_set_count_), (int)count);

  // Diff the bitmap and its copy
  bitmap_diff_gpu(bitmap_, bitmap_copy_, bitmap_len_);
  BitmapVerifyGPU<<<blocks, threads>>>
    (bitmap_copy_, &bits_set_d[bits_set_count_], bits_add_set_count_, 
     bits_not_set_d, bits_not_set_count_);
  count = bitmap_count_gpu(bitmap_copy_, bitmap_len_);
  EXPECT_EQ((int)(bits_add_set_count_), (int)count);
  
  // Second revert the set and unset bits, and verify the bitmap
  BitmapUnsetVectorGPU<<<blocks, threads>>>(bitmap_, bits_set_d, 
                                            bits_set_count_);
  BitmapSetVectorGPU<<<blocks, threads>>>(bitmap_, bits_not_set_d,
                                          bits_not_set_count_);
  BitmapVerifyGPU<<<blocks, threads>>>(bitmap_, bits_not_set_d,
                                       bits_not_set_count_, bits_set_d,
                                       bits_set_count_);
  count = bitmap_count_gpu(bitmap_, bitmap_len_);
  EXPECT_EQ((int)bits_not_set_count_, (int)count);

  // Copy the bitmap and verify the copied-bitmap
  bitmap_copy_gpu(bitmap_, bitmap_copy_, bitmap_len_);
  BitmapVerifyGPU<<<blocks, threads>>>(bitmap_copy_, bits_not_set_d,
                                       bits_not_set_count_, bits_set_d,
                                       bits_set_count_);
  count = bitmap_count_gpu(bitmap_copy_, bitmap_len_);
  EXPECT_EQ((int)bits_not_set_count_, (int)count);


  bitmap_finalize_gpu(bitmap_);
  bitmap_finalize_gpu(bitmap_copy_);
  cudaFree(bits_set_d);
  cudaFree(bits_not_set_d);
}

INSTANTIATE_TEST_CASE_P(BitmapGPUAndCPUTest, BitmapTest,
                        Values(110, 1234, 458374, 19740));

#else

// From Google documentation:
// Google Test may not support value-parameterized tests with some
// compilers. This dummy test keeps gtest_main linked in.
TEST_P(DummyTest, ValueParameterizedTestsAreNotSupportedOnThisPlatform) {}

#endif  // GTEST_HAS_PARAM_TEST
