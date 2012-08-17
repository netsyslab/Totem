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
      for (; j < bits_set_count_; j++) {
        if (bits_set_[j] == i) {
          break;
        }
      }
      if (j == bits_set_count_) {
        // This bit is not set, add it to the bits_not_set array
        bits_not_set_[bits_not_set_count_++] = i;
      }
    }
  }
  
 void BitmapInitTest() {
   // Randomly set a group of bits
   bits_set_[0] = 0;
   bits_set_[1] = bits_set_count_ - 1;
   srand (seed_);
   for (vid_t i = 2; i < bits_set_count_; i++) {
     bits_set_[i] = rand() % bitmap_len_;
   }   
   // Get the bits that are not set
   BitmapGetNotSet();
 }
 
 void BitmapSetVectorCPU(vid_t* bits, vid_t count) {
   OMP(omp parallel for)
   for (vid_t i = 0; i < count; i++) {
     bitmap_set_cpu(bitmap_, bits[i]);
     // Try to unset again
     EXPECT_FALSE(bitmap_set_cpu(bitmap_, bits[i]));
   }
 }
 
 void BitmapUnsetVectorCPU(vid_t* bits, vid_t count) {
   OMP(omp parallel for)
   for (vid_t i = 0; i < count; i++) {
     bitmap_unset_cpu(bitmap_, bits[i]);
     // Try to unset again
     EXPECT_FALSE(bitmap_unset_cpu(bitmap_, bits[i]));
   }
 }
 
 void BitmapVerifyCPU(vid_t* is_set, vid_t is_set_count, vid_t* is_not_set, 
                      vid_t is_not_set_count) {
   OMP(omp parallel for)
   for (vid_t i = 0; i < is_set_count; i++) {
     EXPECT_TRUE(bitmap_is_set(bitmap_, is_set[i]));
   }
   OMP(omp parallel for)
   for (vid_t i = 0; i < is_not_set_count; i++) {
     EXPECT_FALSE(bitmap_is_set(bitmap_, is_not_set[i]));
   }
 }
 
 protected:
  // Initialize the test
  int seed_;
  bitmap_t bitmap_;
  static const vid_t bitmap_len_ = 1006;
  static const vid_t bits_set_count_ = bitmap_len_ / 2;
  vid_t bits_set_[bits_set_count_];
  vid_t bits_not_set_[bitmap_len_];
  vid_t bits_not_set_count_;
};

TEST_P(BitmapTest, BitmapCPU) {

   // Create the bitmap and initialize the test
   bitmap_ = bitmap_init_cpu(bitmap_len_);
   BitmapInitTest();

  // First set the bits in the bits_set array and verify the bitmap
  BitmapSetVectorCPU(bits_set_, bits_set_count_);
  BitmapVerifyCPU(bits_set_, bits_set_count_, bits_not_set_, 
                  bits_not_set_count_);

  // Second revert the set and unset bits, and verify the bitmap
  BitmapSetVectorCPU(bits_not_set_, bits_not_set_count_);
  BitmapUnsetVectorCPU(bits_set_, bits_set_count_);
  BitmapVerifyCPU(bits_not_set_, bits_not_set_count_, bits_set_, 
                  bits_set_count_);

  bitmap_finalize_cpu(bitmap_);
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
  BitmapInitTest();

  // Move state to GPU
  vid_t* bits_set_d = NULL;
  CALL_CU_SAFE(cudaMalloc(&bits_set_d, bits_set_count_ * sizeof(vid_t)));
  CALL_CU_SAFE(cudaMemcpy(bits_set_d, bits_set_, 
                          bits_set_count_ * sizeof(vid_t),
                          cudaMemcpyDefault));
  vid_t* bits_not_set_d = NULL;
  CALL_CU_SAFE(cudaMalloc(&bits_not_set_d, 
                          bits_not_set_count_ * sizeof(vid_t)));
  CALL_CU_SAFE(cudaMemcpy(bits_not_set_d, bits_not_set_,
                          bits_not_set_count_ * sizeof(vid_t),
                          cudaMemcpyDefault));

  // First set the bits in the bits_set array and verify the bitmap
  dim3 blocks, threads;
  KERNEL_CONFIGURE(bitmap_len_, blocks, threads);
  BitmapSetVectorGPU<<<blocks, threads>>>(bitmap_, bits_set_d, bits_set_count_);
  BitmapVerifyGPU<<<blocks, threads>>>(bitmap_, bits_set_d, bits_set_count_,
                                       bits_not_set_d, bits_not_set_count_);
  
  // Second revert the set and unset bits, and verify the bitmap
  BitmapUnsetVectorGPU<<<blocks, threads>>>(bitmap_, bits_set_d, 
                                            bits_set_count_);
  BitmapSetVectorGPU<<<blocks, threads>>>(bitmap_, bits_not_set_d,
                                          bits_not_set_count_);
  BitmapVerifyGPU<<<blocks, threads>>>(bitmap_, bits_not_set_d,
                                       bits_not_set_count_, bits_set_d,
                                       bits_set_count_);

  bitmap_finalize_gpu(bitmap_);
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
