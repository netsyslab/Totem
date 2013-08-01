/**
 * A thread-safe bitmap implementation 
 *
 *  Created on: 2013-07-29
 *  Author: Abdullah Gharaibeh
 */

#include "totem_bitmap.cuh"


PRIVATE __global__ void 
bitmap_count_kernel(const bitmap_t __restrict bitmap, const vid_t word_count, 
                    vid_t* count) {
  const int THREADS = MAX_THREADS_PER_BLOCK;
  __shared__ vid_t sdata[THREADS];

  // perform first level of reduction. 
  vid_t tid = THREAD_BLOCK_INDEX;
  vid_t i = BLOCK_GLOBAL_INDEX * (THREADS*2) + THREAD_BLOCK_INDEX;
  vid_t sum = (i < word_count) ? __popc(bitmap[i]) : 0;
  if (i + THREADS < word_count) sum += __popc(bitmap[i + THREADS]);  
  sdata[tid] = sum;
  __syncthreads();
  
  // do reduction in shared mem
  if (THREADS >= 1024) {
    if (tid < 512) sdata[tid] = sum = sum + sdata[tid + 512];
    __syncthreads();
  }
  if (THREADS >= 512) {
    if (tid < 256) sdata[tid] = sum = sum + sdata[tid + 256];
    __syncthreads();
  }  
  if (THREADS >= 256) {
    if (tid < 128) sdata[tid] = sum = sum + sdata[tid + 128];      
    __syncthreads();
  }  
  if (THREADS >= 128) {
    if (tid <  64) sdata[tid] = sum = sum + sdata[tid +  64];
    __syncthreads();
  }
  
  if (tid < 32) {
    // now that we are using warp-synchronous programming (below)
    // we need to declare our shared memory volatile so that the compiler
    // doesn't reorder stores to it and induce incorrect behavior.
    volatile vid_t* smem = sdata;    
    if (THREADS >= 64) smem[tid] = sum = sum + smem[tid + 32];     
    if (THREADS >= 32) smem[tid] = sum = sum + smem[tid + 16];      
    if (THREADS >= 16) smem[tid] = sum = sum + smem[tid +  8];
    if (THREADS >= 8) smem[tid] = sum = sum + smem[tid +  4];      
    if (THREADS >= 4)  smem[tid] = sum = sum + smem[tid +  2];
    if (THREADS >= 2) smem[tid] = sum = sum + smem[tid +  1];
  }
  
  // Write the final result
  if (tid == 0 && sdata[0]) atomicAdd(count, sdata[0]);
}

PRIVATE __global__ void
bitmap_diff_copy_kernel(const bitmap_t __restrict cur, bitmap_t diff, 
                       bitmap_t copy, size_t words) {
  int index = THREAD_GLOBAL_INDEX;
  if (index >= words) return;
  diff[index] ^= cur[index];
  copy[index] = cur[index];
}

vid_t bitmap_diff_copy_count_gpu(bitmap_t cur, bitmap_t diff, bitmap_t copy,
                                 size_t len, vid_t* count_d,   
                                 cudaStream_t stream) {
  dim3 blocks;
  vid_t words = bitmap_bits_to_words(len);
  kernel_configure(words, blocks);
  bitmap_diff_copy_kernel<<<blocks, DEFAULT_THREADS_PER_BLOCK, 0, stream>>>
    (cur, diff, copy, words);
  CALL_CU_SAFE(cudaGetLastError());

  // Count the number of set bits in the diff bitmap
  cudaMemsetAsync(count_d, 0, sizeof(vid_t), stream);
  // We need a number of threads that is equal to half of the number of words as
  // each thread will start by summing the number of set bits in two words
  CALL_SAFE(kernel_configure(words / 2 + 1, blocks));
  bitmap_count_kernel<<<blocks, MAX_THREADS_PER_BLOCK, 0, stream>>>
    (diff, words, count_d);
  vid_t count;
  CALL_CU_SAFE(cudaMemcpyAsync(&count, count_d, sizeof(vid_t), 
                               cudaMemcpyDefault, stream));
  CALL_CU_SAFE(cudaStreamSynchronize(stream));
  return count;
}

vid_t bitmap_diff_copy_count_cpu(bitmap_t cur, bitmap_t diff, bitmap_t copy,
                                 size_t len) {
  vid_t words = bitmap_bits_to_words(len);
  vid_t count = 0;
  OMP(omp parallel for schedule(static) reduction(+ : count))
  for (vid_t w = 0; w < words; w++) {
    diff[w] ^= cur[w];
    copy[w] = cur[w];
    // popcount returns the number of set bits in the word
    count += __builtin_popcount(diff[w]);
  }
  return count;
}
