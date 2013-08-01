/**
 * GPU-based functions of the intensive benchmark for power measurement
 * experiments
 *
 * Created on: 2013-06-10
 * Author: Sidney Pontes Filho
 */

// totem includes
#include "benchmark.h"
#include "totem_comkernel.cuh"

/**
 * Using CUDA-device to compute some mathematical operations.
 */
__global__
void intern_compute_intensive_gpu(float *num1, float *num2) {
  for (uint64_t i = 0; i < 10000; i++) {
    num2[THREAD_GLOBAL_INDEX] = (i * num1[THREAD_GLOBAL_INDEX] 
                                 * num2[THREAD_GLOBAL_INDEX])
      - (num1[THREAD_GLOBAL_INDEX] / num2[THREAD_GLOBAL_INDEX])
      + (pow(num1[THREAD_GLOBAL_INDEX], num2[THREAD_GLOBAL_INDEX]));
  }
}

/**
 * A gpu-intensive routine that computes some mathematical operations
 * in parallel using CUDA.
 */
__host__
error_t compute_intensive_gpu(double duration) {
  printf("GPU Intensive Benchmark\n");
  time_t start, end;
  time(&start);
  dim3 blocks;
  dim3 threads_per_block;
  KERNEL_CONFIGURE(MAX_BLOCK_PER_DIMENSION, blocks, threads_per_block);
  float *num1_h = (float *)malloc(MAX_BLOCK_PER_DIMENSION * sizeof(float));
  float *num2_h = (float *)malloc(MAX_BLOCK_PER_DIMENSION * sizeof(float));
  OMP(omp parallel for)
  for (uint64_t i = 0; i < MAX_BLOCK_PER_DIMENSION; i++) {
    num1_h[i] = (float)drand48();
    num2_h[i] = (float)drand48() * FLT_MAX;
  }
  float *num1_d;
  CHK_CU_SUCCESS(cudaMalloc((void**)&num1_d, MAX_BLOCK_PER_DIMENSION 
                            * sizeof(float)), err_free_num1_d);
  CHK_CU_SUCCESS(cudaMemcpy(num1_d, num1_h, MAX_BLOCK_PER_DIMENSION 
                            * sizeof(float), cudaMemcpyHostToDevice), err);
  float *num2_d;
  CHK_CU_SUCCESS(cudaMalloc((void**)&num2_d, MAX_BLOCK_PER_DIMENSION 
                            * sizeof(float)), err_free_num2_d);
  CHK_CU_SUCCESS(cudaMemcpy(num2_d, num2_h, MAX_BLOCK_PER_DIMENSION 
                            * sizeof(float), cudaMemcpyHostToDevice), err);
  do {
    intern_compute_intensive_gpu<<<blocks, threads_per_block>>>(num1_d, num2_d);
    time(&end);
  } while (difftime(end, start) < duration);
  CHK_CU_SUCCESS(cudaMemcpy(num2_h, num2_d, MAX_BLOCK_PER_DIMENSION 
                            * sizeof(float), cudaMemcpyDeviceToHost), err);
  cudaFree(num1_d);
  cudaFree(num2_d);
  free(num1_h);
  free(num2_h);
  return SUCCESS;
 err_free_num1_d:
  cudaFree(num1_d);
 err_free_num2_d:
  cudaFree(num2_d);
 err:
  return FAILURE;
}

/**
 * Using CUDA-device to compute some mathematical operations using numbers from
 * the array.
 */
__global__
void intern_compute_memory_intensive_gpu(uint64_t *array, uint64_t index) {

  // Reads a random position on array, then that read number is used to 
  // calculate multiplication with index number to change the value of index
  // and the modulus is used to keep the index number in the array range.
  for (uint64_t i = 0; i < 10000; i++) {
    index = (i * array[index]) % MAX_BLOCK_PER_DIMENSION;
  }
}

/**
 * A GPU and memory intensive routine that reads an array in
 * random positions and uses the read number to calculate
 * multiplication and modulus in parallel using CUDA.
 */
__host__
error_t compute_memory_intensive_gpu(double duration) {
  printf("GPU and Memory Intensive Benchmark\n");  
  time_t start, end;
  time(&start);
  dim3 blocks;
  dim3 threads_per_block;
  KERNEL_CONFIGURE(MAX_BLOCK_PER_DIMENSION, blocks, threads_per_block);
  uint64_t *array_h = (uint64_t *)malloc(MAX_BLOCK_PER_DIMENSION 
                                         * sizeof(uint64_t));
  uint64_t num;

  // Initialize the array elements with a random number in array range, 
  // but without be the same number of its index.
  OMP(omp parallel for)
  for (uint64_t i = 0; i < MAX_BLOCK_PER_DIMENSION; i++) {
    do {
      num = random_uint64(MAX_BLOCK_PER_DIMENSION);
    } while (i == num);
    array_h[i] = num;
  }
  uint64_t *array_d;
  CHK_CU_SUCCESS(cudaMalloc((void**)&array_d, MAX_BLOCK_PER_DIMENSION 
                            * sizeof(uint64_t)), err_free_array_d);
  CHK_CU_SUCCESS(cudaMemcpy(array_d, array_h, MAX_BLOCK_PER_DIMENSION 
                            * sizeof(uint64_t), cudaMemcpyHostToDevice), err);
  do {
    uint64_t index = random_uint64(LARGE_ARRAY_SIZE);
    intern_compute_memory_intensive_gpu<<<blocks, threads_per_block>>>
      (array_d, index);
    time(&end);
  } while (difftime(end, start) < duration);
  CHK_CU_SUCCESS(cudaMemcpy(array_h, array_d, MAX_BLOCK_PER_DIMENSION 
                            * sizeof(uint64_t), cudaMemcpyDeviceToHost), err);
  cudaFree(array_d);
  free(array_h);
  return SUCCESS;
 err_free_array_d:
  cudaFree(array_d);
 err:
  return FAILURE;
}

/**
 * Reads an array in ascending order in parallel using a CUDA-device.
 */
__global__
void intern_memory_intensive_cache_friendly_gpu(uint64_t *array) {
  uint64_t index = 0;
  for (uint64_t i = 0; i < 10000; i++) {
    index = array[index];   
  }
}

/**
 * A cache friendly memory intensive routine that reads an array
 * in ascending order in parallel using CUDA.
 */
__host__
error_t memory_intensive_cache_friendly_gpu(double duration) {
  printf("GPU Cache Friendly Memory Intensive Benchmark\n");   
  time_t start, end;
  time(&start);
  dim3 blocks;
  dim3 threads_per_block;
  KERNEL_CONFIGURE(MAX_BLOCK_PER_DIMENSION, blocks, threads_per_block);
  uint64_t *array_h = (uint64_t *)malloc(SMALL_ARRAY_SIZE * sizeof(uint64_t));

  // Initialize the array elements with the same number of the
  // index, but the last element is zero to return to beginning
  // of the array.
  OMP(omp parallel for)
  for (uint64_t i = 0; i < SMALL_ARRAY_SIZE; i++) {
    array_h[i] = i + 1;
  }
  array_h[SMALL_ARRAY_SIZE - 1] = 0;

  uint64_t *array_d;
  CHK_CU_SUCCESS(cudaMalloc((void**)&array_d, SMALL_ARRAY_SIZE 
                            * sizeof(uint64_t)), err_free_array_d);
  CHK_CU_SUCCESS(cudaMemcpy(array_d, array_h, SMALL_ARRAY_SIZE 
                            * sizeof(uint64_t), cudaMemcpyHostToDevice), err);
  do {
    intern_memory_intensive_cache_friendly_gpu<<<blocks, threads_per_block>>>
      (array_d);
    time(&end);
  } while (difftime(end, start) < duration);
  CHK_CU_SUCCESS(cudaMemcpy(array_h, array_d, SMALL_ARRAY_SIZE 
                            * sizeof(uint64_t), cudaMemcpyDeviceToHost), err);
  cudaFree(array_d);
  free(array_h);
  return SUCCESS;
 err_free_array_d:
  cudaFree(array_d);
 err:
  return FAILURE;
}

/**
 * Reads a large array in random positions in parallel using a CUDA-device.
 */
__global__
void intern_memory_intensive_cache_unfriendly_gpu(uint64_t *array, 
                                                  uint64_t index) {
  for (uint64_t i = 0; i < 10000; i++) {
    index = array[index];
  }
}

/**
 * A cache unfriendly memory intensive routine that reads a
 * large array in random positions in parallel using CUDA.
 */
__host__
error_t memory_intensive_cache_unfriendly_gpu(double duration) {
  printf("GPU Cache Unfriendly Memory Intensive Benchmark\n");   
  time_t start, end;
  time(&start);
  dim3 blocks;
  dim3 threads_per_block;
  KERNEL_CONFIGURE(MAX_BLOCK_PER_DIMENSION, blocks, threads_per_block);
  uint64_t *array_h = (uint64_t *)malloc(LARGE_ARRAY_SIZE * sizeof(uint64_t));

  // Initialize the array elements with a random number in array range, 
  // but without be the same number of its index.
  OMP(omp parallel for)
  for (uint64_t i = 0; i < LARGE_ARRAY_SIZE; i++) {
    uint64_t num;
    do {
      num = random_uint64(LARGE_ARRAY_SIZE);
    } while (i == num);
    array_h[i] = num;
  }
  uint64_t *array_d;
  CHK_CU_SUCCESS(cudaMalloc((void**)&array_d, LARGE_ARRAY_SIZE 
                            * sizeof(uint64_t)), err_free_array_d);
  CHK_CU_SUCCESS(cudaMemcpy(array_d, array_h, LARGE_ARRAY_SIZE 
                            * sizeof(uint64_t), cudaMemcpyHostToDevice), err);

  // Reads a random position on array during a defined time.
  uint64_t index;
  do {
    index = random_uint64(LARGE_ARRAY_SIZE);
    intern_memory_intensive_cache_unfriendly_gpu<<<blocks, threads_per_block>>>
      (array_d, index);
    time(&end);
  } while (difftime(end, start) < duration);
  CHK_CU_SUCCESS(cudaMemcpy(array_h, array_d, LARGE_ARRAY_SIZE 
                            * sizeof(uint64_t), cudaMemcpyDeviceToHost), err);
  cudaFree(array_d);
  free(array_h);
  return SUCCESS;
 err_free_array_d:
  cudaFree(array_d);
 err:
  return FAILURE;
}

/**
 * A memory copy routine between two arrays in parallel using a CUDA-device.
 */
__global__
void intern_memory_copy_intensive_gpu(uint64_t *array1, uint64_t *array2) {
  for (uint64_t i = 0; i < 10000; i++) {
    array1[THREAD_GLOBAL_INDEX] = array2[THREAD_GLOBAL_INDEX];
    array2[THREAD_GLOBAL_INDEX] = array1[THREAD_GLOBAL_INDEX];
  }
}

/**
 * A memory copy routine between two arrays in parallel using CUDA.
 */
error_t memory_copy_intensive_gpu(double duration) {
  printf("GPU Memory Copy Intensive Benchmark\n");
  time_t start, end;
  time(&start);
  dim3 blocks;
  dim3 threads_per_block;
  KERNEL_CONFIGURE(MAX_BLOCK_PER_DIMENSION, blocks, threads_per_block);
  uint64_t *array1_h = (uint64_t *)malloc(SMALL_ARRAY_SIZE * sizeof(uint64_t));
  uint64_t *array2_h = (uint64_t *)malloc(SMALL_ARRAY_SIZE * sizeof(uint64_t));
  // Initialize the array elements with its index number.
  OMP(omp parallel for)
  for (uint64_t i = 0; i < SMALL_ARRAY_SIZE; i++) {
    array1_h[i] = i;
    array2_h[i] = 0;
  }
  uint64_t *array1_d;
  CHK_CU_SUCCESS(cudaMalloc((void**)&array1_d, SMALL_ARRAY_SIZE 
                            * sizeof(uint64_t)), err_free_array1_d);
  CHK_CU_SUCCESS(cudaMemcpy(array1_d, array1_h, SMALL_ARRAY_SIZE 
                            * sizeof(uint64_t), cudaMemcpyHostToDevice), err);
  uint64_t *array2_d;
  CHK_CU_SUCCESS(cudaMalloc((void**)&array2_d, SMALL_ARRAY_SIZE
                            * sizeof(uint64_t)), err_free_array2_d);
  CHK_CU_SUCCESS(cudaMemcpy(array2_d, array2_h, SMALL_ARRAY_SIZE
                            * sizeof(uint64_t), cudaMemcpyHostToDevice), err);
  do {
    intern_memory_copy_intensive_gpu<<<blocks, threads_per_block>>>
      (array1_d, array2_d);
    time(&end);
  } while (difftime(end, start) < duration);
  CHK_CU_SUCCESS(cudaMemcpy(array1_h, array1_d, SMALL_ARRAY_SIZE 
                            * sizeof(uint64_t), cudaMemcpyDeviceToHost), err);
  CHK_CU_SUCCESS(cudaMemcpy(array2_h, array2_d, SMALL_ARRAY_SIZE
                            * sizeof(uint64_t), cudaMemcpyDeviceToHost), err);
  cudaFree(array1_d);
  cudaFree(array2_d);
  free(array1_h);
  free(array2_h);
  return SUCCESS;
 err_free_array1_d:
  cudaFree(array1_d);
 err_free_array2_d:
  cudaFree(array2_d);
 err:
  return FAILURE;
}


/**
 * A memory copy routine between two arrays in parallel using a CUDA-device.
 */
__global__
void intern_memory_random_copy_intensive_gpu(uint64_t *array1, 
                                             uint64_t *array2,
                                             uint64_t index) {
  for (uint64_t i = 0; i < 10000; i++) {
    array1[THREAD_GLOBAL_INDEX] = array2[index];
    array2[THREAD_GLOBAL_INDEX] = array1[index];
  }
}

/**
 * A memory copy routine between two arrays in parallel using CUDA.
 */
error_t memory_random_copy_intensive_gpu(double duration) {
  printf("GPU Memory Copy Intensive Benchmark\n");
  time_t start, end;
  time(&start);
  dim3 blocks;
  dim3 threads_per_block;
  KERNEL_CONFIGURE(MAX_BLOCK_PER_DIMENSION, blocks, threads_per_block);
  uint64_t *array1_h = (uint64_t *)malloc(SMALL_ARRAY_SIZE * sizeof(uint64_t));
  uint64_t *array2_h = (uint64_t *)malloc(SMALL_ARRAY_SIZE * sizeof(uint64_t));
  // Initialize the array elements with its index number.
  OMP(omp parallel for)
    for (uint64_t i = 0; i < SMALL_ARRAY_SIZE; i++) {
      array1_h[i] = i;
      array2_h[i] = 0;
    }
  uint64_t *array1_d;
  CHK_CU_SUCCESS(cudaMalloc((void**)&array1_d, SMALL_ARRAY_SIZE
                            * sizeof(uint64_t)), err_free_array1_d);
  CHK_CU_SUCCESS(cudaMemcpy(array1_d, array1_h, SMALL_ARRAY_SIZE
                            * sizeof(uint64_t), cudaMemcpyHostToDevice), err);
  uint64_t *array2_d;
  CHK_CU_SUCCESS(cudaMalloc((void**)&array2_d, SMALL_ARRAY_SIZE
                            * sizeof(uint64_t)), err_free_array2_d);
  CHK_CU_SUCCESS(cudaMemcpy(array2_d, array2_h, SMALL_ARRAY_SIZE
                            * sizeof(uint64_t), cudaMemcpyHostToDevice), err);
  do {
    uint64_t index = random_uint64(SMALL_ARRAY_SIZE);
    intern_memory_random_copy_intensive_gpu<<<blocks, threads_per_block>>>
      (array1_d, array2_d, index);
    time(&end);
  } while (difftime(end, start) < duration);
  CHK_CU_SUCCESS(cudaMemcpy(array1_h, array1_d, SMALL_ARRAY_SIZE
                            * sizeof(uint64_t), cudaMemcpyDeviceToHost), err);
  CHK_CU_SUCCESS(cudaMemcpy(array2_h, array2_d, SMALL_ARRAY_SIZE
                            * sizeof(uint64_t), cudaMemcpyDeviceToHost), err);
  cudaFree(array1_d);
  cudaFree(array2_d);
  free(array1_h);
  free(array2_h);
  return SUCCESS;
 err_free_array1_d:
  cudaFree(array1_d);
 err_free_array2_d:
  cudaFree(array2_d);
 err:
  return FAILURE;
}
