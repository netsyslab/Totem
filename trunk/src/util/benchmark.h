/**
 * Declarations of common functions and definitions of the intensive benchmark
 * for power measurement experiments
 *
 * Created on: 2013-05-28
 * Author: Sidney Pontes Filho
 */

#ifndef BENCHMARK_H
#define BENCHMARK_H

// system includes
#include <inttypes.h>
#include <sys/types.h>

// totem includes
#include "totem_comdef.h"

const uint64_t SMALL_ARRAY_SIZE = 1000;
const uint64_t LARGE_ARRAY_SIZE = 1000000;

/*
 * Benchmark function data type
 */
typedef error_t (*benchmark_func_t)(double);

/**
 * Random Unsigned Integer Generator.
 * @param[in] max_range generates a random number between 0 and max_range
 * @return the generated number
 */
inline uint64_t random_uint64(uint64_t max_range) {
  return (uint64_t)(drand48() * (double)max_range);
}

/**
 * A cpu-intensive routine that computes some mathematical operations
 * in parallel using OMP.
 * @param[in] duration duration of the running time in seconds
 * @return generic success or failure
 */
error_t compute_intensive_cpu(double duration);

/**
 * A CPU and memory intensive routine that reads a large array in
 * random positions and uses the read number to calculate
 * multiplication, addition or subtraction in parallel using OMP.
 * @param[in] duration duration of the running time in seconds
 * @return generic success or failure
 */
error_t compute_memory_intensive_cpu(double duration);

/**
 * A cache friendly memory intensive routine that reads a array
 * in ascending order in parallel using OMP.
 * @param[in] duration duration of the running time in seconds
 * @return generic success or failure
 */
error_t memory_intensive_cache_friendly_cpu(double duration);

/**
 * A cache unfriendly memory intensive routine that reads a
 * large array in random positions in parallel using OMP.
 * @param[in] duration duration of the running time in seconds
 * @return generic success or failure
 */
error_t memory_intensive_cache_unfriendly_cpu(double duration);

/**
 * A memory copy routine between two arrays in parallel using OMP.
 * @param[in] duration duration of the running time in seconds
 * @return generic success or failure
 */
error_t memory_copy_intensive_cpu(double duration);

/**
 * A gpu-intensive routine that computes some mathematical operations
 * in parallel using CUDA.
 * @param[in] duration duration of the running time in seconds
 * @return generic success or failure
 */
error_t compute_intensive_gpu(double duration);

/**
 * A GPU and memory intensive routine that reads a large array in
 * random positions and uses the read number to calculate
 * multiplication and modulus in parallel using CUDA.
 * @param[in] duration duration of the running time in seconds
 * @return generic success or failure
 */
error_t compute_memory_intensive_gpu(double duration);

/**
 * A cache friendly memory intensive routine that reads a array
 * in ascending order in parallel using CUDA.
 * @param[in] duration duration of the running time in seconds
 * @return generic success or failure
 */
error_t memory_intensive_cache_friendly_gpu(double duration);

/**
 * A cache unfriendly memory intensive routine that reads a
 * large array in random positions in parallel using CUDA.
 * @param[in] duration duration of the running time in seconds
 * @return generic success or failure
 */
error_t memory_intensive_cache_unfriendly_gpu(double duration);

/**
 * A memory copy routine between two arrays in parallel using CUDA.
 * @param[in] duration duration of the running time in seconds
 * @return generic success or failure
 */
error_t memory_copy_intensive_gpu(double duration);

#endif  // BENCHMARK_H
