/**
 * CPU-based functions of the intensive benchmark for power measurement
 * experiments
 *
 * Created on: 2013-05-28
 * Author: Sidney Pontes Filho
 */

// totem includes
#include "benchmark.h"

/**
 * A cpu-intensive routine that computes some mathematical operations
 * in parallel using OMP.
 */
error_t compute_intensive_cpu(double duration) {
  printf("CPU Intensive Benchmark\n");
  time_t start, end;
  time(&start);
  OMP(omp parallel)
  {
    double a, b;
    do {
      a = drand48();
      b = drand48() * DBL_MAX;
      b = (a * b) - (a / b) + (pow(a, b));
      time(&end);
    } while (difftime(end, start) < duration);
  }
  return SUCCESS;
}

/**
 * A CPU and memory intensive routine that reads a large array in
 * random positions and uses the read number to calculate
 * multiplication, addition or subtraction in parallel using OMP.
 */
error_t compute_memory_intensive_cpu(double duration) {
  printf("CPU and Memory Intensive Benchmark\n");
  time_t start, end;
  time(&start);
  uint64_t *ar = (uint64_t *)malloc(LARGE_ARRAY_SIZE * sizeof(uint64_t));

  // Initialize the array elements with a random number
  // in a range of the array size, but without be the same
  // number of its index.
  OMP(omp parallel for)
  for (uint64_t i = 0; i < LARGE_ARRAY_SIZE; i++) {
    uint64_t num;
    do {
      num = random_uint64(LARGE_ARRAY_SIZE);
    } while (i == num);
    ar[i] = num;
  }

  // Reads a random position on array, then that read number is
  // used to calculate multiplication and modulus with a random
  // number during a defined time.
  OMP(omp parallel)
  {
    uint64_t index = random_uint64(LARGE_ARRAY_SIZE);
    uint64_t num;
    do {

      // "(uint64_t)(-1)" is equal to the maximum possible value.
      num = random_uint64((uint64_t)(-1));
      index = (num * ar[index]) % LARGE_ARRAY_SIZE;
      time(&end);
    } while (difftime(end, start) < duration);
  }
  free(ar);
  return SUCCESS;
}

/**
 * A cache friendly memory intensive routine that reads a array
 * in ascending order in parallel using OMP.
 */
error_t memory_intensive_cache_friendly_cpu(double duration) {
  printf("Cache Friendly Memory Intensive Benchmark\n");
  time_t start, end;
  time(&start);
  unsigned char *ar =
    (unsigned char *)malloc(SMALL_ARRAY_SIZE * sizeof(unsigned char));

  // Initialize the array elements with the same number of the
  // index, but the last element is zero to return to beginning
  // of the array.
  OMP(omp parallel for)
  for (uint64_t i = 0; i < SMALL_ARRAY_SIZE; i++) {
    ar[i] = i + 1;
  }
  ar[SMALL_ARRAY_SIZE - 1] = 0;

  // Reads the array in ascending order during a defined time.
  OMP(omp parallel)
  {
    uint64_t index = 0;
    do {
      index = ar[index];
      time(&end);
    } while (difftime(end, start) < duration);
  }
  free(ar);
  return SUCCESS;
}

/**
 * A cache unfriendly memory intensive routine that reads a
 * large array in random positions in parallel using OMP.
 */
error_t memory_intensive_cache_unfriendly_cpu(double duration) {
  printf("Cache Unfriendly Memory Intensive Benchmark\n"); 
  time_t start, end;
  time(&start);
  uint64_t *ar = (uint64_t *)malloc(LARGE_ARRAY_SIZE * sizeof(uint64_t));

  // Initialize the array elements with a random number
  // in a range of the array size, but without be the same
  // number of its index.
  OMP(omp parallel for)
  for (uint64_t i = 0; i < LARGE_ARRAY_SIZE; i++) {
    uint64_t num;
    do {
      num = random_uint64(LARGE_ARRAY_SIZE);
    } while (i == num);
    ar[i] = num;
  }

  // Reads a random position on array during a defined time.
  OMP(omp parallel)
  {
    uint64_t index = random_uint64(LARGE_ARRAY_SIZE);
    do {
      index = ar[index];
      time(&end);
    } while (difftime(end, start) < duration);
  }
  free(ar);
  return SUCCESS;
}

/**
 * A memory copy routine between two arrays in parallel using OMP.
 */
error_t memory_copy_intensive_cpu(double duration) {
  printf("Memory Copy Intensive Benchmark\n");
  time_t start, end;
  time(&start);
  uint64_t *ar1 = (uint64_t *)malloc(SMALL_ARRAY_SIZE * sizeof(uint64_t));
  uint64_t *ar2 = (uint64_t *)malloc(SMALL_ARRAY_SIZE * sizeof(uint64_t));

  // Initialize the array elements with its index number.
  OMP(omp parallel for)
  for (uint64_t i = 0; i < SMALL_ARRAY_SIZE; i++) {
    ar1[i] = i;
    ar2[i] = 0;
  }
  OMP(omp parallel)
  {
    uint64_t nthreads = omp_get_num_threads();
    uint64_t th_id = omp_get_thread_num();

    // Calculates the size of the piece of array for thread in charge
    uint64_t piece_array_size = (SMALL_ARRAY_SIZE / nthreads)
      + (((SMALL_ARRAY_SIZE % nthreads) > th_id) ? 1 : 0);

    // Calculates the beginning index of the piece of array for thread
    // in charge
    uint64_t piece_array_index = th_id * (SMALL_ARRAY_SIZE / nthreads)
      + (((SMALL_ARRAY_SIZE % nthreads) > th_id)
         ? th_id : (SMALL_ARRAY_SIZE % nthreads));
    do {

      // Copy from ar1 to ar2
      memcpy(&ar2[piece_array_index], &ar1[piece_array_index],
             piece_array_size * sizeof(uint64_t));
      OMP(omp barrier)

      // Copy from ar2 to ar1
      memcpy(&ar1[piece_array_index], &ar2[piece_array_index],
             piece_array_size * sizeof(uint64_t));
      OMP(omp barrier)
      time(&end);
    } while (difftime(end, start) < duration);
  }
  free(ar1);
  free(ar2);
  return SUCCESS;
}

/**
 * A random memory copy routine between two arrays in parallel using OMP.
 */
error_t memory_random_copy_intensive_cpu(double duration) {
  printf("Memory Copy Intensive Benchmark\n");
  time_t start, end;
  time(&start);
  uint64_t *ar1 = (uint64_t *)malloc(SMALL_ARRAY_SIZE * sizeof(uint64_t));
  uint64_t *ar2 = (uint64_t *)malloc(SMALL_ARRAY_SIZE * sizeof(uint64_t));

  // Initialize the array elements with its index number.
  OMP(omp parallel for)
    for (uint64_t i = 0; i < SMALL_ARRAY_SIZE; i++) {
      ar1[i] = i;
      ar2[i] = 0;
    }
  OMP(omp parallel)
  {
    uint64_t nthreads = omp_get_num_threads();
    uint64_t th_id = omp_get_thread_num();
    
    // Calculates the size of the piece of array for thread in charge
    uint64_t piece_array_size = (SMALL_ARRAY_SIZE / nthreads)
      + (((SMALL_ARRAY_SIZE % nthreads) > th_id) ? 1 : 0);
    
    // Calculates the beginning index of the piece of array for thread
    // in charge
    uint64_t piece_array_index = th_id * (SMALL_ARRAY_SIZE / nthreads)
      + (((SMALL_ARRAY_SIZE % nthreads) > th_id)
         ? th_id : (SMALL_ARRAY_SIZE % nthreads));
    do {
      uint64_t th_id_random = random_uint64(nthreads);
      
      // Calculates the size of the piece of array for a random position
      uint64_t piece_array_size_random = (SMALL_ARRAY_SIZE / nthreads)
        + (((SMALL_ARRAY_SIZE % nthreads) > th_id_random) ? 1 : 0);
      
      // Calculates the beginning index of the piece of array for a random
      // position
      uint64_t piece_array_index_random = th_id_random 
        * (SMALL_ARRAY_SIZE / nthreads) 
        + (((SMALL_ARRAY_SIZE % nthreads) > th_id_random)
           ? th_id_random : (SMALL_ARRAY_SIZE % nthreads));

      // Copy from ar1 to ar2
      memcpy(&ar2[piece_array_index_random], &ar1[piece_array_index],
             ((piece_array_size > piece_array_size_random) 
              ? piece_array_size_random : piece_array_size) 
             * sizeof(uint64_t));
      OMP(omp barrier)
          
      // Copy from ar2 to ar1
      memcpy(&ar1[piece_array_index_random], &ar2[piece_array_index],
             ((piece_array_size > piece_array_size_random)
              ? piece_array_size_random : piece_array_size)
             * sizeof(uint64_t));

      OMP(omp barrier)
      time(&end);
    } while (difftime(end, start) < duration);
  }
  free(ar1);
  free(ar2);
  return SUCCESS;
}
