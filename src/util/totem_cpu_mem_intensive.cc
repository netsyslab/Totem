/** 
 * CPU and Memory Intensive Benchmark for Power Measurement Experiments
 * 
 * Created on: 2013-05-28
 * Author: Sidney Pontes Filho
 */

// totem includes
#include "totem_intensive_benchmark.h"

/**
 * A CPU and memory intensive routine that reads a large array in 
 * random positions and uses the read number to calculate 
 * multiplication and modulus in parallel using OMP.
 * @param[in] argv[1] duration of the running time in seconds 
 */
int main (int argc, char *argv[]) {
  double duration;
  time_t start, end;
  time(&start);
  if (argc > 2) {
    help_message();
    exit(EXIT_FAILURE);
  } 
  duration = (argc == 2) ? atof(argv[1]) : DURATION_DEFAULT;
  if (duration == 0.0f) {
    help_message();
    exit(EXIT_FAILURE);
  }
  uint64_t *ar = (uint64_t *) malloc (LARGE_ARRAY_SIZE * sizeof(uint64_t));

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
  return 0;
}
