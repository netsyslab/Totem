/** 
 * Cache friendly Memory Intensive for Power Measurement Experiments
 * 
 * Created on: 2013-05-17
 * Author: Sidney Pontes Filho
 */

// totem includes
#include "totem_intensive_benchmark.h"

/**
 * A cache friendly memory intensive routine that reads a array
 * in ascending order in parallel using OMP.
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
  unsigned char *ar = (unsigned char *) malloc (SMALL_ARRAY_SIZE  
                                                * sizeof(unsigned char));
  
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
  return 0;  
}
