/**
 * Memory Copy Intensive for Power Measurement Experiments
 *
 * Created on: 2013-05-27
 * Author: Sidney Pontes Filho
 */

// totem includes
#include "totem_intensive_benchmark.h"

/**
 * A memory copy routine between two arrays in parallel using OMP.
 * @param[in] argv[1] duration of the running time in seconds
 */
int main (int argc, char *argv[]) {
  time_t start, end;
  time(&start);
  if (argc > 2) {
    help_message();
    exit(EXIT_FAILURE);
  }
  double duration = (argc == 2) ? atof(argv[1]) : DURATION_DEFAULT;
  if (duration == 0.0f) {
    help_message();
    exit(EXIT_FAILURE);
  }
  uint64_t *ar1 = (uint64_t *) malloc (SMALL_ARRAY_SIZE * sizeof(uint64_t));
  uint64_t *ar2 = (uint64_t *) malloc (SMALL_ARRAY_SIZE * sizeof(uint64_t));

  // Initialize the array elements with its index number.
  OMP(omp parallel for)
  for (uint64_t i = 0; i < SMALL_ARRAY_SIZE; i++) {
    ar1[i] = i;
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
  return 0;
}
