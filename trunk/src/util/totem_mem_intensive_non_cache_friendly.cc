/** 
 * Cache unfriendly Memory Intensive Benchmark 
 * for Power Measurement Experiments
 * 
 * Created on: 2013-05-17
 * Author: Sidney Pontes Filho
 */

// system includes
#include <inttypes.h>
#include <limits.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>

// totem includes
#include "totem_comdef.h"

/**
 * Defines the duration of the running time when user does not
 * pass any argument.
 */
#define DURATION_DEFAULT 1.0

/**
 * Defines the array size.
 */
#define ARRAY_SIZE 1000000

/**
 * Random Integer Generator. 
 */
uint64_t random(uint64_t x) {
  return (uint64_t)(drand48() * (double)x);
}

/** 
 * Help message is displayed when user passes a wrong argument.
 */
void help_message() {
  printf("Usage: ./[EXECUTABLE]\n");
  printf("  or:  ./[EXECUTABLE] [NUMBER]\n");
  printf("  or:  ./[EXECUTABLE] [NUMBER].[NUMBER]\n");
}

/**
 * A non-cache friendly memory intensive routine that reads a 
 * large array in random positions in parallel using OMP.
 * @param[in] argv[1] duration of the running time in seconds 
 */
int main (int argc, char *argv[]) {
  double duration;
  time_t start, end;
  time(&start);
  if (argc > 2) {
    help_message();
    exit(EXIT_FAILURE);
  } else {
    duration = (argc == 2) ? atof(argv[1]) : DURATION_DEFAULT;
    if (duration == 0.0f) {
      help_message();
      exit(EXIT_FAILURE);
    }
    uint64_t *ar;
    ar = (uint64_t *) malloc (ARRAY_SIZE * sizeof(uint64_t));

    // Initialize the array elements with a random number 
    // in a range of the array size, but without be the same
    // number of its index.
    OMP(omp parallel for)
    for (uint64_t i = 0; i < ARRAY_SIZE; i++) {
      uint64_t num;
      do {
        num = random(ARRAY_SIZE);
      } while (ar[i] == num);
      ar[i] = num;
    }

    // Reads a random position on array during a defined time.
    OMP(omp parallel)
    {
      uint64_t index = random(ARRAY_SIZE);
      do {
        index = ar[index];
        time(&end);
      } while (difftime(end, start) < duration);
    }
  }
  return 0;
}
