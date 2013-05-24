/** 
 * CPU Intensive benchmark for power measurement experiments
 * 
 * Created on: 2013-05-17
 * Author: Sidney Pontes Filho
 */

// system includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// totem includes
#include "totem_comdef.h"

/**
 * Defines the duration of the running time when user does not
 * pass any argument.
 */
#define DURATION_DEFAULT 1.0

// Help message is displayed when user passes a wrong argument.
void help_message() { 
  printf("Usage: ./[EXECUTABLE]\n");
  printf("  or:  ./[EXECUTABLE] [NUMBER]\n");
  printf("  or:  ./[EXECUTABLE] [NUMBER].[NUMBER]\n");
} 

/**
 * A cpu-intensive routine that computes products and subtractions
 * of a elements in an array in parallel using OMP. 
 * @param[in] argv[1] duration of the running time in seconds
 */
int main (int argc, char *argv[]) {
  double a = 1.0;
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
    OMP(omp parallel for)
    for (long x = 0; x < 1000; x++) {
      a = (double)x - a * 0.00001;
      x = x - a * a;      
      time(&end);

      // Stop condition. 
      // OpenMP does not allow 'break' inside a parallel region.
      if (difftime(end, start) >= duration) {
        x = 10000000;
      }
    }
  }  
  return 0;
}
