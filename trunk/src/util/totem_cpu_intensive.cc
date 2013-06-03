/** 
 * CPU Intensive benchmark for power measurement experiments
 * 
 * Created on: 2013-05-17
 * Author: Sidney Pontes Filho
 */

// totem includes
#include "totem_intensive_benchmark.h"

/**
 * A cpu-intensive routine that computes some mathematical operations
 * in parallel using OMP.
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
  return 0;
}
