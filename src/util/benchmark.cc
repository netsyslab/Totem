/**
 * Main function for power measurement experiments
 *
 * Created on: 2013-05-31
 * Author: Sidney Pontes Filho
 */

// totem includes
#include "benchmark.h"

/**
 * Help message is displayed when user passes a wrong argument.
 */
void help_message(const char *exec_name) {
  printf("Usage: %s [BENCHMARK] [DURATION]\n", exec_name);
  printf("\nBENCHMARK:\n");
  printf("1 - CPU Intensive Benchmark\n");
  printf("2 - CPU and Memory Intensive Benchmark\n");
  printf("3 - Cache Friendly Memory Intensive Benchmark\n");
  printf("4 - Cache Unfriendly Memory Intensive Benchmark\n");
  printf("5 - Memory Copy Intensive Benchmark\n");
  printf("\nDURATION: (in seconds)\n");
  printf("Usage: [NUMBER]\n");
  printf("  or:  [NUMBER].[NUMBER]\n");
}

/**
 * Gets the arguments of command line
 * @param[in] argc the value of argc of Main Function
 * @param[in] argv the array of argv of Main Function
 * @param[out] benchmark number of the function
 * @param[out] duration duration of the running time
 */
void get_options(int argc, char *argv[], int *benchmark, double *duration) {
  if (argc != 3) {
    help_message(argv[0]);
    exit(EXIT_FAILURE);
  }

  // Benchmark argument between 1 and 5, and index between 0 and 4.
  *benchmark = (int)(*argv[1] - '0') - 1;
  if (*benchmark < 0 || *benchmark > 4) {
    help_message(argv[0]);
    exit(EXIT_FAILURE);
  }
  *duration = atof(argv[2]);
  if (duration <= 0) {
    help_message(argv[0]);
    exit(EXIT_FAILURE);
  }
}

/*
 * Array with the name of the functions
 */
benchmark_func_t benchmark_func[] =
  {compute_intensive_cpu,
   compute_memory_intensive_cpu,
   memory_intensive_cache_friendly_cpu,
   memory_intensive_cache_unfriendly_cpu,
   memory_copy_intensive_cpu};

/**
 * The main function for the intensive benchmarks.
 * @param[in] argv[1] benchmark number
 * @param[in] argv[2] duration of the running time
 */
int main(int argc, char *argv[]) {
  int benchmark;
  double duration;
  get_options(argc, argv, &benchmark, &duration);
  printf("RUNNING...\n");
  benchmark_func[benchmark](duration);
  printf("FINISHED\n");
  return 0;
}

