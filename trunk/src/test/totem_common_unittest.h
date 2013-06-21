/**
 * Contains common definitions for unit tests
 *
 *  Created on: 2011-03-18
 *      Author: Abdullah Gharaibeh
 */

#ifndef TOTEM_COMMON_UNITTEST_H
#define TOTEM_COMMON_UNITTEST_H

// system includes
#include "gtest/gtest.h"

// totem includes
#include "totem.h"
#include "totem_comdef.h"
#include "totem_graph.h"
#include "totem_alg.h"
#include "totem_mem.h"
#include "totem_util.h"

/**
 *  Defines a relative path of a graph file
 */
#define DATA_FOLDER(graph_file) "../../data/"graph_file
#define TEMP_FOLDER(graph_file) "/tmp/"graph_file

#define CUDA_CHECK_VERSION()               \
  do {                                     \
    if(check_cuda_version() != SUCCESS) {  \
      exit(EXIT_FAILURE);                  \
    }                                      \
  } while (0)

/**
 * A simple macro to do basic true/false condition testing for kernels
 * TODO(abdullah): change the way state is tested to use standard report from
 * the GTest framework as follows:
 * 1. to use the macro to test (so the code will be simple).
 * 2. still have the variable to store the line number where it fails or -1
 * otherwise.
 * 3. in the test fixture you would copy back the variable with the line number
 * and expects -1.
 */
#define KERNEL_EXPECT_TRUE(stmt)                \
  do {                                          \
    if (!(stmt)) {                              \
      printf("Error line: %d\n", __LINE__);     \
    }                                           \
  } while(0)

// Hybrid algorithms attributes
const float CPU_SHARE_ZERO = 0;
const float CPU_SHARE_ONE_THIRD = 0.33;
const bool GPU_PAR_RANDOMIZED_DISABLED = false;
const bool GPU_PAR_RANDOMIZED_ENABLED = true;
const int  GPU_COUNT_ONE = 1;
PRIVATE totem_attr_t totem_attrs[] = {
  // CPU only
  {PAR_RANDOM, PLATFORM_CPU, GPU_COUNT_ONE, GPU_GRAPH_MEM_DEVICE, 
   GPU_PAR_RANDOMIZED_DISABLED, CPU_SHARE_ZERO, MSG_SIZE_ZERO, MSG_SIZE_ZERO},
  // GPU only
  {PAR_RANDOM, PLATFORM_GPU, GPU_COUNT_ONE, GPU_GRAPH_MEM_DEVICE, 
   GPU_PAR_RANDOMIZED_DISABLED, CPU_SHARE_ZERO, MSG_SIZE_ZERO, MSG_SIZE_ZERO},
  // Multi GPU
  {PAR_RANDOM, PLATFORM_GPU, get_gpu_count(), GPU_GRAPH_MEM_DEVICE,
   GPU_PAR_RANDOMIZED_DISABLED, CPU_SHARE_ZERO, MSG_SIZE_ZERO, MSG_SIZE_ZERO},
  // Hybrid 1 CPU + 1 GPU, memory mapped GPU partition disabled
  {PAR_RANDOM, PLATFORM_HYBRID, GPU_COUNT_ONE, GPU_GRAPH_MEM_DEVICE,
   GPU_PAR_RANDOMIZED_DISABLED, CPU_SHARE_ONE_THIRD, MSG_SIZE_ZERO, 
   MSG_SIZE_ZERO},
  // Hybrid 1 CPU + all GPU, memory mapped GPU partition disabled
  {PAR_RANDOM, PLATFORM_HYBRID, get_gpu_count(), GPU_GRAPH_MEM_DEVICE,
   GPU_PAR_RANDOMIZED_DISABLED, CPU_SHARE_ONE_THIRD, MSG_SIZE_ZERO, 
   MSG_SIZE_ZERO},
  // Hybrid 1 CPU + 1 GPU, memory mapped GPU partition enabled
  {PAR_RANDOM, PLATFORM_HYBRID, GPU_COUNT_ONE, GPU_GRAPH_MEM_MAPPED,
   GPU_PAR_RANDOMIZED_DISABLED, CPU_SHARE_ONE_THIRD, MSG_SIZE_ZERO, 
   MSG_SIZE_ZERO},
  // Hybrid 1 CPU + all GPU, memory mapped GPU partition enabled
  {PAR_RANDOM, PLATFORM_HYBRID, get_gpu_count(), GPU_GRAPH_MEM_MAPPED,
   GPU_PAR_RANDOMIZED_DISABLED, CPU_SHARE_ONE_THIRD, MSG_SIZE_ZERO, 
   MSG_SIZE_ZERO},
  // Hybrid 1 CPU + all GPU, memory mapped GPU partition enabled and cross-gpu 
  // vertex placement randomized
  {PAR_RANDOM, PLATFORM_HYBRID, get_gpu_count(), GPU_GRAPH_MEM_MAPPED,
   GPU_PAR_RANDOMIZED_ENABLED, CPU_SHARE_ONE_THIRD, MSG_SIZE_ZERO, 
   MSG_SIZE_ZERO}
};

#endif // TOTEM_COMMON_UNITTEST_H
