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
 *  Defines a relative path of a graph file.
 */
#define DATA_FOLDER(graph_file) "../data/"graph_file
#define TEMP_FOLDER(graph_file) "/tmp/"graph_file

#define CUDA_CHECK_VERSION()               \
  do {                                     \
    if (check_cuda_version() != SUCCESS) { \
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
  } while (0)

// Hybrid algorithms attributes
const float CPU_SHARE_ZERO = 0;
const float CPU_SHARE_ONE_THIRD = 0.33;
const bool GPU_PAR_RANDOMIZED_DISABLED = false;
const bool GPU_PAR_RANDOMIZED_ENABLED = true;
const bool VERTEX_IDS_SORTED = true;
const bool VERTEX_IDS_NOT_SORTED = false;
const bool EDGE_SORT_DSC = false;
const bool EDGE_SORT_BY_DEGREE = false;
const bool COMPRESSED_VERTICES_SUPPORTED = false;
const bool SEPARATE_SINGLETONS = false;
const int  GPU_COUNT_ONE = 1;
const float LAMBDA = 0;
PRIVATE totem_attr_t totem_attrs[] = {
  {  // (0) CPU only
    PAR_RANDOM, PLATFORM_CPU, GPU_COUNT_ONE, GPU_GRAPH_MEM_DEVICE,
    GPU_PAR_RANDOMIZED_DISABLED, VERTEX_IDS_SORTED,
    EDGE_SORT_DSC, EDGE_SORT_BY_DEGREE, COMPRESSED_VERTICES_SUPPORTED,
    SEPARATE_SINGLETONS, LAMBDA,
    CPU_SHARE_ZERO, MSG_SIZE_ZERO, MSG_SIZE_ZERO
  },
  {  // (1) GPU only
    PAR_RANDOM, PLATFORM_GPU, GPU_COUNT_ONE, GPU_GRAPH_MEM_DEVICE,
    GPU_PAR_RANDOMIZED_DISABLED, VERTEX_IDS_SORTED,
    EDGE_SORT_DSC, EDGE_SORT_BY_DEGREE, COMPRESSED_VERTICES_SUPPORTED,
    SEPARATE_SINGLETONS, LAMBDA,
    CPU_SHARE_ZERO, MSG_SIZE_ZERO, MSG_SIZE_ZERO
  },
  {  // (2) Multi GPU
    PAR_RANDOM, PLATFORM_GPU, get_gpu_count(), GPU_GRAPH_MEM_DEVICE,
    GPU_PAR_RANDOMIZED_DISABLED, VERTEX_IDS_SORTED,
    EDGE_SORT_DSC, EDGE_SORT_BY_DEGREE, COMPRESSED_VERTICES_SUPPORTED,
    SEPARATE_SINGLETONS, LAMBDA,
    CPU_SHARE_ZERO, MSG_SIZE_ZERO, MSG_SIZE_ZERO
  },

  {  // (3) Hybrid CPU + 1 GPU
    PAR_RANDOM, PLATFORM_HYBRID, GPU_COUNT_ONE, GPU_GRAPH_MEM_DEVICE,
    GPU_PAR_RANDOMIZED_DISABLED, VERTEX_IDS_NOT_SORTED,
    EDGE_SORT_DSC, EDGE_SORT_BY_DEGREE, COMPRESSED_VERTICES_SUPPORTED,
    SEPARATE_SINGLETONS, LAMBDA,
    CPU_SHARE_ONE_THIRD, MSG_SIZE_ZERO, MSG_SIZE_ZERO
  },
  {  // (4) Hybrid CPU + 1 GPU
    PAR_SORTED_ASC, PLATFORM_HYBRID, GPU_COUNT_ONE, GPU_GRAPH_MEM_DEVICE,
    GPU_PAR_RANDOMIZED_DISABLED, VERTEX_IDS_NOT_SORTED,
    EDGE_SORT_DSC, EDGE_SORT_BY_DEGREE, COMPRESSED_VERTICES_SUPPORTED,
    SEPARATE_SINGLETONS, LAMBDA,
    CPU_SHARE_ONE_THIRD, MSG_SIZE_ZERO, MSG_SIZE_ZERO
  },
  {  // (5) Hybrid CPU + 1 GPU
    PAR_SORTED_DSC, PLATFORM_HYBRID, GPU_COUNT_ONE, GPU_GRAPH_MEM_DEVICE,
    GPU_PAR_RANDOMIZED_DISABLED, VERTEX_IDS_NOT_SORTED,
    EDGE_SORT_DSC, EDGE_SORT_BY_DEGREE, COMPRESSED_VERTICES_SUPPORTED,
    SEPARATE_SINGLETONS, LAMBDA,
    CPU_SHARE_ONE_THIRD, MSG_SIZE_ZERO, MSG_SIZE_ZERO
  },

  {  // (6) Hybrid CPU + 1 GPU, sorted vertices
    PAR_RANDOM, PLATFORM_HYBRID, GPU_COUNT_ONE, GPU_GRAPH_MEM_DEVICE,
    GPU_PAR_RANDOMIZED_DISABLED, VERTEX_IDS_SORTED,
    EDGE_SORT_DSC, EDGE_SORT_BY_DEGREE, COMPRESSED_VERTICES_SUPPORTED,
    SEPARATE_SINGLETONS, LAMBDA,
    CPU_SHARE_ONE_THIRD, MSG_SIZE_ZERO, MSG_SIZE_ZERO
  },
  {  // (7) Hybrid CPU + 1 GPU, sorted vertices
    PAR_SORTED_ASC, PLATFORM_HYBRID, GPU_COUNT_ONE, GPU_GRAPH_MEM_DEVICE,
    GPU_PAR_RANDOMIZED_DISABLED, VERTEX_IDS_SORTED,
    EDGE_SORT_DSC, EDGE_SORT_BY_DEGREE, COMPRESSED_VERTICES_SUPPORTED,
    SEPARATE_SINGLETONS, LAMBDA,
    CPU_SHARE_ONE_THIRD, MSG_SIZE_ZERO, MSG_SIZE_ZERO
  },
  {  // (8) Hybrid CPU + 1 GPU, sorted vertices
    PAR_SORTED_DSC, PLATFORM_HYBRID, GPU_COUNT_ONE, GPU_GRAPH_MEM_DEVICE,
    GPU_PAR_RANDOMIZED_DISABLED, VERTEX_IDS_SORTED,
    EDGE_SORT_DSC, EDGE_SORT_BY_DEGREE, COMPRESSED_VERTICES_SUPPORTED,
    SEPARATE_SINGLETONS, LAMBDA,
    CPU_SHARE_ONE_THIRD, MSG_SIZE_ZERO, MSG_SIZE_ZERO
  },

  {  // (9) Hybrid CPU + 1 GPU (memory mapped GPU partition)
    PAR_RANDOM, PLATFORM_HYBRID, GPU_COUNT_ONE, GPU_GRAPH_MEM_MAPPED,
    GPU_PAR_RANDOMIZED_DISABLED, VERTEX_IDS_SORTED,
    EDGE_SORT_DSC, EDGE_SORT_BY_DEGREE, COMPRESSED_VERTICES_SUPPORTED,
    SEPARATE_SINGLETONS, LAMBDA,
    CPU_SHARE_ONE_THIRD, MSG_SIZE_ZERO, MSG_SIZE_ZERO
  },
  {  // (10) Hybrid CPU + 1 GPU (memory mapped GPU partition)
    PAR_SORTED_ASC, PLATFORM_HYBRID, GPU_COUNT_ONE, GPU_GRAPH_MEM_MAPPED,
    GPU_PAR_RANDOMIZED_DISABLED, VERTEX_IDS_SORTED,
    EDGE_SORT_DSC, EDGE_SORT_BY_DEGREE, COMPRESSED_VERTICES_SUPPORTED,
    SEPARATE_SINGLETONS, LAMBDA,
    CPU_SHARE_ONE_THIRD, MSG_SIZE_ZERO, MSG_SIZE_ZERO
  },
  {  // (11) Hybrid CPU + 1 GPU (memory mapped GPU partition)
    PAR_SORTED_DSC, PLATFORM_HYBRID, GPU_COUNT_ONE, GPU_GRAPH_MEM_MAPPED,
    GPU_PAR_RANDOMIZED_DISABLED, VERTEX_IDS_SORTED,
    EDGE_SORT_DSC, EDGE_SORT_BY_DEGREE, COMPRESSED_VERTICES_SUPPORTED,
    SEPARATE_SINGLETONS, LAMBDA,
    CPU_SHARE_ONE_THIRD, MSG_SIZE_ZERO, MSG_SIZE_ZERO
  },

  {  // (12) Hybrid CPU + all GPU
    PAR_RANDOM, PLATFORM_HYBRID, get_gpu_count(), GPU_GRAPH_MEM_DEVICE,
    GPU_PAR_RANDOMIZED_DISABLED, VERTEX_IDS_NOT_SORTED,
    EDGE_SORT_DSC, EDGE_SORT_BY_DEGREE, COMPRESSED_VERTICES_SUPPORTED,
    SEPARATE_SINGLETONS, LAMBDA,
    CPU_SHARE_ONE_THIRD, MSG_SIZE_ZERO, MSG_SIZE_ZERO
  },
  {  // (13) Hybrid CPU + all GPU
    PAR_SORTED_ASC, PLATFORM_HYBRID, get_gpu_count(), GPU_GRAPH_MEM_DEVICE,
    GPU_PAR_RANDOMIZED_DISABLED, VERTEX_IDS_NOT_SORTED,
    EDGE_SORT_DSC, EDGE_SORT_BY_DEGREE, COMPRESSED_VERTICES_SUPPORTED,
    SEPARATE_SINGLETONS, LAMBDA,
    CPU_SHARE_ONE_THIRD, MSG_SIZE_ZERO, MSG_SIZE_ZERO
  },
  {  // (14) Hybrid CPU + all GPU
    PAR_SORTED_DSC, PLATFORM_HYBRID, get_gpu_count(), GPU_GRAPH_MEM_DEVICE,
    GPU_PAR_RANDOMIZED_DISABLED, VERTEX_IDS_NOT_SORTED,
    EDGE_SORT_DSC, EDGE_SORT_BY_DEGREE, COMPRESSED_VERTICES_SUPPORTED,
    SEPARATE_SINGLETONS, LAMBDA,
    CPU_SHARE_ONE_THIRD, MSG_SIZE_ZERO, MSG_SIZE_ZERO
  },

  {  // (15) Hybrid CPU + all GPU, sorted vertices
    PAR_RANDOM, PLATFORM_HYBRID, get_gpu_count(), GPU_GRAPH_MEM_DEVICE,
    GPU_PAR_RANDOMIZED_DISABLED, VERTEX_IDS_SORTED,
    EDGE_SORT_DSC, EDGE_SORT_BY_DEGREE, COMPRESSED_VERTICES_SUPPORTED,
    SEPARATE_SINGLETONS, LAMBDA,
    CPU_SHARE_ONE_THIRD, MSG_SIZE_ZERO, MSG_SIZE_ZERO
  },
  {  // (16) Hybrid CPU + all GPU, sorted vertices
    PAR_SORTED_ASC, PLATFORM_HYBRID, get_gpu_count(), GPU_GRAPH_MEM_DEVICE,
    GPU_PAR_RANDOMIZED_DISABLED, VERTEX_IDS_SORTED,
    EDGE_SORT_DSC, EDGE_SORT_BY_DEGREE, COMPRESSED_VERTICES_SUPPORTED,
    SEPARATE_SINGLETONS, LAMBDA,
    CPU_SHARE_ONE_THIRD, MSG_SIZE_ZERO, MSG_SIZE_ZERO
  },
  {  // (17) Hybrid CPU + all GPU, sorted vertices
    PAR_SORTED_DSC, PLATFORM_HYBRID, get_gpu_count(), GPU_GRAPH_MEM_DEVICE,
    GPU_PAR_RANDOMIZED_DISABLED, VERTEX_IDS_SORTED,
    EDGE_SORT_DSC, EDGE_SORT_BY_DEGREE, COMPRESSED_VERTICES_SUPPORTED,
    SEPARATE_SINGLETONS, LAMBDA,
    CPU_SHARE_ONE_THIRD, MSG_SIZE_ZERO, MSG_SIZE_ZERO
  },

  {  // (18) Hybrid CPU + all GPU, randomized vertex placement
    PAR_RANDOM, PLATFORM_HYBRID, get_gpu_count(), GPU_GRAPH_MEM_DEVICE,
    GPU_PAR_RANDOMIZED_ENABLED, VERTEX_IDS_NOT_SORTED,
    EDGE_SORT_DSC, EDGE_SORT_BY_DEGREE, COMPRESSED_VERTICES_SUPPORTED,
    SEPARATE_SINGLETONS, LAMBDA,
    CPU_SHARE_ONE_THIRD, MSG_SIZE_ZERO, MSG_SIZE_ZERO
  },
  {  // (19) Hybrid CPU + all GPU, randomized vertex placement
    PAR_SORTED_ASC, PLATFORM_HYBRID, get_gpu_count(), GPU_GRAPH_MEM_DEVICE,
    GPU_PAR_RANDOMIZED_ENABLED, VERTEX_IDS_NOT_SORTED,
    EDGE_SORT_DSC, EDGE_SORT_BY_DEGREE, COMPRESSED_VERTICES_SUPPORTED,
    SEPARATE_SINGLETONS, LAMBDA,
    CPU_SHARE_ONE_THIRD, MSG_SIZE_ZERO, MSG_SIZE_ZERO
  },
  {  // (20) Hybrid CPU + all GPU, randomized vertex placement
    PAR_SORTED_DSC, PLATFORM_HYBRID, get_gpu_count(), GPU_GRAPH_MEM_DEVICE,
    GPU_PAR_RANDOMIZED_ENABLED, VERTEX_IDS_NOT_SORTED,
    EDGE_SORT_DSC, EDGE_SORT_BY_DEGREE, COMPRESSED_VERTICES_SUPPORTED,
    SEPARATE_SINGLETONS, LAMBDA,
    CPU_SHARE_ONE_THIRD, MSG_SIZE_ZERO, MSG_SIZE_ZERO
  },

  {  // (21) Hybrid CPU + all GPU, sorted vertices, randomized vertex placement
    PAR_RANDOM, PLATFORM_HYBRID, get_gpu_count(), GPU_GRAPH_MEM_DEVICE,
    GPU_PAR_RANDOMIZED_ENABLED, VERTEX_IDS_SORTED,
    EDGE_SORT_DSC, EDGE_SORT_BY_DEGREE, COMPRESSED_VERTICES_SUPPORTED,
    SEPARATE_SINGLETONS, LAMBDA,
    CPU_SHARE_ONE_THIRD, MSG_SIZE_ZERO, MSG_SIZE_ZERO
  },
  {  // (22) Hybrid CPU + all GPU, sorted vertices, randomized vertex placement
    PAR_SORTED_ASC, PLATFORM_HYBRID, get_gpu_count(), GPU_GRAPH_MEM_DEVICE,
    GPU_PAR_RANDOMIZED_ENABLED, VERTEX_IDS_SORTED,
    EDGE_SORT_DSC, EDGE_SORT_BY_DEGREE, COMPRESSED_VERTICES_SUPPORTED,
    SEPARATE_SINGLETONS, LAMBDA,
    CPU_SHARE_ONE_THIRD, MSG_SIZE_ZERO, MSG_SIZE_ZERO
  },
  {  // (23) Hybrid CPU + all GPU, sorted vertices, randomized vertex placement
    PAR_SORTED_DSC, PLATFORM_HYBRID, get_gpu_count(), GPU_GRAPH_MEM_DEVICE,
    GPU_PAR_RANDOMIZED_ENABLED, VERTEX_IDS_SORTED,
    EDGE_SORT_DSC, EDGE_SORT_BY_DEGREE, COMPRESSED_VERTICES_SUPPORTED,
    SEPARATE_SINGLETONS, LAMBDA,
    CPU_SHARE_ONE_THIRD, MSG_SIZE_ZERO, MSG_SIZE_ZERO
  },
};

// A macro that computes the number of elements of a static array.
#define STATIC_ARRAY_COUNT(array) sizeof(array) / sizeof(*array);

// The number of hybrid configurations in the totem_attr array.
static const int hybrid_configurations_count = STATIC_ARRAY_COUNT(totem_attrs);

// This is to allow testing the vanilla and the hybrid functions that are
// based on the Totem framework.
typedef struct {
  totem_attr_t* attr;  // Attributes for totem-based implementations.
  void*         func;  // The algorithm function to be tested.
  totem_cb_func_t hybrid_alloc;  // Allocates state for totem-based versions.
  totem_cb_func_t hybrid_free;  // Frees state for totem-based versions.
} test_param_t;

// Adds a test parameter to the passed vector of parameters.
static void PushParam(std::vector<test_param_t>* params_vector,
                      totem_attr_t* attr, void* func,
                      totem_cb_func_t hybrid_alloc = NULL,
                      totem_cb_func_t hybrid_free = NULL) {
  test_param_t param;
  param.attr = attr;
  param.func = func;
  param.hybrid_alloc = hybrid_alloc;
  param.hybrid_free = hybrid_free;
  params_vector->push_back(param);
}

// Returns a reference to an array of references to the various test parameters
// to be tested.
static test_param_t** GetParameters(test_param_t** params, int params_count,
                                    void** vanilla_funcs, int vanilla_count,
                                    void** hybrid_funcs, int hybrid_count,
                                    totem_cb_func_t* hybrid_alloc_funcs = NULL,
                                    totem_cb_func_t* hybrid_free_funcs = NULL) {
  // When this function is passed as a parameter to "ValuesIn" in the context of
  // INSTANTIATE_TEST_CASE_P macro, it gets invoked more than once within
  // the macro; therefore, the following hack is used to ensure that
  // initialization of the parameters array happens once.
  static bool initialized = false;
  if (initialized) { return params; }

  // This vector maintains the state of the different parameters during the
  // the tests, and hence it is defined static.
  static std::vector<test_param_t> params_vector;

  // Add the vanilla implementations.
  for (int i = 0; i < vanilla_count; i++) {
    PushParam(&params_vector, NULL, vanilla_funcs[i]);
  }

  // Add the hybrid implementations.
  for (int i = 0; i < hybrid_count; i++) {
    // Add the different configurations of the hybrid implementation.
    for (uint32_t j = 0; j < hybrid_configurations_count; j++) {
      PushParam(&params_vector, &totem_attrs[j], hybrid_funcs[i],
                hybrid_alloc_funcs ? hybrid_alloc_funcs[i] : NULL,
                hybrid_free_funcs ? hybrid_free_funcs[i] : NULL);
    }
  }

  // Fill the params array with references to the parameters to be tested
  // (maintained by params_vector throughout the execution of the tests).
  assert(params_count == params_vector.size());
  for (size_t i = 0; i != params_vector.size(); i++) {
    params[i] = &params_vector[i];
  }

  initialized = true;
  return params;
}

#endif  // TOTEM_COMMON_UNITTEST_H
