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
#include "totem_comdef.h"
#include "totem_graph.h"
#include "totem_mem.h"
#include "totem_util.h"

/**
 *  Defines a relative path of a graph file
 */
#define DATA_FOLDER(graph_file) "../../data/"graph_file

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

#endif // TOTEM_COMMON_UNITTEST_H
