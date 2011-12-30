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
#include "totem_partition.h"
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

#endif // TOTEM_COMMON_UNITTEST_H
