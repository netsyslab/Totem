/* 
 * Contains unit tests for an implementation of the p-core decompsition
 * algorithm.
 *
 *  Created on: 2011-06-03
 *      Author: Abdullah Gharaibeh
 */

// totem includes
#include "totem_common_unittest.h"

#if GTEST_HAS_PARAM_TEST

using ::testing::TestWithParam;
using ::testing::Values;

// The following implementation relies on TestWithParam<PCoreFunction> to test
// the two versions of p-core: CPU and GPU.
// Details on how to use TestWithParam<T> can be found at:
// http://code.google.com/p/googletest/source/browse/trunk/samples/sample7_unittest.cc

typedef error_t(*PCoreFunction)(const graph_t*, uint32_t, uint32_t, uint32_t**);

class PCoreTest : public TestWithParam<PCoreFunction> {
 public:
  virtual void SetUp() {
    // Ensure the minimum CUDA architecture is supported
    CUDA_CHECK_VERSION();
    pcore = GetParam();
    _graph = NULL;
    _round = NULL;
  }

  virtual void TearDown() {
    if (_graph) graph_finalize(_graph);
    if(_round) totem_free(_round, TOTEM_MEM_HOST_PINNED);
  }

 protected:
   PCoreFunction pcore;
   graph_t* _graph;
   uint32_t* _round;
};

// Tests p-core for empty graphs.
TEST_P(PCoreTest, Empty) {
  graph_t graph;
  graph.directed = true;
  graph.weighted = true;
  graph.vertex_count = 0;
  graph.edge_count = 0;
  EXPECT_EQ(FAILURE, pcore(&graph, 0, 1, &_round));
  EXPECT_EQ((uint32_t*)NULL, _round);
}

// Tests p-core for single node graphs.
TEST_P(PCoreTest, SingleNode) {
  graph_initialize(DATA_FOLDER("single_node.totem"), true, &_graph);
  _graph->directed = false;
  EXPECT_EQ(SUCCESS, pcore(_graph, 0, 1, &_round));
  EXPECT_FALSE(_round == NULL);
  EXPECT_EQ((uint32_t)0, _round[0]);
}

TEST_P(PCoreTest, SingleNodeLoop) {
  graph_initialize(DATA_FOLDER("single_node_loop.totem"), true, &_graph);
  _graph->directed = false;
  EXPECT_EQ(SUCCESS, pcore(_graph, 0, 1, &_round));
  EXPECT_FALSE(_round == NULL);
  EXPECT_EQ((uint32_t)1, _round[0]);
}

// Tests p-core for graphs with node and no edges.
TEST_P(PCoreTest, EmptyEdges) {
  graph_initialize(DATA_FOLDER("disconnected_1000_nodes.totem"), true, &_graph);
  EXPECT_EQ(SUCCESS, pcore(_graph, 0, 1, &_round));
  EXPECT_FALSE(_round == NULL);
  for (vid_t vertex = 0; vertex < _graph->vertex_count; vertex++) {
    EXPECT_EQ((uint32_t)0, _round[vertex]);
  }
}

// Tests p-core for a chain of 1000 nodes.
TEST_P(PCoreTest, Chain) {
  graph_initialize(DATA_FOLDER("chain_1000_nodes.totem"), true, &_graph);
  EXPECT_EQ(SUCCESS, pcore(_graph, 0, 1, &_round));
  EXPECT_FALSE(_round == NULL);
  for (vid_t vertex = 0; vertex < _graph->vertex_count; vertex++) {
    EXPECT_EQ((uint32_t)1, _round[vertex]);
  }
}

// Tests p-core for a complete graph of 300 nodes.
TEST_P(PCoreTest, CompleteGraph) {
  graph_initialize(DATA_FOLDER("complete_graph_300_nodes.totem"), true, &_graph);
  EXPECT_EQ(SUCCESS, pcore(_graph, 0, 1, &_round));
  EXPECT_FALSE(_round == NULL);
  for (vid_t vertex = 0; vertex < _graph->vertex_count; vertex++) {
    EXPECT_EQ((uint32_t)299, _round[vertex]);
  }
}

// Tests BFS for a star graph of 1000 nodes.
TEST_P(PCoreTest, Star) {
  graph_initialize(DATA_FOLDER("star_1000_nodes.totem"), true, &_graph);
  EXPECT_EQ(SUCCESS, pcore(_graph, 0, 1, &_round));
  EXPECT_FALSE(_round == NULL);
  for (vid_t vertex = 0; vertex < _graph->vertex_count; vertex++) {
    EXPECT_EQ((uint32_t)1, _round[vertex]);
  }
}

// TODO(abdullah): Add test cases for not well defined structures.

// From Google documentation:
// In order to run value-parameterized tests, we need to instantiate them,
// or bind them to a list of values which will be used as test parameters.
//
// Values() receives a list of parameters and the framework will execute the
// whole set of tests PCoreTest for each element of Values()
INSTANTIATE_TEST_CASE_P(PCOREGPUAndCPUTest, PCoreTest,
                        Values(&pcore_cpu,
                               &pcore_gpu));

#else

// From Google documentation:
// Google Test may not support value-parameterized tests with some
// compilers. This dummy test keeps gtest_main linked in.
TEST_P(DummyTest, ValueParameterizedTestsAreNotSupportedOnThisPlatform) {}

#endif  // GTEST_HAS_PARAM_TEST
