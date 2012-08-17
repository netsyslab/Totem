/* TODO(lauro,abdullah,elizeu): Add license.
 *
 * Contains unit tests for an implementation of the max-flow algorithm.
 *
 *  Created on: 2011-10-21
 *      Author: Greg Redekop
 */

// totem includes
#include "totem_common_unittest.h"

#if GTEST_HAS_PARAM_TEST

using ::testing::TestWithParam;
using ::testing::Values;

// The following implementation relies on TestWithParam<MaxFlowFunction> to test
// the two versions of Max Flow implemented: CPU and GPU.
// Details on how to use TestWithParam<T> can be found at:
// http://code.google.com/p/googletest/source/browse/trunk/samples/sample7_unittest.cc

typedef error_t(*MaxFlowFunction)(graph_t*, vid_t, vid_t, weight_t*);

class MaxFlowTest : public TestWithParam<MaxFlowFunction> {
 public:
  virtual void SetUp() {
    // Ensure the minimum CUDA architecture is supported
    CUDA_CHECK_VERSION();
    maxflow = GetParam();
  }

 protected:
   MaxFlowFunction maxflow;
   graph_t* graph;
   weight_t flow;
};

// Tests MaxFlow for empty graphs.
TEST_P(MaxFlowTest, Empty) {
  graph = (graph_t*) mem_alloc(sizeof(graph_t));
  graph->directed = false;
  graph->vertex_count = 0;
  graph->edge_count = 0;

  EXPECT_EQ(FAILURE, maxflow(graph, 0, 0, &flow));
  EXPECT_EQ(FAILURE, maxflow(graph, 99, 0, &flow));
  EXPECT_EQ(FAILURE, maxflow(graph, 0, 99, &flow));

  mem_free(graph);
}

// Tests MaxFlow for single node graphs.
TEST_P(MaxFlowTest, SingleNode) {
  graph_initialize(DATA_FOLDER("single_node.totem"), false, &graph);

  EXPECT_EQ(FAILURE, maxflow(graph, 0, 0, &flow));
  EXPECT_EQ(FAILURE, maxflow(graph, 1, 0, &flow));
  graph_finalize(graph);

  graph_initialize(DATA_FOLDER("single_node_loop.totem"), false, &graph);
  EXPECT_EQ(FAILURE, maxflow(graph, 0, 0, &flow));
  EXPECT_EQ(FAILURE, maxflow(graph, 1, 0, &flow));
  graph_finalize(graph);
}

// Tests MaxFlow through a trivial network.
TEST_P(MaxFlowTest, SourceSinkMaxflow) {
  graph_initialize(DATA_FOLDER("source_sink_maxflow.totem"), true, &graph);

  EXPECT_EQ(SUCCESS, maxflow(graph, 0, 6, &flow));
  EXPECT_EQ((weight_t)4, flow);
  graph_finalize(graph);
}

// Tests MaxFlow for a chain of 100 nodes.
TEST_P(MaxFlowTest, Chain) {
  graph_initialize(DATA_FOLDER("chain_100_nodes_weight_directed.totem"), true,
                   &graph);

  // First vertex as source
  vid_t source = 0;
  vid_t sink = graph->vertex_count - 1;
  EXPECT_EQ(SUCCESS, maxflow(graph, source, sink, &flow));
  EXPECT_EQ((weight_t)1, flow);

  // Last vertex as source
  source = graph->vertex_count - 1;
  EXPECT_EQ(FAILURE, maxflow(graph, source, sink, &flow));

  // A vertex in the middle as source
  source = 50;
  EXPECT_EQ(SUCCESS, maxflow(graph, source, sink, &flow));
  EXPECT_EQ((weight_t)1, flow);

  // Non existent vertex source
  EXPECT_EQ(FAILURE, maxflow(graph, graph->vertex_count, sink, &flow));

  graph_finalize(graph);
}

// Tests MaxFlow for a RMF network of 100 nodes.
TEST_P(MaxFlowTest, RMF100) {
  graph_initialize(DATA_FOLDER("rmf_100_nodes.totem"), true, &graph);

  // First vertex as source
  vid_t source = 0;
  vid_t sink = graph->vertex_count - 1;
  EXPECT_EQ(SUCCESS, maxflow(graph, source, sink, &flow));
  EXPECT_EQ((weight_t)174, flow);
  graph_finalize(graph);
}

// Tests MaxFlow for an acyclic dense network of 100 nodes.
TEST_P(MaxFlowTest, AcyclicDense100) {
  graph_initialize(DATA_FOLDER("acyclic_100_nodes.totem"), true, &graph);

  // First vertex as source
  vid_t source = 0;
  vid_t sink = graph->vertex_count - 1;
  EXPECT_EQ(SUCCESS, maxflow(graph, source, sink, &flow));
  EXPECT_EQ((weight_t)45333, flow);
  graph_finalize(graph);
}

// Tests MaxFlow for a Washing Randomly-generated network of 102 nodes.
TEST_P(MaxFlowTest, WashingtonRandom) {
  graph_initialize(DATA_FOLDER("washington_random.totem"), true, &graph);

  // First vertex as source
  vid_t source = 0;
  vid_t sink = graph->vertex_count - 1;
  EXPECT_EQ(SUCCESS, maxflow(graph, source, sink, &flow));
  EXPECT_EQ((weight_t)863, flow);
  graph_finalize(graph);
}

// From Google documentation:
// In order to run value-parameterized tests, we need to instantiate them,
// or bind them to a list of values which will be used as test parameters.
//
// Values() receives a list of parameters and the framework will execute the
// whole set of tests MaxFlowTest for each element of Values()
INSTANTIATE_TEST_CASE_P(MaxFlowGPUAndCPUTest, MaxFlowTest,
                        Values(&maxflow_cpu,
                               &maxflow_gpu,
                               &maxflow_vwarp_gpu));

#else

// From Google documentation:
// Google Test may not support value-parameterized tests with some
// compilers. This dummy test keeps gtest_main linked in.
TEST_P(DummyTest, ValueParameterizedTestsAreNotSupportedOnThisPlatform) {}

#endif  // GTEST_HAS_PARAM_TEST
