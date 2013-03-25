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
    _mem_type = TOTEM_MEM_HOST_PINNED;
    _flow = 0;
  }

  virtual void TearDown() {
    if (_graph) graph_finalize(_graph);
  }

 protected:
   MaxFlowFunction maxflow;
   graph_t* _graph;
   weight_t _flow;
   totem_mem_t _mem_type;
};

// Tests MaxFlow for empty graphs.
TEST_P(MaxFlowTest, Empty) {
  _graph = (graph_t*) malloc(sizeof(graph_t));
  _graph->directed = false;
  _graph->vertex_count = 0;
  _graph->edge_count = 0;

  EXPECT_EQ(FAILURE, maxflow(_graph, 0, 0, &_flow));
  EXPECT_EQ(FAILURE, maxflow(_graph, 99, 0, &_flow));
  EXPECT_EQ(FAILURE, maxflow(_graph, 0, 99, &_flow));

  free(_graph);
  _graph = NULL;
}

// Tests MaxFlow for single node graphs.
TEST_P(MaxFlowTest, SingleNode) {
  graph_initialize(DATA_FOLDER("single_node.totem"), false, &_graph);
  EXPECT_EQ(FAILURE, maxflow(_graph, 0, 0, &_flow));
  EXPECT_EQ(FAILURE, maxflow(_graph, 1, 0, &_flow));
}
 
TEST_P(MaxFlowTest, SingleNodeLoop) {
  graph_initialize(DATA_FOLDER("single_node_loop.totem"), false, &_graph);
  EXPECT_EQ(FAILURE, maxflow(_graph, 0, 0, &_flow));
  EXPECT_EQ(FAILURE, maxflow(_graph, 1, 0, &_flow));
}

// Tests MaxFlow through a trivial network.
TEST_P(MaxFlowTest, SourceSinkMaxflow) {
  graph_initialize(DATA_FOLDER("source_sink_maxflow.totem"), true, &_graph);
  EXPECT_EQ(SUCCESS, maxflow(_graph, 0, 6, &_flow));
  EXPECT_EQ((weight_t)4, _flow);
}

// Tests MaxFlow for a chain of 100 nodes.
TEST_P(MaxFlowTest, Chain) {
  graph_initialize(DATA_FOLDER("chain_100_nodes_weight_directed.totem"), true,
                   &_graph);

  // First vertex as source
  vid_t source = 0;
  vid_t sink = _graph->vertex_count - 1;
  EXPECT_EQ(SUCCESS, maxflow(_graph, source, sink, &_flow));
  EXPECT_EQ((weight_t)1, _flow);

  // Last vertex as source
  source = _graph->vertex_count - 1;
  EXPECT_EQ(FAILURE, maxflow(_graph, source, sink, &_flow));

  // A vertex in the middle as source
  source = 50;
  EXPECT_EQ(SUCCESS, maxflow(_graph, source, sink, &_flow));
  EXPECT_EQ((weight_t)1, _flow);

  // Non existent vertex source
  EXPECT_EQ(FAILURE, maxflow(_graph, _graph->vertex_count, sink, &_flow));
}

// Tests MaxFlow for a RMF network of 100 nodes.
TEST_P(MaxFlowTest, RMF100) {
  graph_initialize(DATA_FOLDER("rmf_100_nodes.totem"), true, &_graph);

  // First vertex as source
  vid_t source = 0;
  vid_t sink = _graph->vertex_count - 1;
  EXPECT_EQ(SUCCESS, maxflow(_graph, source, sink, &_flow));
  EXPECT_EQ((weight_t)174, _flow);
}

// Tests MaxFlow for an acyclic dense network of 100 nodes.
TEST_P(MaxFlowTest, AcyclicDense100) {
  graph_initialize(DATA_FOLDER("acyclic_100_nodes.totem"), true, &_graph);

  // First vertex as source
  vid_t source = 0;
  vid_t sink = _graph->vertex_count - 1;
  EXPECT_EQ(SUCCESS, maxflow(_graph, source, sink, &_flow));
  EXPECT_EQ((weight_t)45333, _flow);
}

// Tests MaxFlow for a Washing Randomly-generated network of 102 nodes.
TEST_P(MaxFlowTest, WashingtonRandom) {
  graph_initialize(DATA_FOLDER("washington_random.totem"), true, &_graph);

  // First vertex as source
  vid_t source = 0;
  vid_t sink = _graph->vertex_count - 1;
  EXPECT_EQ(SUCCESS, maxflow(_graph, source, sink, &_flow));
  EXPECT_EQ((weight_t)863, _flow);
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
