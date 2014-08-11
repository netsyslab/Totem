/*
 * Unit tests for label propagation algorithm.
 *
 * Created on: 2014-08-08
 * Authors: Tanuj Kr Aasawat
 *
 */

// totem includes
#include "totem_common_unittest.h"

#if GTEST_HAS_PARAM_TEST

using ::testing::TestWithParam;
using ::testing::Values;

// The following implementation relies on
// TestWithParam<LabelPropagationFunction> to test one version of
// Label Propagation implemented for: CPU.  Details on how to use
// TestWithParam<T> can be found at:
// http:
// code.google.com/p/googletest/source/browse/trunk/samples/sample7_unittest.cc

typedef error_t(*LabelPropagationFunction)(const graph_t*, vid_t*);

class LabelPropagationTest :
public TestWithParam<LabelPropagationFunction> {
 public:
  virtual void SetUp() {
    // Ensure the minimum CUDA architecture is supported
    CUDA_CHECK_VERSION();
    labelPropagation = GetParam();
    _graph = NULL;
    _labels = NULL;
    _mem_type = TOTEM_MEM_HOST_PINNED;
  }
  virtual void TearDown() {
    if (_graph) graph_finalize(_graph);
    if (_labels) totem_free(_labels, _mem_type);
  }

 protected:
  LabelPropagationFunction labelPropagation;
  graph_t* _graph;
  vid_t* _labels;
  totem_mem_t _mem_type;
};

// Tests LabelPropagation for an empty graph.
TEST_P(LabelPropagationTest, EmptyGraph) {
  graph_t graph;
  graph.directed = false;
  graph.vertex_count = 0;
  graph.edge_count = 0;
  EXPECT_EQ(FAILURE, labelPropagation(&graph, _labels));
}

// Tests LabelPropagation for a single node graph.
TEST_P(LabelPropagationTest, SingleNodeGrpah) {
  EXPECT_EQ(SUCCESS, graph_initialize(DATA_FOLDER("single_node.totem"),
                                      false, &_graph));
  CALL_SAFE(totem_malloc(_graph->vertex_count * sizeof(vid_t), _mem_type,
                         reinterpret_cast<void**>(&_labels)));
  EXPECT_EQ(SUCCESS, labelPropagation(_graph, _labels));
  EXPECT_FALSE(_labels == NULL);
  EXPECT_EQ(0, _labels[0]);
}

// Tests LabelPropagation for an undirected complete graph with
// 300 nodes.
TEST_P(LabelPropagationTest, CompleteGraph300NodesUndirected) {
  EXPECT_EQ(SUCCESS,
            graph_initialize(DATA_FOLDER("complete_graph_300_nodes.totem"),
                             false, &_graph));
  CALL_SAFE(totem_malloc(_graph->vertex_count * sizeof(vid_t), _mem_type,
                         reinterpret_cast<void**>(&_labels)));
  EXPECT_EQ(SUCCESS, labelPropagation(_graph, _labels));
  EXPECT_FALSE(_labels == NULL);
  for (vid_t vertex = 0; vertex < _graph->vertex_count; vertex++) {
    EXPECT_FLOAT_EQ(0, _labels[vertex]);
  }
}

// Tests LabelPropagation for an undirected grid graph with 15 nodes.
TEST_P(LabelPropagationTest, GridGraph15NodesUndirected) {
  EXPECT_EQ(SUCCESS,
            graph_initialize(DATA_FOLDER("grid_graph_15_nodes_weight.totem"),
                             false, &_graph));
  CALL_SAFE(totem_malloc(_graph->vertex_count * sizeof(vid_t), _mem_type,
                        reinterpret_cast<void**>(&_labels)));
  EXPECT_EQ(SUCCESS, labelPropagation(_graph, _labels));
  EXPECT_FALSE(_labels == NULL);
  for (vid_t vertex = 0; vertex < _graph->vertex_count; vertex++) {
    if (vertex == 0 || vertex == 1 || vertex == 5 || vertex == 6 ||
        vertex == 10 || vertex == 11)
      EXPECT_FLOAT_EQ(6, _labels[vertex]);
    else if (vertex == 2 || vertex == 7 || vertex == 12)
      EXPECT_FLOAT_EQ(7, _labels[vertex]);
    else if (vertex == 3 || vertex == 4 || vertex == 8 || vertex == 9 ||
        vertex == 13 || vertex == 14)
      EXPECT_FLOAT_EQ(8, _labels[vertex]);
  }
}

// Tests LabelPropagation for an undirected chain graph with 1000 nodes.
TEST_P(LabelPropagationTest, ChainGraph1000NodesUndirected) {
  EXPECT_EQ(SUCCESS,
            graph_initialize(DATA_FOLDER("chain_1000_nodes.totem"),
                             false, &_graph));
  CALL_SAFE(totem_malloc(_graph->vertex_count * sizeof(vid_t), _mem_type,
                         reinterpret_cast<void**>(&_labels)));
  EXPECT_EQ(SUCCESS, labelPropagation(_graph, _labels));
  EXPECT_FALSE(_labels == NULL);
  // Labels of all the vertices are expected to converge with label being 1.
  // For the chain graph, the number of iterations required for all the
  // vertices to reach convergence appears to be quite large. After 25
  // iterations only 6 vertices converged.
  for (vid_t vertex = 0; vertex < 6; vertex++) {
    EXPECT_FLOAT_EQ(1, _labels[vertex]);
  }
}

// Tests LabelPropagation for an undirected star graph with 1K nodes.
TEST_P(LabelPropagationTest, StarGraph1000NodesUndirected) {
  EXPECT_EQ(SUCCESS,
            graph_initialize(DATA_FOLDER("star_1000_nodes.totem"),
                             false, &_graph));
  CALL_SAFE(totem_malloc(_graph->vertex_count * sizeof(vid_t), _mem_type,
                         reinterpret_cast<void**>(&_labels)));
  EXPECT_EQ(SUCCESS, labelPropagation(_graph, _labels));
  EXPECT_FALSE(_labels == NULL);
  for (vid_t vertex = 0; vertex < _graph->vertex_count; vertex++) {
    EXPECT_FLOAT_EQ(0, _labels[vertex]);
  }
}

// Tests LabelPropagation for a graph with 1K disconnected nodes.
TEST_P(LabelPropagationTest, DisconnectedGraph1000Nodes) {
  EXPECT_EQ(SUCCESS,
            graph_initialize(DATA_FOLDER("disconnected_1000_nodes.totem"),
                             false, &_graph));
  CALL_SAFE(totem_malloc(_graph->vertex_count * sizeof(vid_t), _mem_type,
                         reinterpret_cast<void**>(&_labels)));
  EXPECT_EQ(SUCCESS, labelPropagation(_graph, _labels));
  EXPECT_FALSE(_labels == NULL);
  for (vid_t vertex = 0; vertex < _graph->vertex_count; vertex++) {
    EXPECT_FLOAT_EQ(vertex, _labels[vertex]);
  }
}

// Tests LabelPropagation for an undirected wheel graph with 1K
// nodes.
TEST_P(LabelPropagationTest, WheelGraph1000NodesUndirected) {
  EXPECT_EQ(SUCCESS,
            graph_initialize(DATA_FOLDER("wheel_graph_1000_nodes.totem"),
                             false, &_graph));
  CALL_SAFE(totem_malloc(_graph->vertex_count * sizeof(vid_t), _mem_type,
                         reinterpret_cast<void**>(&_labels)));
  EXPECT_EQ(SUCCESS, labelPropagation(_graph, _labels));
  EXPECT_FALSE(_labels == NULL);
  for (vid_t vertex = 0; vertex < _graph->vertex_count; vertex++) {
    EXPECT_FLOAT_EQ(0, _labels[vertex]);
  }
}

// From Google documentation:
// In order to run value-parameterized tests, we need to instantiate them,
// or bind them to a list of values which will be used as test parameters.
//
// Values() receives a list of parameters and the framework will execute the
// whole set of tests LabelPropagationTest for each element of Values()
INSTANTIATE_TEST_CASE_P(LabelPropagationCPUTest, LabelPropagationTest,
                        Values(&label_propagation_cpu));

#else

// From Google documentation:
// Google Test may not support value-parameterized tests with some
// compilers. This dummy test keeps gtest_main linked in.
TEST_P(DummyTest, ValueParameterizedTestsAreNotSupportedOnThisPlatform) {}

#endif  // GTEST_HAS_PARAM_TEST
