/* TODO(lauro,abdullah,elizeu): Add license.
 *
 * Tests the node degree counting algorithms for the CPU and GPU.
 *
 *  Created on: 2012-04-30
 *      Author: Greg Redekop
 */

// totem includes
#include "totem_common_unittest.h"

#if GTEST_HAS_PARAM_TEST

using ::testing::TestWithParam;
using ::testing::Values;

// The following implementation relies on TestWithParam<NodeDegreeFunction> to
// test the two versions implemented: CPU and GPU.
// Details on how to use TestWithParam<T> can be found at:
// http://code.google.com/p/googletest/source/browse/trunk/samples/sample7_unittest.cc

typedef error_t(*NodeDegreeFunction)(const graph_t*, vid_t**);

class NodeDegreeTest : public TestWithParam<NodeDegreeFunction> {
 public:
  virtual void SetUp() {
    // Ensure the minimum CUDA architecture is supported
    CUDA_CHECK_VERSION();
    node_degree = GetParam();
  }

 protected:
  NodeDegreeFunction node_degree;
  graph_t* graph;
  vid_t* node_degree_count;
};


// Tests Node Degree for an empty vertex set graph.
TEST_P(NodeDegreeTest, EmptyVertexSet) {
  graph = (graph_t*)mem_alloc(sizeof(graph_t));
  graph->directed = false;
  graph->edge_count = 0;
  graph->vertex_count = 0;
  graph->weighted = false;
  graph->weights = NULL;

  EXPECT_EQ(FAILURE, node_degree(graph, &node_degree_count));
  mem_free(graph);
}

// Tests NodeDegree for a graph with an empty edge set.
TEST_P(NodeDegreeTest, EmptyEdgeSet) {
  graph = (graph_t*)mem_alloc(sizeof(graph_t));
  graph->directed = false;
  graph->edge_count = 0;
  graph->vertex_count = 123;
  graph->vertices = (eid_t*)mem_alloc(124 * sizeof(eid_t));
  memset(graph->vertices, (eid_t)0, 124 * sizeof(eid_t));
  graph->weighted = true;
  graph->weights = NULL;

  EXPECT_EQ(SUCCESS, node_degree(graph, &node_degree_count));
  EXPECT_FALSE(NULL == node_degree_count);
  for (vid_t vertex = 0; vertex < graph->vertex_count; vertex++) {
    EXPECT_EQ((vid_t)0, node_degree_count[vertex]);
  }
  mem_free(graph->vertices);
  mem_free(graph);
  mem_free(node_degree_count);
}

// Tests NodeDegree for single node graphs.
TEST_P(NodeDegreeTest, SingleNode) {
  graph_initialize(DATA_FOLDER("single_node.totem"), true, &graph);

  EXPECT_EQ(SUCCESS, node_degree(graph, &node_degree_count));
  EXPECT_FALSE(NULL == node_degree_count);
  EXPECT_EQ((vid_t)0, node_degree_count[0]);
  mem_free(node_degree_count);
  graph_finalize(graph);
}

// Tests NodeDegree algorithm in star graph with 1000 nodes.
TEST_P(NodeDegreeTest, Star) {
  graph_initialize(DATA_FOLDER("star_1000_nodes.totem"), true, &graph);

  EXPECT_EQ(SUCCESS, node_degree(graph, &node_degree_count));
  EXPECT_FALSE(NULL == node_degree_count);
  // Test all vertices
  EXPECT_EQ((vid_t)999, node_degree_count[0]);
  for (vid_t vert = 1; vert < graph->vertex_count; vert++) {
    EXPECT_EQ((weight_t)1, node_degree_count[vert]);
  }
  mem_free(node_degree_count);
  graph_finalize(graph);
}

// Tests NodeDegree algorithm a complete graph with 300 nodes.
TEST_P(NodeDegreeTest, Complete) {
  graph_initialize(DATA_FOLDER("complete_graph_300_nodes.totem"), true,
                   &graph);

  EXPECT_EQ(SUCCESS, node_degree(graph, &node_degree_count));
  EXPECT_FALSE(NULL == node_degree_count);
  // Test all vertices
  for (vid_t vert = 0; vert < graph->vertex_count; vert++) {
    EXPECT_EQ((vid_t)299, node_degree_count[vert]);
  }
  mem_free(node_degree_count);
  graph_finalize(graph);
}

INSTANTIATE_TEST_CASE_P(NodeDegreeGPUAndCPUTest, NodeDegreeTest,
                        Values(&node_degree_cpu, &node_degree_gpu));

#else

// From Google documentation:
// Google Test may not support value-parameterized tests with some
// compilers. This dummy test keeps gtest_main linked in.
TEST_P(DummyTest, ValueParameterizedTestsAreNotSupportedOnThisPlatform) {}

#endif  // GTEST_HAS_PARAM_TEST
