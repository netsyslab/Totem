/* TODO(lauro,abdullah,elizeu): Add license.
 *
 * Tests the all pairs shortest path implementation based on the Floyd-Warshall
 * algorithm and a parallel Dijkstra across all nodes for the CPU and GPU,
 * respectively.
 *
 *  Created on: 2011-11-04
 *      Author: Greg Redekop
 */

// totem includes
#include "totem_common_unittest.h"

#if GTEST_HAS_PARAM_TEST

using ::testing::TestWithParam;
using ::testing::Values;

// The following implementation relies on TestWithParam<APSPFunction> to
// test the two versions of All Pairs Shortest Path implemented: CPU and GPU.
// Details on how to use TestWithParam<T> can be found at:
// http://code.google.com/p/googletest/source/browse/trunk/samples/sample7_unittest.cc

typedef error_t(*APSPFunction)(graph_t*, weight_t**);

class APSPTest : public TestWithParam<APSPFunction> {
 public:
  virtual void SetUp() {
    // Ensure the minimum CUDA architecture is supported
    CUDA_CHECK_VERSION();
    apsp = GetParam();
  }

 protected:
  APSPFunction apsp;
  graph_t* graph;
};


// Tests Alll Pairs Shortest Path for an empty vertex set graph.
TEST_P(APSPTest, EmptyVertexSet) {
  graph = (graph_t*)mem_alloc(sizeof(graph_t));
  graph->directed = false;
  graph->edge_count = 0;
  graph->vertex_count = 0;
  graph->weighted = false;
  graph->weights = NULL;

  weight_t* distances;
  EXPECT_EQ(FAILURE, apsp(graph, &distances));
  EXPECT_EQ((weight_t*)NULL, distances);
  mem_free(graph);
}

// Tests APSP for a graph with an empty edge set.
TEST_P(APSPTest, EmptyEdgeSet) {
  graph = (graph_t*)mem_alloc(sizeof(graph_t));
  graph->directed = false;
  graph->edge_count = 0;
  graph->vertex_count = 123;
  graph->weighted = true;
  graph->weights = NULL;

  weight_t* distances;
  EXPECT_EQ(SUCCESS, apsp(graph, &distances));
  EXPECT_FALSE(NULL == distances);
  for (id_t src = 0; src < graph->vertex_count; src++) {
    weight_t* base = &distances[src * graph->vertex_count];
    for (id_t dest = 0; dest < graph->vertex_count; dest++) {
      if (src == dest)
        EXPECT_EQ((weight_t)0, base[dest]);
      else
        EXPECT_EQ((weight_t)WEIGHT_MAX, base[dest]);
    }
  }
  mem_free(graph);
  mem_free(distances);
}

// Tests APSP for single node graphs.
TEST_P(APSPTest, SingleNode) {
  graph_initialize(DATA_FOLDER("single_node.totem"), true, &graph);

  weight_t* distances;
  EXPECT_EQ(SUCCESS, apsp(graph, &distances));
  EXPECT_FALSE(NULL == distances);
  EXPECT_EQ(0, distances[0]);
  mem_free(distances);
  graph_finalize(graph);
}

// Tests APSP for a single node graph that contains a loop.
TEST_P(APSPTest, SingleNodeLoopWeighted) {
  graph_initialize(DATA_FOLDER("single_node_loop_weight.totem"), true, &graph);

  weight_t* distances;
  EXPECT_EQ(SUCCESS, apsp(graph, &distances));
  EXPECT_FALSE(NULL == distances);
  EXPECT_EQ((weight_t)0, distances[0]);
  mem_free(distances);
  graph_finalize(graph);
}

// Tests APSP algorithm in star graph with 1000 nodes.
TEST_P(APSPTest, Star) {
  graph_initialize(DATA_FOLDER("star_1000_nodes_weight.totem"), true, &graph);

  weight_t* distances;
  EXPECT_EQ(SUCCESS, apsp(graph, &distances));
  EXPECT_FALSE(NULL == distances);
  // Test all vertices
  for (id_t src = 0; src < graph->vertex_count; src++) {
    weight_t* base = &distances[src * graph->vertex_count];
    for (id_t dest = 0; dest < graph->vertex_count; dest++) {
      if (dest == src)
        EXPECT_EQ((weight_t)0, base[dest]);
      else if (dest == 0 || src == 0)
         EXPECT_EQ((weight_t)1, base[dest]);
      else
         EXPECT_EQ((weight_t)2, base[dest]);
    }
  }
  mem_free(distances);
  graph_finalize(graph);
}

// Tests APSP algorithm a complete graph with 300 nodes.
TEST_P(APSPTest, Complete) {
  graph_initialize(DATA_FOLDER("complete_graph_300_nodes_weight.totem"), true,
                   &graph);

  weight_t* distances;
  EXPECT_EQ(SUCCESS, apsp(graph, &distances));
  EXPECT_FALSE(NULL == distances);
  // Test all vertices
  for (id_t src = 0; src < graph->vertex_count; src++) {
    weight_t* base = &distances[src * graph->vertex_count];
    for (id_t dest = 0; dest < graph->vertex_count; dest++) {
      if (dest == src)
        EXPECT_EQ((weight_t)0, base[dest]);
      else
         EXPECT_EQ((weight_t)1, base[dest]);
    }
  }
  mem_free(distances);
  graph_finalize(graph);
}

// Tests APSP algorithm a star graph with 1K nodes with different edge weights.
TEST_P(APSPTest, StarDiffWeight) {
  graph_initialize(DATA_FOLDER("star_1000_nodes_diff_weight.totem"), true,
                   &graph);

  weight_t* distances;
  EXPECT_EQ(SUCCESS, apsp(graph, &distances));
  EXPECT_FALSE(NULL == distances);
  // Test all vertices
  for (id_t src = 0; src < graph->vertex_count; src++) {
    weight_t* base = &distances[src * graph->vertex_count];
    for (id_t dest = 0; dest < graph->vertex_count; dest++) {
      if (dest == src)
        EXPECT_EQ((weight_t)0, base[dest]);
      else if (dest == 0 || src == 0)
        EXPECT_EQ((weight_t)(src + 1), base[dest]);
      else
        EXPECT_EQ((weight_t)(src + 2), base[dest]);
    }
  }
  mem_free(distances);
  graph_finalize(graph);
}

// Tests APSP algorithm a complete graph with 300 nodes, different edge weights.
TEST_P(APSPTest, CompleteDiffWeight) {
  graph_initialize(DATA_FOLDER("complete_graph_300_nodes_diff_weight.totem"),
                   true, &graph);

  weight_t* distances;
  EXPECT_EQ(SUCCESS, apsp(graph, &distances));
  EXPECT_FALSE(NULL == distances);
  // Test all vertices
  for (id_t src = 0; src < graph->vertex_count; src++) {
    weight_t* base = &distances[src * graph->vertex_count];
    for (id_t dest = 0; dest < graph->vertex_count; dest++) {
      if (dest == src)
        EXPECT_EQ((weight_t)0, base[src]);
      else
        EXPECT_EQ((weight_t)(src + 1), base[dest]);
    }
  }
  mem_free(distances);
  graph_finalize(graph);
}

INSTANTIATE_TEST_CASE_P(APSPGPUAndCPUTest, APSPTest,
                        Values(&apsp_cpu,
                               &apsp_gpu));

#else

// From Google documentation:
// Google Test may not support value-parameterized tests with some
// compilers. This dummy test keeps gtest_main linked in.
TEST_P(DummyTest, ValueParameterizedTestsAreNotSupportedOnThisPlatform) {}

#endif  // GTEST_HAS_PARAM_TEST
