/* TODO(lauro,abdullah,elizeu): Add license.
 *
 * Tests the single source shortest path implementation based on the Dijkstra
 * algorithm.
 *
 *  Created on: 2011-03-24
 *      Author: Elizeu Santos-Neto
 */

// totem includes
#include "totem_common_unittest.h"

#if GTEST_HAS_PARAM_TEST

using ::testing::TestWithParam;
using ::testing::Values;

// The following implementation relies on TestWithParam<DijkstraFunction> to
// test the two versions of PageRank implemented: CPU and GPU.
// Details on how to use TestWithParam<T> can be found at:
// totem_bfs_unittest.cc and
// http://code.google.com/p/googletest/source/browse/trunk/samples/sample7_unittest.cc

typedef error_t(*DijkstraFunction)(const graph_t*, id_t, weight_t**);

class DijkstraTest : public TestWithParam<DijkstraFunction> {
 public:
  virtual void SetUp() {
    // Ensure the minimum CUDA architecture is supported
    CUDA_CHECK_VERSION();
    dijkstra = GetParam();
  }

 protected:
  DijkstraFunction dijkstra;
};


// Tests Dijkstra for empty vertex set graph.
TEST_P(DijkstraTest, EmptyVertexSet) {
  graph_t graph;
  graph.directed = false;
  graph.edge_count = 0;
  graph.vertex_count = 0;
  graph.weighted = false;
  graph.weights = NULL;

  weight_t* short_distances;
  EXPECT_EQ(FAILURE, dijkstra(&graph, 0, &short_distances));
  EXPECT_EQ(NULL, short_distances);
  EXPECT_EQ(FAILURE, dijkstra(&graph, 666, &short_distances));
  EXPECT_EQ(NULL, short_distances);
}

// Tests Dijkstra for a graph with an empty edge set.
TEST_P(DijkstraTest, EmptyEdgeSet) {
  graph_t graph;
  graph.directed = false;
  graph.edge_count = 0;
  graph.vertex_count = 123;
  graph.weighted = true;
  graph.weights = NULL;

  weight_t* short_distances;
  EXPECT_EQ(SUCCESS, dijkstra(&graph, 0, &short_distances));
  EXPECT_FALSE(NULL == short_distances);
  EXPECT_EQ((weight_t)0, short_distances[0]);
  for (id_t vertex_id = 1; vertex_id < graph.vertex_count; vertex_id++) {
    EXPECT_EQ(WEIGHT_MAX, short_distances[vertex_id]);
  }
  mem_free(short_distances);
}

// Tests Dijkstra for single node graphs.
TEST_P(DijkstraTest, SingleNode) {
  graph_t* graph;
  graph_initialize(DATA_FOLDER("single_node.totem"), true, &graph);

  weight_t* short_distances;
  EXPECT_EQ(SUCCESS, dijkstra(graph, 0, &short_distances));
  EXPECT_FALSE(NULL == short_distances);
  EXPECT_EQ(0, short_distances[0]);
  mem_free(short_distances);
  graph_finalize(graph);
}

// Tests Dijkstra implementation for a single node graph that contains a loop.
TEST_P(DijkstraTest, SingleNodeLoopWeighted) {
  graph_t* graph;
  graph_initialize(DATA_FOLDER("single_node_loop_weight.totem"), true, &graph);

  weight_t* short_distances;
  EXPECT_EQ(SUCCESS, dijkstra(graph, 0, &short_distances));
  EXPECT_FALSE(NULL == short_distances);
  EXPECT_EQ((weight_t)0, short_distances[0]);
  mem_free(short_distances);

  EXPECT_EQ(FAILURE, dijkstra(graph, 100, &short_distances));
  EXPECT_EQ(NULL, short_distances);
  graph_finalize(graph);
}

// Tests SSSP algorithm for a chain of 1000 nodes.
TEST_P(DijkstraTest, Chain) {
  graph_t* graph;
  graph_initialize(DATA_FOLDER("chain_1000_nodes_weight.totem"), true, &graph);

  // First vertex as source
  id_t source = 0;
  weight_t* short_distances;
  EXPECT_EQ(SUCCESS, dijkstra(graph, source, &short_distances));
  EXPECT_FALSE(NULL == short_distances);
  for(id_t vertex_id = source; vertex_id < graph->vertex_count; vertex_id++){
    EXPECT_EQ(vertex_id, short_distances[vertex_id]);
  }
  mem_free(short_distances);

  // Last vertex as source
  source = graph->vertex_count - 1;
  EXPECT_EQ(SUCCESS, dijkstra(graph, source, &short_distances));
  for(id_t vertex_id = 0; vertex_id < graph->vertex_count; vertex_id++){
    EXPECT_EQ(source - vertex_id, short_distances[vertex_id]);
  }
  mem_free(short_distances);

  // A vertex in the middle as source
  source = 199;
  EXPECT_EQ(SUCCESS, dijkstra(graph, source, &short_distances));
  for(id_t vertex_id = 0; vertex_id < graph->vertex_count; vertex_id++) {
    EXPECT_EQ((uint32_t)abs(source - vertex_id), short_distances[vertex_id]);
  }
  mem_free(short_distances);

  // Non existent vertex source
  EXPECT_EQ(FAILURE,
    dijkstra(graph, graph->vertex_count, &short_distances));
  EXPECT_EQ(NULL, short_distances);
  graph_finalize(graph);
}

// Tests SSSP algorithm in star graph with 1000 nodes.
TEST_P(DijkstraTest, Star) {
  graph_t* graph;
  graph_initialize(DATA_FOLDER("star_1000_nodes_weight.totem"), true, &graph);

  // First vertex as source
  id_t source = 0;
  weight_t* short_distances;
  EXPECT_EQ(SUCCESS, dijkstra(graph, source, &short_distances));
  EXPECT_FALSE(NULL == short_distances);
  EXPECT_EQ(0, short_distances[0]);
  for(id_t vertex_id = 1; vertex_id < graph->vertex_count; vertex_id++){
    EXPECT_EQ(1, short_distances[vertex_id]);
  }
  mem_free(short_distances);

  // Last vertex as source
  source = graph->vertex_count - 1;
  EXPECT_EQ(SUCCESS, dijkstra(graph, source, &short_distances));
  EXPECT_EQ(0, short_distances[source]);
  EXPECT_EQ(1, short_distances[0]);
  for(id_t vertex_id = 1; vertex_id < graph->vertex_count - 1; vertex_id++){
    EXPECT_EQ(2, short_distances[vertex_id]);
  }
  mem_free(short_distances);

  // Non existent vertex source
  EXPECT_EQ(FAILURE,
    dijkstra(graph, graph->vertex_count, &short_distances));
  EXPECT_EQ(NULL, short_distances);
  graph_finalize(graph);
}

// Tests SSSP algorithm a complete graph with 300 nodes.
TEST_P(DijkstraTest, Complete) {
  graph_t* graph;
  graph_initialize(DATA_FOLDER("complete_graph_300_nodes_weight.totem"), true, &graph);

  // First vertex as source
  id_t source = 0;
  weight_t* short_distances;
  EXPECT_EQ(SUCCESS, dijkstra(graph, source, &short_distances));
  EXPECT_FALSE(NULL == short_distances);
  EXPECT_EQ(0, short_distances[0]);
  for(id_t vertex_id = 1; vertex_id < graph->vertex_count; vertex_id++){
    EXPECT_EQ(1, short_distances[vertex_id]);
  }
  mem_free(short_distances);

  // Last vertex as source
  source = graph->vertex_count - 1;
  EXPECT_EQ(SUCCESS, dijkstra(graph, source, &short_distances));
  EXPECT_EQ(0, short_distances[source]);
  for(id_t vertex_id = 1; vertex_id < graph->vertex_count - 1; vertex_id++){
    EXPECT_EQ(1, short_distances[vertex_id]);
  }
  mem_free(short_distances);

  // Non existent vertex source
  EXPECT_EQ(FAILURE,
    dijkstra(graph, graph->vertex_count, &short_distances));
  EXPECT_EQ(NULL, short_distances);
  graph_finalize(graph);
}

// Tests SSSP algorithm a star graph with 1K nodes with different edge weights.
TEST_P(DijkstraTest, StarDiffWeight) {
  graph_t* graph;
  graph_initialize(DATA_FOLDER("star_1000_nodes_diff_weight.totem"),
    true, &graph);

  // First vertex as source
  id_t source = 0;
  weight_t* short_distances;
  EXPECT_EQ(SUCCESS, dijkstra(graph, source, &short_distances));
  EXPECT_FALSE(NULL == short_distances);
  EXPECT_EQ(0, short_distances[0]);
  for(id_t vertex_id = 1; vertex_id < graph->vertex_count; vertex_id++){
    EXPECT_EQ(1, short_distances[vertex_id]);
  }
  mem_free(short_distances);

  // Last vertex as source
  source = graph->vertex_count - 1;
  EXPECT_EQ(SUCCESS, dijkstra(graph, source, &short_distances));
  EXPECT_EQ(0, short_distances[source]);
  EXPECT_EQ(source + 1, short_distances[0]);
  for(id_t vertex_id = 1; vertex_id < graph->vertex_count - 1; vertex_id++){
    // out edge weight = vertex_id + 1
    EXPECT_EQ(source + 2, short_distances[vertex_id]);
  }
  mem_free(short_distances);

  // Non existent vertex source
  EXPECT_EQ(FAILURE,
    dijkstra(graph, graph->vertex_count, &short_distances));
  EXPECT_EQ(NULL, short_distances);
  graph_finalize(graph);
}

// Tests SSSP algorithm a complete graph with 300 nodes, different edge weights.
TEST_P(DijkstraTest, CompleteDiffWeight) {
  graph_t* graph;
  graph_initialize(DATA_FOLDER("complete_graph_300_nodes_diff_weight.totem"),
    true, &graph);

  // First vertex as source
  id_t source = 0;
  weight_t* short_distances;
  EXPECT_EQ(SUCCESS, dijkstra(graph, source, &short_distances));
  EXPECT_FALSE(NULL == short_distances);
  EXPECT_EQ(0, short_distances[0]);
  for(id_t vertex_id = 1; vertex_id < graph->vertex_count; vertex_id++){
    EXPECT_EQ(1, short_distances[vertex_id]);
  }
  mem_free(short_distances);

  // Last vertex as source
  source = graph->vertex_count - 1;
  EXPECT_EQ(SUCCESS, dijkstra(graph, source, &short_distances));
  EXPECT_EQ(0, short_distances[source]);
  for(id_t vertex_id = 0; vertex_id < graph->vertex_count - 1; vertex_id++){
    // out edge from any node has weight = vertex_id + 1
    EXPECT_EQ(source + 1, short_distances[vertex_id]);
  }
  mem_free(short_distances);

  // Non existent vertex source
  EXPECT_EQ(FAILURE,
    dijkstra(graph, graph->vertex_count, &short_distances));
  EXPECT_EQ(NULL, short_distances);
  graph_finalize(graph);
}

// TODO(elizeu): Add irregular topology graphs.

INSTANTIATE_TEST_CASE_P(DijkstraGPUAndCPUTest, DijkstraTest,
                        Values(&dijkstra_gpu, &dijkstra_cpu,
                               &dijkstra_vwarp_gpu));

#else

// From Google documentation:
// Google Test may not support value-parameterized tests with some
// compilers. This dummy test keeps gtest_main linked in.
TEST_P(DummyTest, ValueParameterizedTestsAreNotSupportedOnThisPlatform) {}

#endif  // GTEST_HAS_PARAM_TEST
