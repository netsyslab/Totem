/* 
 * Contains unit tests for an implementation of the breadth-first search (BFS)
 * graph search algorithm.
 *
 *  Created on: 2011-03-08
 *      Author: Lauro Beltrão Costa
 */

// totem includes
#include "totem_common_unittest.h"

#if GTEST_HAS_PARAM_TEST

using ::testing::TestWithParam;
using ::testing::Values;

// The following implementation relies on TestWithParam<BFSFunction> to test
// the two versions of BFS implemented: CPU and GPU.
// Details on how to use TestWithParam<T> can be found at:
// http://code.google.com/p/googletest/source/browse/trunk/samples/sample7_unittest.cc

typedef error_t(*BFSFunction)(graph_t*, vid_t, cost_t*);

// This is to allow testing the vanilla bfs functions and the hybrid one
// that is based on the framework. Note that have a different signature
// of the hybrid algorithm forced this work-around.
typedef struct bfs_param_s {
  bool          hybrid; // true when using the hybrid algorithm
  BFSFunction   func;   // the vanilla bfs function if hybrid flag is false
} bfs_param_t;

class BFSTest : public TestWithParam<bfs_param_t*> {
 public:
  virtual void SetUp() {
    // Ensure the minimum CUDA architecture is supported
    CUDA_CHECK_VERSION();
    bfs_param = GetParam();
    mem_type = TOTEM_MEM_HOST_PINNED;
  }

  error_t TestGraph(graph_t* graph, vid_t src, cost_t* cost) {
    if (bfs_param->hybrid) {
      totem_attr_t attr = TOTEM_DEFAULT_ATTR;
      attr.push_msg_size = 1;
      if (totem_init(graph, &attr) == FAILURE) {
        return FAILURE;
      }
      error_t err = bfs_hybrid(src, cost);
      totem_finalize();
      return err;
    }
    return bfs_param->func(graph, src, cost);
  }
 protected:
  bfs_param_t* bfs_param;
  totem_mem_t mem_type;
};

// Tests BFS for empty graphs.
TEST_P(BFSTest, Empty) {
  graph_t graph;
  graph.directed = false;
  graph.vertex_count = 0;
  graph.edge_count = 0;

  EXPECT_EQ(FAILURE, TestGraph(&graph, 0, NULL));
  EXPECT_EQ(FAILURE, TestGraph(&graph, 99, NULL));
}

// Tests BFS for single node graphs.
TEST_P(BFSTest, SingleNode) {
  graph_t* graph;
  graph_initialize(DATA_FOLDER("single_node.totem"), false, &graph);
  cost_t* cost = NULL;
  CALL_SAFE(totem_malloc(graph->vertex_count * sizeof(cost_t), mem_type, 
                         (void**)&cost));
  EXPECT_EQ(SUCCESS, TestGraph(graph, 0, cost));
  EXPECT_EQ((cost_t)0, cost[0]);
  EXPECT_EQ(FAILURE, TestGraph(graph, 1, cost));
  graph_finalize(graph);
  totem_free(cost, mem_type);

  graph_initialize(DATA_FOLDER("single_node_loop.totem"), false, &graph);
  CALL_SAFE(totem_malloc(graph->vertex_count * sizeof(cost_t), mem_type, 
                         (void**)&cost));
  EXPECT_EQ(SUCCESS, TestGraph(graph, 0, cost));
  EXPECT_EQ((cost_t)0, cost[0]);
  EXPECT_EQ(FAILURE, TestGraph(graph, 1, cost));
  totem_free(cost, mem_type);
  graph_finalize(graph);
}

// Tests BFS for graphs with node and no edges.
TEST_P(BFSTest, EmptyEdges) {
  graph_t* graph;
  graph_initialize(DATA_FOLDER("disconnected_1000_nodes.totem"), false, &graph);
  cost_t* cost = NULL;
  CALL_SAFE(totem_malloc(graph->vertex_count * sizeof(cost_t), mem_type, 
                         (void**)&cost));

  // First vertex as source
  vid_t source = 0;
  EXPECT_EQ(SUCCESS, TestGraph(graph, source, cost));
  EXPECT_EQ((cost_t)0, cost[source]);
  for(vid_t vertex = source + 1; vertex < graph->vertex_count; vertex++) {
    EXPECT_EQ(INF_COST, cost[vertex]);
  }

  // Last vertex as source
  source = graph->vertex_count - 1;
  EXPECT_EQ(SUCCESS, TestGraph(graph, source, cost));
  EXPECT_EQ((cost_t)0, cost[source]);
  for(vid_t vertex = source; vertex < graph->vertex_count - 1; vertex++){
    EXPECT_EQ(INF_COST, cost[vertex]);
  }

  // A vertex in the middle as source
  source = 199;
  EXPECT_EQ(SUCCESS, TestGraph(graph, source, cost));
  for(vid_t vertex = 0; vertex < graph->vertex_count; vertex++) {
    EXPECT_EQ((vertex == source) ? (cost_t)0 : INF_COST, cost[vertex]);
  }

  // Non existent vertex source
  EXPECT_EQ(FAILURE, TestGraph(graph, graph->vertex_count, cost));

  totem_free(cost, mem_type);
  graph_finalize(graph);
}

// Tests BFS for a chain of 1000 nodes.
TEST_P(BFSTest, Chain) {
  graph_t* graph;
  graph_initialize(DATA_FOLDER("chain_1000_nodes.totem"), false, &graph);
  cost_t* cost = NULL;
  CALL_SAFE(totem_malloc(graph->vertex_count * sizeof(cost_t), mem_type, 
                         (void**)&cost));

  // First vertex as source
  vid_t source = 0;
  EXPECT_EQ(SUCCESS, TestGraph(graph, source, cost));
  EXPECT_FALSE(NULL == cost);
  for(vid_t vertex = source; vertex < graph->vertex_count; vertex++){
    EXPECT_EQ(vertex, cost[vertex]);
  }

  // Last vertex as source
  source = graph->vertex_count - 1;
  EXPECT_EQ(SUCCESS, TestGraph(graph, source, cost));
  for(vid_t vertex = source; vertex < graph->vertex_count; vertex++){
    EXPECT_EQ(source - vertex, cost[vertex]);
  }

  // A vertex in the middle as source
  source = 199;
  EXPECT_EQ(SUCCESS, TestGraph(graph, source, cost));
  for(vid_t vertex = 0; vertex < graph->vertex_count; vertex++) {
    EXPECT_EQ((cost_t)abs((double)source - (double)vertex), cost[vertex]);
  }

  // Non existent vertex source
  EXPECT_EQ(FAILURE, TestGraph(graph, graph->vertex_count, cost));

  totem_free(cost, mem_type);
  graph_finalize(graph);
}

// Tests BFS for a complete graph of 300 nodes.
TEST_P(BFSTest, CompleteGraph) {
  graph_t* graph;
  graph_initialize(DATA_FOLDER("complete_graph_300_nodes.totem"), false,
                   &graph);
  cost_t* cost = NULL;
  CALL_SAFE(totem_malloc(graph->vertex_count * sizeof(cost_t), mem_type, 
                         (void**)&cost));

  // First vertex as source
  vid_t source = 0;
  EXPECT_EQ(SUCCESS, TestGraph(graph, source, cost));
  EXPECT_EQ((cost_t)0, cost[source]);
  for(vid_t vertex = source + 1; vertex < graph->vertex_count; vertex++){
    EXPECT_EQ((cost_t)1, cost[vertex]);
  }

  // Last vertex as source
  source = graph->vertex_count - 1;
  EXPECT_EQ(SUCCESS, TestGraph(graph, source, cost));
  EXPECT_EQ((cost_t)0, cost[source]);
  for(vid_t vertex = 0; vertex < source; vertex++) {
    EXPECT_EQ((cost_t)1, cost[vertex]);
  }

  // A vertex source in the middle
  source = 199;
  EXPECT_EQ(SUCCESS, TestGraph(graph, source, cost));
  for(vid_t vertex = 0; vertex < graph->vertex_count; vertex++) {
    EXPECT_EQ((cost_t)((source == vertex) ? 0 : 1), cost[vertex]);
  }

  // Non existent vertex source
  EXPECT_EQ(FAILURE, TestGraph(graph, graph->vertex_count, cost));

  totem_free(cost, mem_type);
  graph_finalize(graph);
}

// Tests BFS for a complete graph of 1000 nodes.
TEST_P(BFSTest, Star) {
  graph_t* graph;
  graph_initialize(DATA_FOLDER("star_1000_nodes.totem"), false, &graph);
  cost_t* cost = NULL;
  CALL_SAFE(totem_malloc(graph->vertex_count * sizeof(cost_t), mem_type, 
                         (void**)&cost));

  // First vertex as source
  vid_t source = 0;
  EXPECT_EQ(SUCCESS, TestGraph(graph, source, cost));
  EXPECT_EQ((cost_t)0, cost[source]);
  for(vid_t vertex = source + 1; vertex < graph->vertex_count; vertex++){
    EXPECT_EQ((cost_t)1, cost[vertex]);
  }

  // Last vertex as source
  source = graph->vertex_count - 1;
  EXPECT_EQ(SUCCESS, TestGraph(graph, source, cost));
  EXPECT_EQ((cost_t)0, cost[source]);
  EXPECT_EQ((cost_t)1, cost[0]);
  for(vid_t vertex = 1; vertex < source - 1; vertex++) {
    EXPECT_EQ((cost_t)2, cost[vertex]);
  }

  // A vertex source in the middle
  source = 199;
  EXPECT_EQ(SUCCESS, TestGraph(graph, source, cost));
  EXPECT_EQ((cost_t)1, cost[0]);
  for(vid_t vertex = 1; vertex < graph->vertex_count; vertex++) {
    EXPECT_EQ((cost_t)((source == vertex) ? 0 : 2), cost[vertex]);
  }

  // Non existent vertex source
  EXPECT_EQ(FAILURE, TestGraph(graph, graph->vertex_count, cost));

  totem_free(cost, mem_type);
  graph_finalize(graph);
}

// TODO(lauro): Add test cases for not well defined structures.

// Values() seems to accept only pointers, hence the possible parameters
// are defined here, and a pointer to each ot them is used.
bfs_param_t bfs_params[] = {{false, &bfs_cpu},
                            {false, &bfs_queue_cpu},
                            {false, &bfs_gpu},
                            {false, &bfs_vwarp_gpu},
                            {true, NULL}};

// From Google documentation:
// In order to run value-parameterized tests, we need to instantiate them,
// or bind them to a list of values which will be used as test parameters.
//
// Values() receives a list of parameters and the framework will execute the
// whole set of tests BFSTest for each element of Values()
INSTANTIATE_TEST_CASE_P(BFSGPUAndCPUTest, BFSTest, Values(&bfs_params[0],
                                                          &bfs_params[1],
                                                          &bfs_params[2],
                                                          &bfs_params[3],
                                                          &bfs_params[4]));

#else

// From Google documentation:
// Google Test may not support value-parameterized tests with some
// compilers. This dummy test keeps gtest_main linked in.
TEST_P(DummyTest, ValueParameterizedTestsAreNotSupportedOnThisPlatform) {}

#endif  // GTEST_HAS_PARAM_TEST