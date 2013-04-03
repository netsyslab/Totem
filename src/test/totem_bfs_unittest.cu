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
  totem_attr_t* attr;   // totem attributes for totem-based tests
  BFSFunction   func;   // the vanilla bfs function if attr is NULL
} bfs_param_t;

class BFSTest : public TestWithParam<bfs_param_t*> {
 public:
  virtual void SetUp() {
    // Ensure the minimum CUDA architecture is supported
    CUDA_CHECK_VERSION();
    _bfs_param = GetParam();
    _mem_type = TOTEM_MEM_HOST_PINNED;
    _graph = NULL;
    _cost = NULL;
  }
  virtual void TearDown() {
    if (_graph) graph_finalize(_graph);
  }

  error_t TestGraph(vid_t src) {
    if (_bfs_param->attr) {
      _bfs_param->attr->push_msg_size = 1;
      if (totem_init(_graph, _bfs_param->attr) == FAILURE) {
        return FAILURE;
      }
      error_t err = bfs_hybrid(src, _cost);
      totem_finalize();
      return err;
    }
    return _bfs_param->func(_graph, src, _cost);
  }
 protected:
  bfs_param_t* _bfs_param;
  totem_mem_t _mem_type;
  graph_t* _graph;
  cost_t* _cost;
};

// Tests BFS for empty graphs.
TEST_P(BFSTest, Empty) {
  _graph = (graph_t*)calloc(1, sizeof(graph_t));
  EXPECT_EQ(FAILURE, TestGraph(0));
  EXPECT_EQ(FAILURE, TestGraph(99));
  free(_graph);
  _graph = NULL;
}

// Tests BFS for single node graphs.
TEST_P(BFSTest, SingleNode) {
  graph_initialize(DATA_FOLDER("single_node.totem"), false, &_graph);
  CALL_SAFE(totem_malloc(_graph->vertex_count * sizeof(cost_t), _mem_type, 
                         (void**)&_cost));
  EXPECT_EQ(SUCCESS, TestGraph(0));
  EXPECT_EQ((cost_t)0, _cost[0]);
  EXPECT_EQ(FAILURE, TestGraph(1));
}

TEST_P(BFSTest, SingleNodeLoop) {
  graph_initialize(DATA_FOLDER("single_node_loop.totem"), false, &_graph);
  CALL_SAFE(totem_malloc(_graph->vertex_count * sizeof(cost_t), _mem_type, 
                         (void**)&_cost));
  EXPECT_EQ(SUCCESS, TestGraph(0));
  EXPECT_EQ((cost_t)0, _cost[0]);
  EXPECT_EQ(FAILURE, TestGraph(1));
}

// Tests BFS for graphs with node and no edges.
TEST_P(BFSTest, EmptyEdges) {
  graph_initialize(DATA_FOLDER("disconnected_1000_nodes.totem"), false,
                   &_graph);
  CALL_SAFE(totem_malloc(_graph->vertex_count * sizeof(cost_t), _mem_type, 
                         (void**)&_cost));

  // First vertex as source
  vid_t source = 0;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  EXPECT_EQ((cost_t)0, _cost[source]);
  for(vid_t vertex = source + 1; vertex < _graph->vertex_count; vertex++) {
    EXPECT_EQ(INF_COST, _cost[vertex]);
  }

  // Last vertex as source
  source = _graph->vertex_count - 1;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  EXPECT_EQ((cost_t)0, _cost[source]);
  for(vid_t vertex = source; vertex < _graph->vertex_count - 1; vertex++){
    EXPECT_EQ(INF_COST, _cost[vertex]);
  }

  // A vertex in the middle as source
  source = 199;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  for(vid_t vertex = 0; vertex < _graph->vertex_count; vertex++) {
    EXPECT_EQ((vertex == source) ? (cost_t)0 : INF_COST, _cost[vertex]);
  }

  // Non existent vertex source
  EXPECT_EQ(FAILURE, TestGraph(_graph->vertex_count));
}

// Tests BFS for a chain of 1000 nodes.
TEST_P(BFSTest, Chain) {
  graph_initialize(DATA_FOLDER("chain_1000_nodes.totem"), false, &_graph);
  CALL_SAFE(totem_malloc(_graph->vertex_count * sizeof(cost_t), _mem_type, 
                         (void**)&_cost));

  // First vertex as source
  vid_t source = 0;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  for(vid_t vertex = source; vertex < _graph->vertex_count; vertex++){
    EXPECT_EQ(vertex, _cost[vertex]);
  }

  // Last vertex as source
  source = _graph->vertex_count - 1;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  for(vid_t vertex = source; vertex < _graph->vertex_count; vertex++){
    EXPECT_EQ(source - vertex, _cost[vertex]);
  }

  // A vertex in the middle as source
  source = 199;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  for(vid_t vertex = 0; vertex < _graph->vertex_count; vertex++) {
    EXPECT_EQ((cost_t)abs((double)source - (double)vertex), _cost[vertex]);
  }

  // Non existent vertex source
  EXPECT_EQ(FAILURE, TestGraph(_graph->vertex_count));
}

// Tests BFS for a complete graph of 300 nodes.
TEST_P(BFSTest, CompleteGraph) {
  graph_initialize(DATA_FOLDER("complete_graph_300_nodes.totem"), false,
                   &_graph);
  CALL_SAFE(totem_malloc(_graph->vertex_count * sizeof(cost_t), _mem_type, 
                         (void**)&_cost));

  // First vertex as source
  vid_t source = 0;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  EXPECT_EQ((cost_t)0, _cost[source]);
  for(vid_t vertex = source + 1; vertex < _graph->vertex_count; vertex++){
    EXPECT_EQ((cost_t)1, _cost[vertex]);
  }

  // Last vertex as source
  source = _graph->vertex_count - 1;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  EXPECT_EQ((cost_t)0, _cost[source]);
  for(vid_t vertex = 0; vertex < source; vertex++) {
    EXPECT_EQ((cost_t)1, _cost[vertex]);
  }

  // A vertex source in the middle
  source = 199;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  for(vid_t vertex = 0; vertex < _graph->vertex_count; vertex++) {
    EXPECT_EQ((cost_t)((source == vertex) ? 0 : 1), _cost[vertex]);
  }

  // Non existent vertex source
  EXPECT_EQ(FAILURE, TestGraph(_graph->vertex_count));
}

// Tests BFS for a complete graph of 1000 nodes.
TEST_P(BFSTest, Star) {
  graph_initialize(DATA_FOLDER("star_1000_nodes.totem"), false, &_graph);
  CALL_SAFE(totem_malloc(_graph->vertex_count * sizeof(cost_t), _mem_type, 
                         (void**)&_cost));

  // First vertex as source
  vid_t source = 0;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  EXPECT_EQ((cost_t)0, _cost[source]);
  for(vid_t vertex = source + 1; vertex < _graph->vertex_count; vertex++){
    EXPECT_EQ((cost_t)1, _cost[vertex]);
  }

  // Last vertex as source
  source = _graph->vertex_count - 1;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  EXPECT_EQ((cost_t)0, _cost[source]);
  EXPECT_EQ((cost_t)1, _cost[0]);
  for(vid_t vertex = 1; vertex < source - 1; vertex++) {
    EXPECT_EQ((cost_t)2, _cost[vertex]);
  }

  // A vertex source in the middle
  source = 199;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  EXPECT_EQ((cost_t)1, _cost[0]);
  for(vid_t vertex = 1; vertex < _graph->vertex_count; vertex++) {
    EXPECT_EQ((cost_t)((source == vertex) ? 0 : 2), _cost[vertex]);
  }

  // Non existent vertex source
  EXPECT_EQ(FAILURE, TestGraph(_graph->vertex_count));
}

// TODO(lauro): Add test cases for not well defined structures.

// Values() seems to accept only pointers, hence the possible parameters
// are defined here, and a pointer to each ot them is used.
bfs_param_t bfs_params[] = {
  {NULL, &bfs_cpu},
  {NULL, &bfs_queue_cpu},
  {NULL, &bfs_gpu},
  {NULL, &bfs_vwarp_gpu},
  {&totem_attrs[0], NULL},
  {&totem_attrs[1], NULL},
  {&totem_attrs[2], NULL},
  {&totem_attrs[3], NULL},
  {&totem_attrs[4], NULL},
  {&totem_attrs[5], NULL},
  {&totem_attrs[6], NULL},
  {&totem_attrs[7], NULL}
};

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
                                                          &bfs_params[4],
                                                          &bfs_params[5],
                                                          &bfs_params[6],
                                                          &bfs_params[7],
                                                          &bfs_params[8],
                                                          &bfs_params[9],
                                                          &bfs_params[10],
                                                          &bfs_params[11]));

#else

// From Google documentation:
// Google Test may not support value-parameterized tests with some
// compilers. This dummy test keeps gtest_main linked in.
TEST_P(DummyTest, ValueParameterizedTestsAreNotSupportedOnThisPlatform) {}

#endif  // GTEST_HAS_PARAM_TEST
