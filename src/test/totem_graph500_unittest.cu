/*
 * Contains unit tests for an implementation of the Graph500 benchmark
 * graph search algorithm.
 *
 *  Created on: 2013-05-31
 *      Author: Abdullah Gharaibeh
 */

// totem includes
#include "totem_common_unittest.h"

#if GTEST_HAS_PARAM_TEST

using ::testing::TestWithParam;
using ::testing::ValuesIn;

typedef error_t(*GRAPH500Function)(graph_t*, vid_t, bfs_tree_t*);
typedef error_t(*GRAPH500HybridFunction)(vid_t, bfs_tree_t*);

class Graph500Test : public TestWithParam<test_param_t*> {
 public:
  virtual void SetUp() {
    CUDA_CHECK_VERSION();
    _graph500_param = GetParam();
    _mem_type = TOTEM_MEM_HOST_PINNED;
    _graph = NULL;
    _tree = NULL;
  }
  virtual void TearDown() {
    if (_graph) graph_finalize(_graph);
  }

  void InitTestCase(const char* filename) {
    graph_initialize(filename, false, &_graph);
    CALL_SAFE(totem_malloc(_graph->vertex_count * sizeof(bfs_tree_t), _mem_type,
                           reinterpret_cast<void**>(&_tree)));
  }

  void FinalizeTestCase() {
    totem_free(_tree, _mem_type);
  }

  error_t TestGraph(vid_t src) {
    if (_graph500_param->attr) {
      _graph500_param->attr->push_msg_size =
          (sizeof(vid_t) * BITS_PER_BYTE) + 1;
      _graph500_param->attr->alloc_func = _graph500_param->hybrid_alloc;
      _graph500_param->attr->free_func = _graph500_param->hybrid_free;
      if (totem_init(_graph, _graph500_param->attr) == FAILURE) {
        return FAILURE;
      }
      GRAPH500HybridFunction func =
          reinterpret_cast<GRAPH500HybridFunction>(_graph500_param->func);
      error_t err = func(src, _tree);
      totem_finalize();
      return err;
    } else {
      GRAPH500Function func = reinterpret_cast<GRAPH500Function>(
          _graph500_param->func);
      return func(_graph, src, _tree);
    }
  }

 protected:
  test_param_t* _graph500_param;
  totem_mem_t _mem_type;
  graph_t* _graph;
  bfs_tree_t* _tree;
};

TEST_P(Graph500Test, Empty) {
  _graph = reinterpret_cast<graph_t*>(calloc(1, sizeof(graph_t)));
  EXPECT_EQ(FAILURE, TestGraph(0));
  EXPECT_EQ(FAILURE, TestGraph(99));
  free(_graph);
  _graph = NULL;
}

TEST_P(Graph500Test, SingleNode) {
  InitTestCase(DATA_FOLDER("single_node.totem"));
  EXPECT_EQ(SUCCESS, TestGraph(0));
  EXPECT_EQ((vid_t)0, (vid_t)_tree[0]);
  EXPECT_EQ(FAILURE, TestGraph(1));
  FinalizeTestCase();
}

TEST_P(Graph500Test, SingleNodeLoop) {
  InitTestCase(DATA_FOLDER("single_node_loop.totem"));
  EXPECT_EQ(SUCCESS, TestGraph(0));
  EXPECT_EQ((vid_t)0, (vid_t)_tree[0]);
  EXPECT_EQ(FAILURE, TestGraph(1));
  FinalizeTestCase();
}

// Completely disconnected graph
TEST_P(Graph500Test, Disconnected) {
  InitTestCase(DATA_FOLDER("disconnected_1000_nodes.totem"));

  // First vertex as source
  vid_t source = 0;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  EXPECT_EQ(source, (vid_t)_tree[source]);
  for (vid_t vertex = source + 1; vertex < _graph->vertex_count; vertex++) {
    EXPECT_EQ(VERTEX_ID_MAX, (vid_t)_tree[vertex]);
  }

  // Last vertex as source
  source = _graph->vertex_count - 1;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  EXPECT_EQ(source, (vid_t)_tree[source]);
  for (vid_t vertex = source; vertex < _graph->vertex_count - 1; vertex++) {
    EXPECT_EQ(VERTEX_ID_MAX, (vid_t)_tree[vertex]);
  }

  // A vertex in the middle as source
  source = 199;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  for (vid_t vertex = 0; vertex < _graph->vertex_count; vertex++) {
    EXPECT_EQ((vertex == source) ? source :
              VERTEX_ID_MAX, (vid_t)_tree[vertex]);
  }

  // Non existent vertex source
  EXPECT_EQ(FAILURE, TestGraph(_graph->vertex_count));

  FinalizeTestCase();
}

// Chain of 1000 nodes.
TEST_P(Graph500Test, Chain) {
  InitTestCase(DATA_FOLDER("chain_1000_nodes.totem"));

  // First vertex as source
  vid_t source = 0;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  EXPECT_EQ(source, (vid_t)_tree[source]);
  for (vid_t vertex = source + 1; vertex < _graph->vertex_count; vertex++) {
    EXPECT_EQ((vertex - 1), (vid_t)_tree[vertex]);
  }

  // Last vertex as source
  source = _graph->vertex_count - 1;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  EXPECT_EQ(source, (vid_t)_tree[source]);
  for (vid_t vertex = source; vertex < _graph->vertex_count - 1; vertex++) {
    EXPECT_EQ((vertex + 1), (vid_t)_tree[vertex]);
  }

  // A vertex in the middle as source
  source = 199;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  for (vid_t vertex = 0; vertex < _graph->vertex_count; vertex++) {
    if (vertex > source) {
      EXPECT_EQ((vertex - 1), (vid_t)_tree[vertex]);
    } else if (vertex < source) {
      EXPECT_EQ((vertex + 1), (vid_t)_tree[vertex]);
    } else {
      EXPECT_EQ(source, (vid_t)_tree[vertex]);
    }
  }

  // Non existent vertex source
  EXPECT_EQ(FAILURE, TestGraph(_graph->vertex_count));

  FinalizeTestCase();
}

// Complete graph of 300 nodes.
TEST_P(Graph500Test, CompleteGraph) {
  InitTestCase(DATA_FOLDER("complete_graph_300_nodes.totem"));

  // First vertex as source
  vid_t source = 0;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  for (vid_t vertex = 0; vertex < _graph->vertex_count; vertex++) {
    EXPECT_EQ(source, (vid_t)_tree[vertex]);
  }

  // Last vertex as source
  source = _graph->vertex_count - 1;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  for (vid_t vertex = 0; vertex < _graph->vertex_count; vertex++) {
    EXPECT_EQ(source, (vid_t)_tree[vertex]);
  }

  // A vertex source in the middle
  source = 199;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  for (vid_t vertex = 0; vertex < _graph->vertex_count; vertex++) {
    EXPECT_EQ(source, (vid_t)_tree[vertex]);
  }

  // Non existent vertex source
  EXPECT_EQ(FAILURE, TestGraph(_graph->vertex_count));

  FinalizeTestCase();
}

// Star graph of 1000 nodes.
TEST_P(Graph500Test, Star) {
  InitTestCase(DATA_FOLDER("star_1000_nodes.totem"));

  // First vertex as source
  vid_t source = 0;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  for (vid_t vertex = 0; vertex < _graph->vertex_count; vertex++) {
    EXPECT_EQ(source, (vid_t)_tree[vertex]);
  }

  // Last vertex as source
  source = _graph->vertex_count - 1;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  EXPECT_EQ(source, (vid_t)_tree[source]);
  EXPECT_EQ(source, (vid_t)_tree[0]);
  for (vid_t vertex = 1; vertex < _graph->vertex_count; vertex++) {
    if (vertex == source) continue;
    EXPECT_EQ((vid_t)0, (vid_t)_tree[vertex]);
  }

  // A vertex source in the middle
  source = 199;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  EXPECT_EQ(source, (vid_t)_tree[source]);
  EXPECT_EQ(source, (vid_t)_tree[0]);
  for (vid_t vertex = 1; vertex < _graph->vertex_count; vertex++) {
    if (vertex == source) continue;
    EXPECT_EQ((vid_t)0, (vid_t)_tree[vertex]);
  }

  // Non existent vertex source
  EXPECT_EQ(FAILURE, TestGraph(_graph->vertex_count));

  FinalizeTestCase();
}

// Defines the set of GRAPH500 vanilla implementations to be tested. To test
// a new implementation, simply add it to the set below.
void* graph500_vanilla_funcs[] = {
  reinterpret_cast<void*>(&graph500_cpu),
};
const int graph500_vanilla_count = STATIC_ARRAY_COUNT(graph500_vanilla_funcs);

// Defines the set of GRAPH500 hybrid implementations to be tested. To test
// a new implementation, simply add it to the set below.
void* graph500_hybrid_funcs[] = {
  reinterpret_cast<void*>(&graph500_hybrid),
};
totem_cb_func_t graph500_hybrid_alloc_funcs[] = {
  &graph500_alloc
};
totem_cb_func_t graph500_hybrid_free_funcs[] = {
  &graph500_free
};
const int graph500_hybrid_count = STATIC_ARRAY_COUNT(graph500_hybrid_funcs);

// Maintains references to the different configurations (vanilla and hybrid)
// that will be tested by the framework.
static const int graph500_params_count = graph500_vanilla_count +
    graph500_hybrid_count * hybrid_configurations_count;
static test_param_t* graph500_params[graph500_params_count];

// From Google documentation:
// In order to run value-parameterized tests, we need to instantiate them,
// or bind them to a list of values which will be used as test parameters.
//
// Values() receives a list of parameters and the framework will execute the
// whole set of tests GRAPH500Test for each element of Values()
INSTANTIATE_TEST_CASE_P(GRAPH500GPUAndCPUTest, Graph500Test,
                        ValuesIn(GetParameters(
                            graph500_params, graph500_params_count,
                            graph500_vanilla_funcs, graph500_vanilla_count,
                            graph500_hybrid_funcs, graph500_hybrid_count,
                            graph500_hybrid_alloc_funcs,
                            graph500_hybrid_free_funcs),
                                 graph500_params + graph500_params_count));

#else

// From Google documentation:
// Google Test may not support value-parameterized tests with some
// compilers. This dummy test keeps gtest_main linked in.
TEST_P(DummyTest, ValueParameterizedTestsAreNotSupportedOnThisPlatform) {}

#endif  // GTEST_HAS_PARAM_TEST
