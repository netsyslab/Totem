/*
 * Contains unit tests for the different implementations of the PageRank graph
 * algorithm.
 *
 *  Created on: 2011-03-22
 *      Author: Abdullah Gharaibeh
 */

// totem includes
#include "totem_common_unittest.h"

#if GTEST_HAS_PARAM_TEST

using ::testing::TestWithParam;
using ::testing::ValuesIn;


// IMPORTANT NOTE: Some of the unit tests for page_rank_incoming_hybrid will
// fail if the type rank_t is float due to differences in single precision
// calculations between the CPU and the GPU. Defining rank_t as double should
// allow the tests to pass.

// The following implementation relies on TestWithParam<PageRankFunction> to
// test the different versions of PageRank. Details on how to use
// TestWithParam<T> can be found at:
// http://code.google.com/p/googletest/source/browse/trunk/samples/sample7_unittest.cc

typedef error_t(*PageRankFunction)(graph_t*, rank_t*, rank_t*);
typedef error_t(*PageRankHybridFunction)(rank_t*, rank_t*);

class PageRankTest : public TestWithParam<test_param_t*> {
 public:
  virtual void SetUp() {
    // Ensure the minimum CUDA architecture is supported.
    CUDA_CHECK_VERSION();
    _page_rank_param = GetParam();
    _rank = NULL;
    _graph = NULL;
  }

  virtual void TearDown() {
    if (_graph) { graph_finalize(_graph); }
    if (_rank) { totem_free(_rank, TOTEM_MEM_HOST_PINNED); }
  }

  error_t TestGraph() {
    // The graph should be undirected because the test is shared between the
    // two versions of the PageRank algorithm: incoming- and outgoing- based.
    EXPECT_FALSE(_graph->directed);
    if (_graph->vertex_count != 0) {
      CALL_SAFE(totem_malloc(_graph->vertex_count * sizeof(rank_t),
                             TOTEM_MEM_HOST_PINNED,
                             reinterpret_cast<void**>(&_rank)));
    }
    if (_page_rank_param->attr != NULL) {
      _page_rank_param->attr->pull_msg_size = sizeof(rank_t) * BITS_PER_BYTE;
      _page_rank_param->attr->push_msg_size = sizeof(rank_t) * BITS_PER_BYTE;
      if (totem_init(_graph, _page_rank_param->attr) == FAILURE) {
        return FAILURE;
      }
      PageRankHybridFunction func =
          reinterpret_cast<PageRankHybridFunction>(_page_rank_param->func);
      error_t err = func(NULL, _rank);
      totem_finalize();
      return err;
    } else {
      PageRankFunction func =
          reinterpret_cast<PageRankFunction>(_page_rank_param->func);
      return func(_graph, NULL, _rank);
    }
  }

 protected:
  test_param_t* _page_rank_param;
  rank_t* _rank;
  graph_t* _graph;
};

// Tests PageRank for empty graphs.
TEST_P(PageRankTest, Empty) {
  _graph = reinterpret_cast<graph_t*>(calloc(sizeof(graph_t), 1));
  EXPECT_EQ(FAILURE, TestGraph());
  free(_graph);
  _graph = NULL;
}

// Tests PageRank for single node graphs.
TEST_P(PageRankTest, SingleNode) {
  EXPECT_EQ(SUCCESS, graph_initialize(DATA_FOLDER("single_node.totem"),
                                      false, &_graph));
  _graph->directed = false;
  EXPECT_EQ(SUCCESS, TestGraph());
  EXPECT_EQ(1, _rank[0]);
}

// Tests PageRank for a chain of 1000 nodes.
TEST_P(PageRankTest, Chain) {
  EXPECT_EQ(SUCCESS, graph_initialize(DATA_FOLDER("chain_1000_nodes.totem"),
                                      false, &_graph));
  EXPECT_EQ(SUCCESS, TestGraph());
  for (vid_t vertex = 0; vertex < _graph->vertex_count / 2; vertex++) {
    EXPECT_FLOAT_EQ(_rank[vertex], _rank[_graph->vertex_count - vertex - 1]);
  }
}

// Tests PageRank for a complete graph of 300 nodes.
TEST_P(PageRankTest, CompleteGraph) {
  EXPECT_EQ(SUCCESS,
            graph_initialize(DATA_FOLDER("complete_graph_300_nodes.totem"),
                             false, &_graph));
  EXPECT_EQ(SUCCESS, TestGraph());
  for (vid_t vertex = 0; vertex < _graph->vertex_count; vertex++) {
    EXPECT_FLOAT_EQ(_rank[0], _rank[vertex]);
  }
}

// Tests PageRank for a complete graph of 300 nodes.
TEST_P(PageRankTest, Star) {
  EXPECT_EQ(SUCCESS,
            graph_initialize(DATA_FOLDER("star_1000_nodes.totem"),
                             false, &_graph));
  EXPECT_EQ(SUCCESS, TestGraph());
  for (vid_t vertex = 1; vertex < _graph->vertex_count; vertex++) {
    EXPECT_FLOAT_EQ(_rank[1], _rank[vertex]);
    EXPECT_GT(_rank[0], _rank[vertex]);
  }
}

// Defines the set of PageRank vanilla implementations to be tested. To test
// a new implementation, simply add it to the set below.
void* page_rank_vanilla_funcs[] = {
  reinterpret_cast<void*>(&page_rank_cpu),
  reinterpret_cast<void*>(&page_rank_incoming_cpu),
  reinterpret_cast<void*>(&page_rank_gpu),
  reinterpret_cast<void*>(&page_rank_vwarp_gpu),
  reinterpret_cast<void*>(&page_rank_incoming_gpu),
};
const int page_rank_vanilla_count = STATIC_ARRAY_COUNT(page_rank_vanilla_funcs);

// Defines the set of PageRank hybrid implementations to be tested. To test
// a new implementation, simply add it to the set below.
void* page_rank_hybrid_funcs[] = {
  reinterpret_cast<void*>(&page_rank_hybrid),
  reinterpret_cast<void*>(&page_rank_incoming_hybrid),
};
const int page_rank_hybrid_count = STATIC_ARRAY_COUNT(page_rank_hybrid_funcs);

// Maintains references to the different configurations (vanilla and hybrid)
// that will be tested by the framework.
static const int page_rank_params_count = page_rank_vanilla_count +
    page_rank_hybrid_count * hybrid_configurations_count;
static test_param_t* page_rank_params[page_rank_params_count];

// From Google documentation:
// In order to run value-parameterized tests, we need to instantiate them,
// or bind them to a list of values which will be used as test parameters.
//
// ValuesIn() receives a list of parameters and the framework will execute the
// whole set of tests for each entry in the array passed to ValuesIn().
INSTANTIATE_TEST_CASE_P(PageRankGPUAndCPUTest, PageRankTest,
                        ValuesIn(GetParameters(
                            page_rank_params, page_rank_params_count,
                            page_rank_vanilla_funcs, page_rank_vanilla_count,
                            page_rank_hybrid_funcs, page_rank_hybrid_count),
                                 page_rank_params + page_rank_params_count));

#else

// From Google documentation:
// Google Test may not support value-parameterized tests with some
// compilers. This dummy test keeps gtest_main linked in.
TEST_P(DummyTest, ValueParameterizedTestsAreNotSupportedOnThisPlatform) {}

#endif  // GTEST_HAS_PARAM_TEST
