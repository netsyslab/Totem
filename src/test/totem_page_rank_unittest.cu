/*
 * Contains unit tests for an implementation of the PageRank graph algorithm.
 *
 *  Created on: 2011-03-22
 *      Author: Abdullah Gharaibeh
 */

// totem includes
#include "totem_common_unittest.h"

#if GTEST_HAS_PARAM_TEST

using ::testing::TestWithParam;
using ::testing::Values;

// The following implementation relies on TestWithParam<PageRankFunction> to
// test the two versions of PageRank implemented: CPU and GPU.
// Details on how to use TestWithParam<T> can be found at:
// totem_bfs_unittest.cc and
// http://code.google.com/p/googletest/source/browse/trunk/samples/sample7_unittest.cc

typedef error_t(*PageRankFunction)(graph_t*, rank_t*, rank_t*);

// This is to allow testing the vanilla bfs functions and the hybrid one
// that is based on the framework. Note that have a different signature
// of the hybrid algorithm forced this work-around.
typedef struct page_rank_param_s {
  bool             hybrid; // true when using the hybrid algorithm
  PageRankFunction func;   // the vanilla page_rank function if hybrid
                           // flag is false
} page_rank_param_t;

class PageRankTest : public TestWithParam<page_rank_param_t*> {
 public:
  virtual void SetUp() {
    // Ensure the minimum CUDA architecture is supported
    CUDA_CHECK_VERSION();
    _page_rank_param = GetParam();
    _rank = NULL;
    _graph = NULL;
  }

  virtual void TearDown() {
    if (_graph) graph_finalize(_graph);
    if (_rank) totem_free(_rank, TOTEM_MEM_HOST_PINNED);
  }

  error_t TestGraph() {
    // the graph should be undirected because the test is shared between the
    // two versions of the PageRank algorithm: incoming- and outgoing- based.
    EXPECT_FALSE(_graph->directed);
    CALL_SAFE(totem_malloc(_graph->vertex_count * sizeof(rank_t), 
                           TOTEM_MEM_HOST_PINNED, (void**)&_rank));
    if (_page_rank_param->hybrid) {
      totem_attr_t attr = TOTEM_DEFAULT_ATTR;
      attr.push_msg_size = sizeof(rank_t) * BITS_PER_BYTE;
      if (totem_init(_graph, &attr) == FAILURE) {
        return FAILURE;
      }
      error_t err = page_rank_hybrid(NULL, _rank);
      totem_finalize();
      return err;
    }
    return _page_rank_param->func(_graph, NULL, _rank);
  }

 protected:
  page_rank_param_t* _page_rank_param;
  rank_t* _rank;
  graph_t* _graph;
};

// Tests PageRank for empty graphs.
TEST_P(PageRankTest, Empty) {
  _graph = (graph_t*)calloc(sizeof(graph_t), 1);
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
  for(vid_t vertex = 0; vertex < _graph->vertex_count/2; vertex++){
    EXPECT_FLOAT_EQ(_rank[vertex], _rank[_graph->vertex_count - vertex - 1]);
  }
}

// Tests PageRank for a complete graph of 300 nodes.
TEST_P(PageRankTest, CompleteGraph) {
  EXPECT_EQ(SUCCESS,
            graph_initialize(DATA_FOLDER("complete_graph_300_nodes.totem"),
                             false, &_graph));
  EXPECT_EQ(SUCCESS, TestGraph());
  for(vid_t vertex = 0; vertex < _graph->vertex_count; vertex++){
    EXPECT_FLOAT_EQ(_rank[0], _rank[vertex]);
  }
}

// Tests PageRank for a complete graph of 300 nodes.
TEST_P(PageRankTest, Star) {
  EXPECT_EQ(SUCCESS,
            graph_initialize(DATA_FOLDER("star_1000_nodes.totem"),
                             false, &_graph));
  EXPECT_EQ(SUCCESS, TestGraph());
  for(vid_t vertex = 1; vertex < _graph->vertex_count; vertex++){
    EXPECT_FLOAT_EQ(_rank[1], _rank[vertex]);
    EXPECT_GT(_rank[0], _rank[vertex]);
  }
}

// TODO(abdullah): Add test cases for not well defined structures.
// TODO(abdullah,lauro): Add test cases for non-empty vertex set and empty edge
// set.

// Values() seems to accept only pointers, hence the possible parameters
// are defined here, and a pointer to each ot them is used.
page_rank_param_t page_rank_params[] = {{false, &page_rank_cpu},
                              {false, &page_rank_gpu},
                              {false, &page_rank_vwarp_gpu},
                              {false, &page_rank_incoming_cpu},
                              {false, &page_rank_incoming_gpu},
                              {true, NULL}};

// Values() receives a list of parameters and the framework will execute the
// whole set of tests PageRankTest for each element of Values()
// TODO(abdullah): both versions of the PageRank algorithm (the incoming- and
// outgoing- based) can share the same tests because all the graphs are
// undirected. Separate the two for cases where the graphs are directed.
INSTANTIATE_TEST_CASE_P(PageRankGPUAndCPUTest, PageRankTest,
                        Values(&page_rank_params[0],
                               &page_rank_params[1],
                               &page_rank_params[2],
                               &page_rank_params[3],
                               &page_rank_params[4],
                               &page_rank_params[5]));

#else

// From Google documentation:
// Google Test may not support value-parameterized tests with some
// compilers. This dummy test keeps gtest_main linked in.
TEST_P(DummyTest, ValueParameterizedTestsAreNotSupportedOnThisPlatform) {}

#endif  // GTEST_HAS_PARAM_TEST
