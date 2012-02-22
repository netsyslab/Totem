/* TODO(lauro,abdullah,elizeu): Add license.
 *
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

typedef error_t(*PageRankFunction)(graph_t*, float*, float**);

class PageRankTest : public TestWithParam<PageRankFunction> {
 public:
  virtual void SetUp() {
    // Ensure the minimum CUDA architecture is supported
    CUDA_CHECK_VERSION();
    page_rank = GetParam();
  }

 protected:
  PageRankFunction page_rank;
};

// Tests PageRank for empty graphs.
TEST_P(PageRankTest, Empty) {
  graph_t graph;
  graph.directed = false;
  graph.vertex_count = 0;
  graph.edge_count = 0;
  float* rank = NULL;
  EXPECT_EQ(FAILURE, page_rank(&graph, NULL, &rank));
}

// Tests PageRank for single node graphs.
TEST_P(PageRankTest, SingleNode) {
  graph_t* graph;
  EXPECT_EQ(SUCCESS, graph_initialize(DATA_FOLDER("single_node.totem"),
                                      false, &graph));

  float* rank = NULL;
  EXPECT_EQ(SUCCESS, page_rank(graph, NULL, &rank));
  EXPECT_FALSE(rank == NULL);
  EXPECT_EQ(1, rank[0]);
  mem_free(rank);
  EXPECT_EQ(SUCCESS, graph_finalize(graph));
}

// Tests PageRank for a chain of 1000 nodes.
TEST_P(PageRankTest, Chain) {
  graph_t* graph;
  EXPECT_EQ(SUCCESS, graph_initialize(DATA_FOLDER("chain_1000_nodes.totem"),
                                      false, &graph));

  // the graph should be undirected because the test is shared between the 
  // two versions of the PageRank algorithm: incoming- and outgoing- based.
  EXPECT_FALSE(graph->directed);

  float* rank = NULL;
  EXPECT_EQ(SUCCESS, page_rank(graph, NULL, &rank));
  EXPECT_FALSE(rank == NULL);
  for(id_t vertex = 0; vertex < graph->vertex_count/2; vertex++){
    EXPECT_FLOAT_EQ(rank[vertex], rank[graph->vertex_count - vertex - 1]);
  }
  mem_free(rank);
  EXPECT_EQ(SUCCESS, graph_finalize(graph));
}

// Tests PageRank for a complete graph of 300 nodes.
TEST_P(PageRankTest, CompleteGraph) {
  graph_t* graph;
  EXPECT_EQ(SUCCESS,
            graph_initialize(DATA_FOLDER("complete_graph_300_nodes.totem"),
                             false, &graph));
  // the graph should be undirected because the test is shared between the 
  // two versions of the PageRank algorithm: incoming- and outgoing- based.
  EXPECT_FALSE(graph->directed);

  float* rank = NULL;
  EXPECT_EQ(SUCCESS, page_rank(graph, NULL, &rank));
  EXPECT_FALSE(rank == NULL);
  for(id_t vertex = 0; vertex < graph->vertex_count; vertex++){
    EXPECT_FLOAT_EQ(rank[0], rank[vertex]);
  }
  mem_free(rank);
  EXPECT_EQ(SUCCESS, graph_finalize(graph));
}

// Tests PageRank for a complete graph of 300 nodes.
TEST_P(PageRankTest, Star) {
  graph_t* graph;
  EXPECT_EQ(SUCCESS,
            graph_initialize(DATA_FOLDER("star_1000_nodes.totem"),
                             false, &graph));

  // the graph should be undirected because the test is shared between the 
  // two versions of the PageRank algorithm: incoming- and outgoing- based.
  EXPECT_FALSE(graph->directed);

  float* rank = NULL;
  EXPECT_EQ(SUCCESS, page_rank(graph, NULL, &rank));
  EXPECT_FALSE(rank == NULL);
  for(id_t vertex = 1; vertex < graph->vertex_count; vertex++){
    EXPECT_FLOAT_EQ(rank[1], rank[vertex]);
    EXPECT_GT(rank[0], rank[vertex]);
  }
  mem_free(rank);
  EXPECT_EQ(SUCCESS, graph_finalize(graph));
}

// TODO(abdullah): Add test cases for not well defined structures.
// TODO(abdullah,lauro): Add test cases for non-empty vertex set and empty edge
// set.

// Values() receives a list of parameters and the framework will execute the
// whole set of tests PageRankTest for each element of Values()
// TODO(abdullah): both versions of the PageRank algorithm (the incoming- and 
// outgoing- based) can share the same tests because all the graphs are 
// undirected. Separate the two for cases where the graphs are directed.
INSTANTIATE_TEST_CASE_P(PageRankGPUAndCPUTest, PageRankTest,
                        Values(&page_rank_cpu,
                               &page_rank_gpu,
                               &page_rank_vwarp_gpu,
                               &page_rank_hybrid,
                               &page_rank_incoming_gpu,
                               &page_rank_incoming_cpu));

#else

// From Google documentation:
// Google Test may not support value-parameterized tests with some
// compilers. This dummy test keeps gtest_main linked in.
TEST_P(DummyTest, ValueParameterizedTestsAreNotSupportedOnThisPlatform) {}

#endif  // GTEST_HAS_PARAM_TEST
