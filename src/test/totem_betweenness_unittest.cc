/* TODO(lauro,abdullah,elizeu): Add license.
 *
 * Contains unit tests for betweenness centrality.
 *
 *  Created on: 2011-10-21
 *      Author: Greg Redekop
 */

// totem includes
#include "totem_common_unittest.h"

#if GTEST_HAS_PARAM_TEST

using ::testing::TestWithParam;
using ::testing::Values;

// The following implementation relies on
// TestWithParam<BetwCentralityFunction> to test the two versions of Betweenness
// Centrality implemented: CPU and GPU.  Details on how to use TestWithParam<T>
// can be found at:
// http://code.google.com/p/googletest/source/browse/trunk/samples/sample7_unittest.cc

typedef error_t(*BetwCentralityFunction)(const graph_t*, weight_t*);

class BetweennessCentralityTest : public TestWithParam<BetwCentralityFunction> {
 public:
  virtual void SetUp() {
    // Ensure the minimum CUDA architecture is supported
    CUDA_CHECK_VERSION();
    betweenness = GetParam();
    graph = NULL;
    centrality_score = NULL;
  }

  virtual void TearDown() {
    if (graph) graph_finalize(graph);
    if (centrality_score) mem_free(centrality_score);
  }

 protected:
   BetwCentralityFunction betweenness;
   graph_t* graph;
   weight_t* centrality_score;
};

// Tests BetwCentrality for empty graphs.
TEST_P(BetweennessCentralityTest, Empty) {
  graph_t graph;
  graph.directed = false;
  graph.vertex_count = 0;
  graph.edge_count = 0;

  EXPECT_EQ(FAILURE, betweenness(&graph, centrality_score));
  EXPECT_EQ(FAILURE, betweenness(&graph, centrality_score));
  EXPECT_EQ(FAILURE, betweenness(&graph, centrality_score));
}

// Tests BetwCentrality for single node graphs.
TEST_P(BetweennessCentralityTest, SingleNodeUnweighted) {
  EXPECT_EQ(SUCCESS, graph_initialize(DATA_FOLDER("single_node.totem"),
                                      false, &graph));
  centrality_score = (weight_t*)mem_alloc(graph->vertex_count * 
                                          sizeof(weight_t));

  EXPECT_EQ(SUCCESS, betweenness(graph, centrality_score));
  EXPECT_EQ((weight_t)0.0, centrality_score[0]);
}

// Tests BetwCentrality for a chain of 100 nodes.
TEST_P(BetweennessCentralityTest, Chain100Unweighted) {
  graph_initialize(DATA_FOLDER("chain_100_nodes_weight_directed.totem"), false,
                   &graph);
  centrality_score = (weight_t*)mem_alloc(graph->vertex_count * 
                                          sizeof(weight_t));

  // First vertex as source
  EXPECT_EQ(SUCCESS, betweenness(graph, centrality_score));
  weight_t centrality[50];
  for (vid_t i = 0; i < 50; i++) {
    centrality[i] = (99 - i) * (i);
  }
  for (vid_t i = 0; i < 50; i++) {
    EXPECT_EQ(centrality[i], centrality_score[i]);
    EXPECT_EQ(centrality[i], centrality_score[99 - i]);
  }
}

// Tests BetwCentrality for a complete graph of 300 nodes.
TEST_P(BetweennessCentralityTest, CompleteGraphUnweighted) {
  EXPECT_EQ(SUCCESS,
            graph_initialize(DATA_FOLDER("complete_graph_300_nodes.totem"),
                             false, &graph));
  centrality_score = (weight_t*)mem_alloc(graph->vertex_count * 
                                          sizeof(weight_t));

  EXPECT_EQ(SUCCESS, betweenness(graph, centrality_score));
  for(vid_t vertex = 0; vertex < graph->vertex_count; vertex++){
    EXPECT_FLOAT_EQ(0.0, centrality_score[0]);
  }
}

// From Google documentation:
// In order to run value-parameterized tests, we need to instantiate them,
// or bind them to a list of values which will be used as test parameters.
//
// Values() receives a list of parameters and the framework will execute the
// whole set of tests BetweennessCentralityTest for each element of Values()
INSTANTIATE_TEST_CASE_P(BetwCentralityGPUAndCPUTest, BetweennessCentralityTest,
                        Values(&betweenness_unweighted_cpu,
                               &betweenness_unweighted_gpu,
                               &betweenness_unweighted_shi_gpu,
                               &betweenness_cpu));

#else

// From Google documentation:
// Google Test may not support value-parameterized tests with some
// compilers. This dummy test keeps gtest_main linked in.
TEST_P(DummyTest, ValueParameterizedTestsAreNotSupportedOnThisPlatform) {}

#endif  // GTEST_HAS_PARAM_TEST
