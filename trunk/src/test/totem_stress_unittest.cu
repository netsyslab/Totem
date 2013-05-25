/* TODO(lauro,abdullah,elizeu): Add license.
 *
 * Contains unit tests for stress centrality.
 *
 *  Created on: 2012-06-06
 *      Author: Greg Redekop
 */

// totem includes
#include "totem_common_unittest.h"

#if GTEST_HAS_PARAM_TEST

using ::testing::TestWithParam;
using ::testing::Values;

// The following implementation relies on
// TestWithParam<StressCentralityFunction> to test the two versions of Stress
// Centrality implemented: CPU and GPU.  Details on how to use TestWithParam<T>
// can be found at:
// http://code.google.com/p/googletest/source/browse/trunk/samples/sample7_unittest.cc

typedef error_t(*StressCentralityFunction)(const graph_t*, weight_t**);

class StressCentralityTest : public TestWithParam<StressCentralityFunction> {
 public:
  virtual void SetUp() {
    // Ensure the minimum CUDA architecture is supported
    CUDA_CHECK_VERSION();
    stress = GetParam();
  }

 protected:
   StressCentralityFunction stress;
   graph_t* graph;
   weight_t* centrality_score;
};

// Tests StressCentrality for empty graphs.
TEST_P(StressCentralityTest, Empty) {
  graph_t empty_graph;
  empty_graph.directed = false;
  empty_graph.vertex_count = 0;
  empty_graph.edge_count = 0;
  EXPECT_EQ(FAILURE, stress(&empty_graph, &centrality_score));
}

// Tests StressCentrality for single node graphs.
TEST_P(StressCentralityTest, SingleNodeUnweighted) {
  graph_t* graph;
  EXPECT_EQ(SUCCESS, graph_initialize(DATA_FOLDER("single_node.totem"),
                                      false, &graph));

  EXPECT_EQ(SUCCESS, stress(graph, &centrality_score));
  EXPECT_FALSE(centrality_score == NULL);
  EXPECT_EQ((weight_t)0.0, centrality_score[0]);
  totem_free(centrality_score, TOTEM_MEM_HOST_PINNED);
  EXPECT_EQ(SUCCESS, graph_finalize(graph));
}

// Tests StressCentrality for a chain of 100 nodes.
TEST_P(StressCentralityTest, Chain100Unweighted) {
  graph_initialize(DATA_FOLDER("chain_100_nodes.totem"), false,
                   &graph);

  // First vertex as source
  EXPECT_EQ(SUCCESS, stress(graph, &centrality_score));
  EXPECT_EQ(centrality_score[0], (weight_t)0.0);
  EXPECT_EQ(centrality_score[99], (weight_t)0.0);
  for (vid_t i = 1; i < 50; i++) {
    EXPECT_EQ((weight_t)(2 * ((99 * i) - (i * i))), centrality_score[i]);
    EXPECT_EQ((weight_t)(2 * ((99 * i) - (i * i))), centrality_score[99 - i]);
  }

  totem_free(centrality_score, TOTEM_MEM_HOST_PINNED);
  graph_finalize(graph);
}

// Tests StressCentrality for a complete graph of 300 nodes.
TEST_P(StressCentralityTest, CompleteGraphUnweighted) {
  EXPECT_EQ(SUCCESS,
            graph_initialize(DATA_FOLDER("complete_graph_300_nodes.totem"),
                             false, &graph));

  EXPECT_EQ(SUCCESS, stress(graph, &centrality_score));
  EXPECT_FALSE(centrality_score == NULL);
  for(vid_t vertex = 0; vertex < graph->vertex_count; vertex++){
    EXPECT_FLOAT_EQ(0.0, centrality_score[0]);
  }
  totem_free(centrality_score, TOTEM_MEM_HOST_PINNED);
  EXPECT_EQ(SUCCESS, graph_finalize(graph));
}

// From Google documentation:
// In order to run value-parameterized tests, we need to instantiate them,
// or bind them to a list of values which will be used as test parameters.
//
// Values() receives a list of parameters and the framework will execute the
// whole set of tests StressCentralityTest for each element of Values()
INSTANTIATE_TEST_CASE_P(StressCentralityGPUAndCPUTest, StressCentralityTest,
                        Values(&stress_unweighted_cpu, &stress_unweighted_gpu));

#else

// From Google documentation:
// Google Test may not support value-parameterized tests with some
// compilers. This dummy test keeps gtest_main linked in.
TEST_P(DummyTest, ValueParameterizedTestsAreNotSupportedOnThisPlatform) {}

#endif  // GTEST_HAS_PARAM_TEST
