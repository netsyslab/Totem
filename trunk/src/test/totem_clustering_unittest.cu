/*
 * Contains unit tests for local clustering coefficient.
 *
 * Created on: 2013-07-09
 * Author: Sidney Pontes Filho
 */

// totem includes
#include "totem_common_unittest.h"

#if GTEST_HAS_PARAM_TEST

using ::testing::TestWithParam;
using ::testing::Values;

// The following implementation relies on
// TestWithParam<ClusteringCoefficientFunction> to test the two versions of 
// Clustering Coefficient implemented: CPU.  Details on how to use 
// TestWithParam<T> can be found at:
// http:
//code.google.com/p/googletest/source/browse/trunk/samples/sample7_unittest.cc

typedef error_t(*ClusteringCoefficientFunction)(const graph_t*, weight_t**);

class ClusteringCoefficientTest : 
public TestWithParam<ClusteringCoefficientFunction> {
 public:
  virtual void SetUp() {
    // Ensure the minimum CUDA architecture is supported
    // CUDA_CHECK_VERSION();
    clustering = GetParam();
  }

 protected:
   ClusteringCoefficientFunction clustering;
   graph_t* graph;
   weight_t* coefficient_score;
};


// Tests ClusteringCoefficient for empty graphs.
TEST_P(ClusteringCoefficientTest, Empty) {
  graph_t empty_graph;
  empty_graph.directed = false;
  empty_graph.vertex_count = 0;
  empty_graph.edge_count = 0;
  EXPECT_EQ(FAILURE, clustering(&empty_graph, &coefficient_score));
}

// Tests ClusteringCoefficient for single node graphs.
TEST_P(ClusteringCoefficientTest, SingleNodeUnweighted) {
  graph_t* graph;
  EXPECT_EQ(SUCCESS, graph_initialize(DATA_FOLDER("single_node.totem"),
                                      false, &graph));

  EXPECT_EQ(SUCCESS, clustering(graph, &coefficient_score));
  EXPECT_FALSE(coefficient_score == NULL);
  EXPECT_EQ((weight_t)0.0, coefficient_score[0]);
  totem_free(coefficient_score, TOTEM_MEM_HOST_PINNED);
  EXPECT_EQ(SUCCESS, graph_finalize(graph));
}

// Tests ClusteringCoefficient for a complete graph of 300 undirected nodes.
TEST_P(ClusteringCoefficientTest, CompleteGraph300NodesUndirected) {
  EXPECT_EQ(SUCCESS,
            graph_initialize(DATA_FOLDER("complete_graph_300_nodes.totem"),
                             false, &graph));

  EXPECT_EQ(SUCCESS, clustering(graph, &coefficient_score));
  EXPECT_FALSE(coefficient_score == NULL);
  for(vid_t vertex = 0; vertex < graph->vertex_count; vertex++){
    EXPECT_FLOAT_EQ(1.0, coefficient_score[0]);
  }
  totem_free(coefficient_score, TOTEM_MEM_HOST_PINNED);
  EXPECT_EQ(SUCCESS, graph_finalize(graph));
}


// Tests ClusteringCoefficient for a complete acyclic graph of 100 directed 
// nodes.
TEST_P(ClusteringCoefficientTest, CompleteAcyclicGraph100NodesDirected) {
  EXPECT_EQ(SUCCESS,
            graph_initialize(DATA_FOLDER("acyclic_100_nodes.totem"),
                             false, &graph));

  EXPECT_EQ(SUCCESS, clustering(graph, &coefficient_score));
  EXPECT_FALSE(coefficient_score == NULL);
  for(vid_t vertex = 0; vertex < graph->vertex_count; vertex++){
    EXPECT_FLOAT_EQ(1.0, coefficient_score[0]);
  }
  totem_free(coefficient_score, TOTEM_MEM_HOST_PINNED);
  EXPECT_EQ(SUCCESS, graph_finalize(graph));
}


// From Google documentation:
// In order to run value-parameterized tests, we need to instantiate them,
// or bind them to a list of values which will be used as test parameters.
//
// Values() receives a list of parameters and the framework will execute the
// whole set of tests ClusteringCoefficientTest for each element of Values()
INSTANTIATE_TEST_CASE_P(ClusteringCoefficientGPUandCPUTest, 
                        ClusteringCoefficientTest,
                        Values(&local_clustering_coefficient_cpu));

#else

// From Google documentation:
// Google Test may not support value-parameterized tests with some
// compilers. This dummy test keeps gtest_main linked in.
TEST_P(DummyTest, ValueParameterizedTestsAreNotSupportedOnThisPlatform) {}

#endif  // GTEST_HAS_PARAM_TEST
