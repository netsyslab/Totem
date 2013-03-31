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

/**
 * Wrapper for betweenness_cpu to provide the singature expected for use in
 * the unit tests with the other Betweenness Centrality algorithms
 */
PRIVATE error_t betweenness_cpu_exact(const graph_t* graph,
                                      score_t* betweenness_score) {
  // call betweenness_cpu for use in unit test framework with exact precision
  return betweenness_cpu(graph, CENTRALITY_EXACT, betweenness_score);
}   

/**
 * Wrapper for betweenness_gpu to provide the singature expected for use in
 * the unit tests with the other Betweenness Centrality algorithms
 */
PRIVATE error_t betweenness_gpu_exact(const graph_t* graph,
                                      score_t* betweenness_score) {
  // call betweenness_gpu for use in unit test framework with exact precision
  return betweenness_gpu(graph, CENTRALITY_EXACT, betweenness_score);
}

// The following implementation relies on
// TestWithParam<BetwCentralityFunction> to test the two versions of Betweenness
// Centrality implemented: CPU and GPU.  Details on how to use TestWithParam<T>
// can be found at:
// http://code.google.com/p/googletest/source/browse/trunk/samples/sample7_unittest.cc

typedef error_t(*BetwCentralityFunction)(const graph_t*, score_t*);

// This is to allow testing the vanilla BC functions and the hybrid one
// that is based on the framework. Note that have a different signature
// of the hybrid algorithm forced this work-around.
typedef struct betweenness_param_s {
  bool                     hybrid; // true when using the hybrid algorithm
  BetwCentralityFunction   func;   // the vanilla bfs function if not hybrid
} betweenness_param_t;

class BetweennessCentralityTest : public TestWithParam<betweenness_param_t*> {
 public:
  virtual void SetUp() {
    // Ensure the minimum CUDA architecture is supported
    CUDA_CHECK_VERSION();
    betweenness_param = GetParam();
    graph = NULL;
    centrality_score = NULL;
  }

  virtual void TearDown() {
    if (graph) graph_finalize(graph);
    if (centrality_score) mem_free(centrality_score);
  }

  error_t TestGraph(graph_t* graph, score_t* score) {
    if (betweenness_param->hybrid) {
      totem_attr_t attr = TOTEM_DEFAULT_ATTR;
      attr.push_msg_size = sizeof(uint32_t) * BITS_PER_BYTE;
      attr.pull_msg_size = sizeof(betweenness_backward_t) * BITS_PER_BYTE;

      if (totem_init(graph, &attr) == FAILURE) {
        return FAILURE;
      }

      // Will use exact betweenness centrality for test framework
      error_t err = betweenness_hybrid(CENTRALITY_EXACT, score);
      totem_finalize();
      return err;
    }
    return betweenness_param->func(graph, score);
  }

 protected:
   betweenness_param_t* betweenness_param;
   graph_t* graph;
   score_t* centrality_score;
};

// Tests BetwCentrality for empty graphs.
TEST_P(BetweennessCentralityTest, Empty) {
  graph_t graph;
  graph.directed = false;
  graph.vertex_count = 0;
  graph.edge_count = 0;

  EXPECT_EQ(FAILURE, TestGraph(&graph, centrality_score));
  EXPECT_EQ(FAILURE, TestGraph(&graph, centrality_score));
  EXPECT_EQ(FAILURE, TestGraph(&graph, centrality_score));
}

// Tests BetwCentrality for single node graphs.
TEST_P(BetweennessCentralityTest, SingleNodeUnweighted) {
  EXPECT_EQ(SUCCESS, graph_initialize(DATA_FOLDER("single_node.totem"),
                                      false, &graph));
  centrality_score = (score_t*)mem_alloc(graph->vertex_count * 
                                          sizeof(score_t));

  EXPECT_EQ(SUCCESS, TestGraph(graph, centrality_score));
  EXPECT_EQ((score_t)0.0, centrality_score[0]);
}

// Tests BetwCentrality for a chain of 100 nodes.
TEST_P(BetweennessCentralityTest, Chain100Unweighted) {
  graph_initialize(DATA_FOLDER("chain_100_nodes_weight_directed.totem"), false,
                   &graph);
  centrality_score = (score_t*)mem_alloc(graph->vertex_count * 
                                          sizeof(score_t));

  // First vertex as source
  EXPECT_EQ(SUCCESS, TestGraph(graph, centrality_score));
  score_t centrality[50];
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
  centrality_score = (score_t*)mem_alloc(graph->vertex_count * 
                                          sizeof(score_t));

  EXPECT_EQ(SUCCESS, TestGraph(graph, centrality_score));
  for(vid_t vertex = 0; vertex < graph->vertex_count; vertex++){
    EXPECT_FLOAT_EQ(0.0, centrality_score[0]);
  }
}

// Functions to test in framework
betweenness_param_t betweenness_params[] = {
  {false, &betweenness_unweighted_cpu},
  {false, &betweenness_unweighted_gpu},
  {false, &betweenness_unweighted_shi_gpu},
  {false, &betweenness_cpu_exact},
  {false, &betweenness_gpu_exact},
  {true, NULL}}; // Null entry corresponds to hybrid betweenness centrality

// From Google documentation:
// In order to run value-parameterized tests, we need to instantiate them,
// or bind them to a list of values which will be used as test parameters.
//
// Values() receives a list of parameters and the framework will execute the
// whole set of tests BetweennessCentralityTest for each element of Values()
INSTANTIATE_TEST_CASE_P(BetwCentralityGPUAndCPUTest, BetweennessCentralityTest,
                        Values(&betweenness_params[0],
                               &betweenness_params[1],
                               &betweenness_params[2],
                               &betweenness_params[3],
                               &betweenness_params[4],
                               &betweenness_params[5]));
#else

// From Google documentation:
// Google Test may not support value-parameterized tests with some
// compilers. This dummy test keeps gtest_main linked in.
TEST_P(DummyTest, ValueParameterizedTestsAreNotSupportedOnThisPlatform) {}

#endif  // GTEST_HAS_PARAM_TEST
