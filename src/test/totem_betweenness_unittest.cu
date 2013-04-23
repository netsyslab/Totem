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
  totem_attr_t* attr;          // totem attributes for totem-based tests
  BetwCentralityFunction func; // the vanilla BC function if attr is NULL
} betweenness_param_t;

class BetweennessCentralityTest : public TestWithParam<betweenness_param_t*> {
 public:
  virtual void SetUp() {
    // Ensure the minimum CUDA architecture is supported
    CUDA_CHECK_VERSION();
    _betweenness_param = GetParam();
    _graph = NULL;
    _betweenness_score = NULL;
  }

  virtual void TearDown() {
    if (_graph) graph_finalize(_graph);
    if (_betweenness_score) {
      totem_free(_betweenness_score, TOTEM_MEM_HOST);
    }
  }

  error_t TestGraph() {
    if (_graph->vertex_count) {
      totem_malloc(_graph->vertex_count * sizeof(score_t), 
                   TOTEM_MEM_HOST, (void**)&_betweenness_score);
    }
    if (_betweenness_param->attr != NULL) {
      _betweenness_param->attr->push_msg_size = 
        sizeof(uint32_t) * BITS_PER_BYTE;
      _betweenness_param->attr->pull_msg_size =
        sizeof(score_t) * BITS_PER_BYTE;
      if (totem_init(_graph, _betweenness_param->attr) == FAILURE) {
        return FAILURE;
      }

      // Will use exact betweenness centrality for test framework
      error_t err = betweenness_hybrid(CENTRALITY_EXACT, _betweenness_score);
      totem_finalize();
      return err;
    }
    return _betweenness_param->func(_graph, _betweenness_score);
  }

 protected:
   betweenness_param_t* _betweenness_param;
   graph_t* _graph;
   score_t* _betweenness_score;
};

// Tests BetwCentrality for empty graphs.
TEST_P(BetweennessCentralityTest, Empty) {
  _graph = (graph_t*)calloc(sizeof(graph_t), 1);
  EXPECT_EQ(FAILURE, TestGraph());
  free(_graph);
  _graph = NULL;
}

// Tests BetwCentrality for single node graphs.
TEST_P(BetweennessCentralityTest, SingleNodeUnweighted) {
  EXPECT_EQ(SUCCESS, graph_initialize(DATA_FOLDER("single_node.totem"),
                                      false, &_graph));
  EXPECT_EQ(SUCCESS, TestGraph());
  EXPECT_EQ((score_t)0.0, _betweenness_score[0]);
}

// Tests BetwCentrality for a chain of 100 nodes.
TEST_P(BetweennessCentralityTest, Chain100Unweighted) {
  graph_initialize(DATA_FOLDER("chain_100_nodes_weight_directed.totem"), false,
                   &_graph);
  // First vertex as source
  EXPECT_EQ(SUCCESS, TestGraph());
  score_t centrality[50];
  for (vid_t i = 0; i < 50; i++) {
    centrality[i] = (99 - i) * (i);
  }
  for (vid_t i = 0; i < 50; i++) {
    EXPECT_EQ(centrality[i], _betweenness_score[i]);
    EXPECT_EQ(centrality[i], _betweenness_score[99 - i]);
  }
}

// Tests BetwCentrality for a complete graph of 300 nodes.
TEST_P(BetweennessCentralityTest, CompleteGraphUnweighted) {
  EXPECT_EQ(SUCCESS,
            graph_initialize(DATA_FOLDER("complete_graph_300_nodes.totem"),
                             false, &_graph));

  EXPECT_EQ(SUCCESS, TestGraph());
  for(vid_t vertex = 0; vertex < _graph->vertex_count; vertex++){
    EXPECT_FLOAT_EQ(0.0, _betweenness_score[vertex]);
  }
}

// Tests BetwCentrality for a star graph.
TEST_P(BetweennessCentralityTest, StarGraphUnweighted) {
  EXPECT_EQ(SUCCESS, graph_initialize(DATA_FOLDER("star_1000_nodes.totem"),
                                      false, &_graph));

  EXPECT_EQ(SUCCESS, TestGraph());
  EXPECT_FLOAT_EQ((_graph->vertex_count - 1) * (_graph->vertex_count - 2), 
                  _betweenness_score[0]);
  for(vid_t vertex = 1; vertex < _graph->vertex_count; vertex++){
    EXPECT_FLOAT_EQ(0.0, _betweenness_score[vertex]);
  }
}

// Functions to test in framework
betweenness_param_t betweenness_params[] = {
  {NULL, &betweenness_unweighted_cpu},
  {NULL, &betweenness_unweighted_gpu},
  {NULL, &betweenness_unweighted_shi_gpu},
  {NULL, &betweenness_cpu_exact},
  {NULL, &betweenness_gpu_exact},
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
// whole set of tests BetweennessCentralityTest for each element of Values()
INSTANTIATE_TEST_CASE_P(BetwCentralityGPUAndCPUTest, BetweennessCentralityTest,
                        Values(&betweenness_params[0],
                               &betweenness_params[1],
                               &betweenness_params[2],
                               &betweenness_params[3],
                               &betweenness_params[4],
                               &betweenness_params[5],
                               &betweenness_params[6],
                               &betweenness_params[7],
                               &betweenness_params[8],
                               &betweenness_params[9],
                               &betweenness_params[10],
                               &betweenness_params[11]));
#else

// From Google documentation:
// Google Test may not support value-parameterized tests with some
// compilers. This dummy test keeps gtest_main linked in.
TEST_P(DummyTest, ValueParameterizedTestsAreNotSupportedOnThisPlatform) {}

#endif  // GTEST_HAS_PARAM_TEST
