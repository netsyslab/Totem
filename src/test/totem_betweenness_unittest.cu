/*
 * Contains unit tests for betweenness centrality.
 *
 *  Created on: 2011-10-21
 *      Author: Greg Redekop
 */

// totem includes
#include "totem_common_unittest.h"

#if GTEST_HAS_PARAM_TEST

using ::testing::TestWithParam;
using ::testing::ValuesIn;

typedef error_t(*BetwCentralityFunction)(const graph_t*, double, score_t*);
typedef error_t(*BetwCentralityHybridFunction)(double, score_t*);

class BetweennessCentralityTest : public TestWithParam<test_param_t*> {
 public:
  virtual void SetUp() {
    // Ensure the minimum CUDA architecture is supported.
    CUDA_CHECK_VERSION();
    _param = GetParam();
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
                   TOTEM_MEM_HOST,
                   reinterpret_cast<void**>(&_betweenness_score));
    }
    if (_param->attr != NULL) {
      _param->attr->push_msg_size =
        sizeof(uint32_t) * BITS_PER_BYTE;
      _param->attr->pull_msg_size =
        sizeof(score_t) * BITS_PER_BYTE;
      if (totem_init(_graph, _param->attr) == FAILURE) {
        return FAILURE;
      }

      BetwCentralityHybridFunction func =
          reinterpret_cast<BetwCentralityHybridFunction>
          (_param->func);
      error_t err = func(CENTRALITY_EXACT, _betweenness_score);
      totem_finalize();
      return err;
    }
    BetwCentralityFunction func =
        reinterpret_cast<BetwCentralityFunction>(_param->func);
    return func(_graph, CENTRALITY_EXACT, _betweenness_score);
  }

 protected:
  test_param_t* _param;
  graph_t*      _graph;
  score_t*      _betweenness_score;
};

// Tests BetwCentrality for empty graphs.
TEST_P(BetweennessCentralityTest, Empty) {
  _graph = reinterpret_cast<graph_t*>(calloc(sizeof(graph_t), 1));
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
  // First vertex as source.
  EXPECT_EQ(SUCCESS, TestGraph());
  score_t* expected_centrality =
      reinterpret_cast<score_t*>(calloc(_graph->vertex_count / 2,
                                        sizeof(score_t)));
  for (vid_t i = 0; i < (_graph->vertex_count / 2); i++) {
    expected_centrality[i] = (_graph->vertex_count - 1 - i) * (i);
  }
  for (vid_t i = 0; i < (_graph->vertex_count / 2); i++) {
    EXPECT_EQ(expected_centrality[i], _betweenness_score[i]);
    EXPECT_EQ(expected_centrality[i],
              _betweenness_score[_graph->vertex_count - 1 - i]);
  }
  free(expected_centrality);
}

// Tests BetwCentrality for a complete graph of 300 nodes.
TEST_P(BetweennessCentralityTest, CompleteGraphUnweighted) {
  EXPECT_EQ(SUCCESS,
            graph_initialize(DATA_FOLDER("complete_graph_300_nodes.totem"),
                             false, &_graph));

  EXPECT_EQ(SUCCESS, TestGraph());
  for (vid_t vertex = 0; vertex < _graph->vertex_count; vertex++) {
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
  for (vid_t vertex = 1; vertex < _graph->vertex_count; vertex++) {
    EXPECT_FLOAT_EQ(0.0, _betweenness_score[vertex]);
  }
}

// Defines the set of Betweenness vanilla implementations to be tested. To test
// a new implementation, simply add it to the set below.
static void* vanilla_funcs[] = {
  reinterpret_cast<void*>(&betweenness_cpu),
  reinterpret_cast<void*>(&betweenness_gpu),
};
static const int vanilla_count = STATIC_ARRAY_COUNT(vanilla_funcs);

// Defines the set of PageRank hybrid implementations to be tested. To test
// a new implementation, simply add it to the set below.
static void* hybrid_funcs[] = {
  reinterpret_cast<void*>(&betweenness_hybrid),
};
static const int hybrid_count = STATIC_ARRAY_COUNT(hybrid_funcs);

// Maintains references to the different configurations (vanilla and hybrid)
// that will be tested by the framework.
static const int params_count = vanilla_count +
    hybrid_count * hybrid_configurations_count;
static test_param_t* params[params_count];

// From Google documentation:
// In order to run value-parameterized tests, we need to instantiate them,
// or bind them to a list of values which will be used as test parameters.
//
// ValuesIn() receives a list of parameters and the framework will execute the
// whole set of tests for each entry in the array passed to ValuesIn().
INSTANTIATE_TEST_CASE_P(BetweennessGPUAndCPUTest, BetweennessCentralityTest,
                        ValuesIn(GetParameters(params, params_count,
                                               vanilla_funcs, vanilla_count,
                                               hybrid_funcs, hybrid_count),
                                 params + params_count));
#else

// From Google documentation:
// Google Test may not support value-parameterized tests with some
// compilers. This dummy test keeps gtest_main linked in.
TEST_P(DummyTest, ValueParameterizedTestsAreNotSupportedOnThisPlatform) {}

#endif  // GTEST_HAS_PARAM_TEST
