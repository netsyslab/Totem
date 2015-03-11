/*
 * Contains unit tests for an implementation of the breadth-first search (CC)
 * graph search algorithm.
 *
 *  Created on: 2015-03-15
 *      Author: Abdullah Gharaibeh
 */

// totem includes
#include "totem_common_unittest.h"

#if GTEST_HAS_PARAM_TEST

using ::testing::TestWithParam;
using ::testing::ValuesIn;

// The following implementation relies on TestWithParam<CCFunction> to test
// the the different versions of CC. Details on how to use TestWithParam<T>
// can be found at:
// http://code.google.com/p/googletest/source/browse/trunk/samples/sample7_unittest.cc

typedef error_t(*CCFunction)(graph_t*, vid_t*);
typedef error_t(*CCHybridFunction)(vid_t*);

class CCTest : public TestWithParam<test_param_t*> {
 public:
  virtual void SetUp() {
    // Ensure the minimum CUDA architecture is supported
    CUDA_CHECK_VERSION();
    _cc_param = GetParam();
    _mem_type = TOTEM_MEM_HOST_PINNED;
    _graph = NULL;
    _label = NULL;
  }
  virtual void TearDown() {
    if (_graph) graph_finalize(_graph);
  }

  error_t TestGraph() {
    if (_cc_param->attr) {
      _cc_param->attr->push_msg_size = sizeof(vid_t) * BITS_PER_BYTE + 1;
      _cc_param->attr->alloc_func = _cc_param->hybrid_alloc;
      _cc_param->attr->free_func = _cc_param->hybrid_free;
      if (totem_init(_graph, _cc_param->attr) == FAILURE) {
        return FAILURE;
      }
      CCHybridFunction func =
          reinterpret_cast<CCHybridFunction>(_cc_param->func);
      error_t err = func(_label);
      totem_finalize();
      return err;
    }
    CCFunction func = reinterpret_cast<CCFunction>(_cc_param->func);
    return func(_graph, _label);
  }

 protected:
  test_param_t* _cc_param;
  totem_mem_t   _mem_type;
  graph_t*      _graph;
  vid_t*       _label;
};

// Tests CC for empty graphs.
TEST_P(CCTest, Empty) {
  _graph = reinterpret_cast<graph_t*>(calloc(1, sizeof(graph_t)));
  EXPECT_EQ(FAILURE, TestGraph());
  free(_graph);
  _graph = NULL;
}

// Tests CC for single node graphs.
TEST_P(CCTest, SingleNode) {
  graph_initialize(DATA_FOLDER("single_node.totem"), false, &_graph);
  CALL_SAFE(totem_malloc(_graph->vertex_count * sizeof(vid_t), _mem_type,
                         reinterpret_cast<void**>(&_label)));
  EXPECT_EQ(SUCCESS, TestGraph());
  EXPECT_EQ((vid_t)0, _label[0]);
}

TEST_P(CCTest, SingleNodeLoop) {
  graph_initialize(DATA_FOLDER("single_node_loop.totem"), false, &_graph);
  CALL_SAFE(totem_malloc(_graph->vertex_count * sizeof(vid_t), _mem_type,
                         reinterpret_cast<void**>(&_label)));
  EXPECT_EQ(SUCCESS, TestGraph());
  EXPECT_EQ((vid_t)0, _label[0]);
}

// Tests CC for graphs with node and no edges.
TEST_P(CCTest, EmptyEdges) {
  graph_initialize(DATA_FOLDER("disconnected_1000_nodes.totem"), false,
                   &_graph);
  CALL_SAFE(totem_malloc(_graph->vertex_count * sizeof(vid_t), _mem_type,
                         reinterpret_cast<void**>(&_label)));
  EXPECT_EQ(SUCCESS, TestGraph());
  EXPECT_EQ((vid_t)0, _label[0]);
  for (vid_t vertex = 0; vertex < _graph->vertex_count; vertex++) {
    EXPECT_EQ(vertex, _label[vertex]);
  }
}

// Tests CC for a chain of 1000 nodes.
TEST_P(CCTest, Chain) {
  graph_initialize(DATA_FOLDER("chain_1000_nodes.totem"), false, &_graph);
  CALL_SAFE(totem_malloc(_graph->vertex_count * sizeof(vid_t), _mem_type,
                         reinterpret_cast<void**>(&_label)));
  EXPECT_EQ(SUCCESS, TestGraph());
  for (vid_t vertex = 0; vertex < _graph->vertex_count; vertex++) {
    EXPECT_EQ(0, _label[vertex]);
  }
}

// Tests CC for a chain of 1000 nodes.
TEST_P(CCTest, MultiChain) {
  graph_initialize(DATA_FOLDER("chain_4_comp_40_nodes.totem"), false, &_graph);
  CALL_SAFE(totem_malloc(_graph->vertex_count * sizeof(vid_t), _mem_type,
                         reinterpret_cast<void**>(&_label)));
  EXPECT_EQ(SUCCESS, TestGraph());

  const int last_vertex_in_first_comp = 9;
  for (vid_t vertex = 0; vertex <= last_vertex_in_first_comp; vertex++) {
    EXPECT_EQ(0, _label[vertex]);
  }

  const int last_vertex_in_second_comp = 19;
  for (vid_t vertex = last_vertex_in_first_comp + 1;
       vertex <= last_vertex_in_second_comp; vertex++) {
    EXPECT_EQ(last_vertex_in_first_comp + 1, _label[vertex]);
  }

  const int last_vertex_in_third_comp = 30;
  for (vid_t vertex = last_vertex_in_second_comp + 1;
       vertex <= last_vertex_in_third_comp; vertex++) {
    EXPECT_EQ(last_vertex_in_second_comp + 1, _label[vertex]);
  }

  const int last_vertex_in_fourth_comp = _graph->vertex_count - 1;
  for (vid_t vertex = last_vertex_in_third_comp + 1;
       vertex <= last_vertex_in_fourth_comp; vertex++) {
    EXPECT_EQ(last_vertex_in_third_comp + 1, _label[vertex]);
  }
}

// Tests CC for a complete graph of 300 nodes.
TEST_P(CCTest, CompleteGraph) {
  graph_initialize(DATA_FOLDER("complete_graph_300_nodes.totem"), false,
                   &_graph);
  CALL_SAFE(totem_malloc(_graph->vertex_count * sizeof(vid_t), _mem_type,
                         reinterpret_cast<void**>(&_label)));
  EXPECT_EQ(SUCCESS, TestGraph());
  for (vid_t vertex = 0; vertex < _graph->vertex_count; vertex++) {
    EXPECT_EQ(0, _label[vertex]);
  }
}

// Tests CC for a complete graph of 1000 nodes.
TEST_P(CCTest, Star) {
  graph_initialize(DATA_FOLDER("star_1000_nodes.totem"), false, &_graph);
  CALL_SAFE(totem_malloc(_graph->vertex_count * sizeof(vid_t), _mem_type,
                         reinterpret_cast<void**>(&_label)));

  EXPECT_EQ(SUCCESS, TestGraph());
  for (vid_t vertex = 0; vertex < _graph->vertex_count; vertex++) {
    EXPECT_EQ((vid_t)0, _label[vertex]);
  }
}

// Defines the set of CC vanilla implementations to be tested. To test
// a new implementation, simply add it to the set below.
void* cc_vanilla_funcs[] = {
};
const int cc_vanilla_count = STATIC_ARRAY_COUNT(cc_vanilla_funcs);

// Defines the set of CC hybrid implementations to be tested. To test
// a new implementation, simply add it to the set below.
void* cc_hybrid_funcs[] = {
  reinterpret_cast<void*>(&cc_hybrid),
};
totem_cb_func_t cc_hybrid_alloc_funcs[] = {
  NULL,
};
totem_cb_func_t cc_hybrid_free_funcs[] = {
  NULL,
};
const int cc_hybrid_count = STATIC_ARRAY_COUNT(cc_hybrid_funcs);

// Maintains references to the different configurations (vanilla and hybrid)
// that will be tested by the framework.
static const int cc_params_count = cc_vanilla_count +
    cc_hybrid_count * hybrid_configurations_count;
static test_param_t* cc_params[cc_params_count];

// From Google documentation:
// In order to run value-parameterized tests, we need to instantiate them,
// or bind them to a list of values which will be used as test parameters.
//
// Values() receives a list of parameters and the framework will execute the
// whole set of tests CCTest for each element of Values()
INSTANTIATE_TEST_CASE_P(CCGPUAndCPUTest, CCTest,
                        ValuesIn(GetParameters(
                            cc_params, cc_params_count,
                            cc_vanilla_funcs, cc_vanilla_count,
                            cc_hybrid_funcs, cc_hybrid_count,
                            cc_hybrid_alloc_funcs, cc_hybrid_free_funcs),
                                 cc_params + cc_params_count));
#else

// From Google documentation:
// Google Test may not support value-parameterized tests with some
// compilers. This dummy test keeps gtest_main linked in.
TEST_P(DummyTest, ValueParameterizedTestsAreNotSupportedOnThisPlatform) {}

#endif  // GTEST_HAS_PARAM_TEST
