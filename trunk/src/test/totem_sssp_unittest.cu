/* 
 * Contains unit tests for an implementation of the single source shortest path
 * (SSSP) algorithm.
 *
 * Created on: 2011-03-24
 * Author: Elizeu Santos-Neto
 *         Tanuj Kr Aasawat
 *         Tahsin Reza
 */

// totem includes
#include "totem_common_unittest.h"

#if GTEST_HAS_PARAM_TEST

using ::testing::TestWithParam;
using ::testing::Values;

// The following implementation relies on TestWithParam<SSSPFunction> to test
// the two versions of SSSP implemented: CPU and GPU.
// Details on how to use TestWithParam<T> can be found at:
// http://code.google.com/p/googletest/source/browse/trunk/samples/
// sample7_unittest.cc

typedef error_t(*SSSPFunction)(graph_t*, vid_t, weight_t*);

// This is to allow testing the vanilla sssp functions and the hybrid one
// that is based on the framework. Note that have a different signature
// of the hybrid algorithm forced this work-around.
typedef struct sssp_param_s {
  totem_attr_t* attr;  // totem attributes for totem-based tests
  SSSPFunction  func;  // the vanilla sssp function if attr is NULL
} sssp_param_t;

class SSSPTest : public TestWithParam<sssp_param_t*> {
 public:
  virtual void SetUp() {
    // Ensure the minimum CUDA architecture is supported
    CUDA_CHECK_VERSION();
    _sssp_param = GetParam();
    _mem_type = TOTEM_MEM_HOST_PINNED;
    _graph = NULL;
    _distances = NULL;
  }
  virtual void TearDown() {
    if (_graph) graph_finalize(_graph);
    if (_distances) totem_free(_distances, _mem_type);
  }

  error_t TestGraph(vid_t src) {
    if (_sssp_param->attr) {
      _sssp_param->attr->push_msg_size = sizeof(weight_t) * BITS_PER_BYTE;
      if (totem_init(_graph, _sssp_param->attr) == FAILURE) {
        return FAILURE;
      }
      error_t err = sssp_hybrid(src, _distances);
      totem_finalize();
      return err;
    }
    return _sssp_param->func(_graph, src, _distances);
  }

 protected:
  sssp_param_t* _sssp_param;
  totem_mem_t _mem_type;
  graph_t* _graph;
  weight_t* _distances;
};

// Tests SSSP for empty graphs.
TEST_P(SSSPTest, Empty) {
  _graph = reinterpret_cast<graph_t*>(calloc(1, sizeof(graph_t)));
  EXPECT_EQ(FAILURE, TestGraph(0));
  EXPECT_EQ(FAILURE, TestGraph(99));
  free(_graph);
  _graph = NULL;
}

// Tests SSSP for single node graphs.
TEST_P(SSSPTest, SingleNode) {
  graph_initialize(DATA_FOLDER("single_node.totem"), true, &_graph);
  CALL_SAFE(totem_malloc(_graph->vertex_count * sizeof(weight_t), _mem_type,
                         reinterpret_cast<void**>(&_distances)));
  EXPECT_EQ(SUCCESS, TestGraph(0));
  EXPECT_EQ(0.0, _distances[0]);
  EXPECT_EQ(FAILURE, TestGraph(1));
}

TEST_P(SSSPTest, SingleNodeLoop) {
  graph_initialize(DATA_FOLDER("single_node_loop_weight.totem"), true,
    &_graph);
  CALL_SAFE(totem_malloc(_graph->vertex_count * sizeof(weight_t), _mem_type,
                         reinterpret_cast<void**>(&_distances)));
  EXPECT_EQ(SUCCESS, TestGraph(0));
  EXPECT_EQ(0.0, _distances[0]);
  EXPECT_EQ(FAILURE, TestGraph(1));
}

// Tests SSSP for graphs with node and no edges.
TEST_P(SSSPTest, EmptyEdges) {
  graph_initialize(DATA_FOLDER("disconnected_1000_nodes.totem"), true,
                   &_graph);
  CALL_SAFE(totem_malloc(_graph->vertex_count * sizeof(weight_t), _mem_type,
                         reinterpret_cast<void**>(&_distances)));

  // First vertex as source
  vid_t source = 0;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  EXPECT_EQ(0.0, _distances[source]);
  for (vid_t vertex = source + 1; vertex < _graph->vertex_count; vertex++) {
    EXPECT_EQ(WEIGHT_MAX, _distances[vertex]);
  }

  // Last vertex as source
  source = _graph->vertex_count - 1;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  EXPECT_EQ(0.0, _distances[source]);
  for (vid_t vertex = source; vertex < _graph->vertex_count - 1; vertex++) {
    EXPECT_EQ(WEIGHT_MAX, _distances[vertex]);
  }

  // A vertex in the middle as source
  source = 199;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  for (vid_t vertex = 0; vertex < _graph->vertex_count; vertex++) {
    EXPECT_EQ((vertex == source) ? 0.0 : WEIGHT_MAX, _distances[vertex]);
  }

  // Non existent vertex source
  EXPECT_EQ(FAILURE, TestGraph(_graph->vertex_count));
}

// Tests SSSP for a chain of 1000 nodes.
TEST_P(SSSPTest, Chain) {
  graph_initialize(DATA_FOLDER("chain_1000_nodes_weight.totem"), true,
    &_graph);
  CALL_SAFE(totem_malloc(_graph->vertex_count * sizeof(weight_t), _mem_type,
                         reinterpret_cast<void**>(&_distances)));

  // First vertex as source
  vid_t source = 0;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  for (vid_t vertex = source; vertex < _graph->vertex_count; vertex++) {
    EXPECT_EQ((weight_t)vertex, _distances[vertex]);
  }

  // Last vertex as source
  source = _graph->vertex_count - 1;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  for (vid_t vertex = 0; vertex < _graph->vertex_count; vertex++) {
    EXPECT_EQ((weight_t)(source - vertex), _distances[vertex]);
  }

  // A vertex in the middle as source
  source = 199;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  for (vid_t vertex = 0; vertex < _graph->vertex_count; vertex++) {
    EXPECT_EQ((weight_t)abs((weight_t)source - (weight_t)vertex),
      _distances[vertex]);
  }

  // Non existent vertex source
  EXPECT_EQ(FAILURE, TestGraph(_graph->vertex_count));
}

// Tests SSSP for a complete graph of 300 nodes.
TEST_P(SSSPTest, CompleteGraph) {
  graph_initialize(DATA_FOLDER("complete_graph_300_nodes_weight.totem"), true,
                   &_graph);
  CALL_SAFE(totem_malloc(_graph->vertex_count * sizeof(weight_t), _mem_type,
                         reinterpret_cast<void**>(&_distances)));

  // First vertex as source
  vid_t source = 0;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  EXPECT_EQ(0.0, _distances[source]);
  for (vid_t vertex = 1; vertex < _graph->vertex_count; vertex++) {
    EXPECT_EQ(1.0, _distances[vertex]);
  }

  // Last vertex as source
  source = _graph->vertex_count - 1;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  EXPECT_EQ(0.0, _distances[source]);
  for (vid_t vertex = 0; vertex < source; vertex++) {
    EXPECT_EQ(1.0, _distances[vertex]);
  }

  // A vertex source in the middle
  source = 199;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  for (vid_t vertex = 0; vertex < _graph->vertex_count; vertex++) {
    EXPECT_EQ(((vertex == source) ? 0.0 : 1.0), _distances[vertex]);
  }

  // Non existent vertex source
  EXPECT_EQ(FAILURE, TestGraph(_graph->vertex_count));
}

// Tests SSSP for a star graph of 1000 nodes.
TEST_P(SSSPTest, Star) {
  graph_initialize(DATA_FOLDER("star_1000_nodes_weight.totem"), true, &_graph);
  CALL_SAFE(totem_malloc(_graph->vertex_count * sizeof(weight_t), _mem_type,
                         reinterpret_cast<void**>(&_distances)));

  // First vertex as source
  vid_t source = 0;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  EXPECT_EQ(0.0, _distances[source]);
  for (vid_t vertex = 1; vertex < _graph->vertex_count; vertex++) {
    EXPECT_EQ(1.0, _distances[vertex]);
  }

  // Last vertex as source
  source = _graph->vertex_count - 1;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  EXPECT_EQ(0.0, _distances[source]);
  EXPECT_EQ(1.0, _distances[0]);
  for (vid_t vertex = 1; vertex < source - 1; vertex++) {
    EXPECT_EQ(2.0, _distances[vertex]);
  }

  // A vertex source in the middle
  source = 199;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  EXPECT_EQ(1.0, _distances[0]);
  for (vid_t vertex = 1; vertex < _graph->vertex_count; vertex++) {
    EXPECT_EQ(((vertex == source) ? 0.0 : 2.0), _distances[vertex]);
  }

  // Non existent vertex source
  EXPECT_EQ(FAILURE, TestGraph(_graph->vertex_count));
}

// Tests SSSP for a grid graph with 15  nodes.
TEST_P(SSSPTest, Grid) {
  graph_initialize(DATA_FOLDER("grid_graph_sssp_15_nodes_weight.totem"), true,
    &_graph);
  CALL_SAFE(totem_malloc(_graph->vertex_count * sizeof(weight_t), _mem_type,
                         reinterpret_cast<void**>(&_distances)));

  // First vertex as source
  vid_t source = 0;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  EXPECT_EQ((weight_t)0.0, _distances[source]);
  for (vid_t vertex = 1; vertex < _graph->vertex_count; vertex++) {
    EXPECT_EQ((weight_t)(vertex - source), _distances[vertex]);
  }

  // Last vertex as source
  source = _graph->vertex_count - 1;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  EXPECT_EQ((weight_t)0.0, _distances[source]);
  for (vid_t vertex = 1; vertex < source - 1; vertex++) {
    EXPECT_EQ((weight_t)(source - vertex), _distances[vertex]);
  }

  // Non existent vertex source
  EXPECT_EQ(FAILURE, TestGraph(_graph->vertex_count));
}

// Tests SSSP algorithm a star graph with 1K nodes with different edge
// weights.
TEST_P(SSSPTest, StarDiffWeight) {
  graph_initialize(DATA_FOLDER("star_1000_nodes_diff_weight.totem"), true,
                   &_graph);
  CALL_SAFE(totem_malloc(_graph->vertex_count * sizeof(weight_t), _mem_type,
                         reinterpret_cast<void**>(&_distances)));

  // First vertex as source
  vid_t source = 0;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  EXPECT_EQ(0, _distances[0]);
  for (vid_t vertex_id = 1; vertex_id < _graph->vertex_count; vertex_id++) {
    EXPECT_EQ(1, _distances[vertex_id]);
  }

  // Last vertex as source
  source = _graph->vertex_count - 1;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  EXPECT_EQ(0, _distances[source]);
  EXPECT_EQ(source + 1, _distances[0]);
  for (vid_t vertex_id = 1; vertex_id < _graph->vertex_count - 1;
    vertex_id++) {
    // out edge weight = vertex_id + 1
    EXPECT_EQ(source + 2, _distances[vertex_id]);
  }

  // Non existent vertex source
  EXPECT_EQ(FAILURE, TestGraph(_graph->vertex_count));
}

// Tests SSSP algorithm a complete graph with 300 nodes, different edge
// weights.
TEST_P(SSSPTest, CompleteDiffWeight) {
  graph_initialize(DATA_FOLDER("complete_graph_300_nodes_diff_weight.totem"),
    true, &_graph);
  CALL_SAFE(totem_malloc(_graph->vertex_count * sizeof(weight_t), _mem_type,
                         reinterpret_cast<void**>(&_distances)));

  // First vertex as source
  vid_t source = 0;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  EXPECT_EQ(0, _distances[0]);
  for (vid_t vertex_id = 1; vertex_id < _graph->vertex_count; vertex_id++) {
    EXPECT_EQ(1, _distances[vertex_id]);
  }

  // Last vertex as source
  source = _graph->vertex_count - 1;
  EXPECT_EQ(SUCCESS, TestGraph(source));
  EXPECT_EQ(0, _distances[source]);
  for (vid_t vertex_id = 0; vertex_id < _graph->vertex_count - 1; vertex_id++) {
    // out edge from any node has weight = vertex_id + 1
    EXPECT_EQ(source + 1, _distances[vertex_id]);
  }

  // Non existent vertex source
  EXPECT_EQ(FAILURE, TestGraph(_graph->vertex_count));
}

// Values() seems to accept only pointers, hence the possible parameters
// are defined here, and a pointer to each ot them is used.
sssp_param_t sssp_params[] = {
  {&totem_attrs[0], NULL},
  {&totem_attrs[1], NULL},
  {&totem_attrs[2], NULL},
  {&totem_attrs[3], NULL},
  {&totem_attrs[4], NULL},
  {&totem_attrs[5], NULL},
  {&totem_attrs[6], NULL},
  {&totem_attrs[7], NULL},
  {&totem_attrs[8], NULL},
  {&totem_attrs[9], NULL},
  {&totem_attrs[10], NULL},
  {&totem_attrs[11], NULL},
  {&totem_attrs[12], NULL},
  {&totem_attrs[13], NULL},
  {&totem_attrs[14], NULL},
  {&totem_attrs[15], NULL},
  {&totem_attrs[16], NULL},
  {&totem_attrs[17], NULL},
  {&totem_attrs[18], NULL},
  {&totem_attrs[19], NULL},
  {&totem_attrs[20], NULL},
  {&totem_attrs[21], NULL},
  {&totem_attrs[22], NULL},
  {&totem_attrs[23], NULL}
};

// From Google documentation:
// In order to run value-parameterized tests, we need to instantiate them,
// or bind them to a list of values which will be used as test parameters.
//
// Values() receives a list of parameters and the framework will execute the
// whole set of tests SSSPTest for each element of Values()
INSTANTIATE_TEST_CASE_P(SSSPGPUAndCPUTest, SSSPTest, Values(&sssp_params[0],
                                                            &sssp_params[1],
                                                            &sssp_params[2],
                                                            &sssp_params[3],
                                                            &sssp_params[4],
                                                            &sssp_params[5],
                                                            &sssp_params[6],
                                                            &sssp_params[7],
                                                            &sssp_params[8],
                                                            &sssp_params[9],
                                                            &sssp_params[10],
                                                            &sssp_params[11],
                                                            &sssp_params[12],
                                                            &sssp_params[13],
                                                            &sssp_params[14],
                                                            &sssp_params[15],
                                                            &sssp_params[16],
                                                            &sssp_params[17],
                                                            &sssp_params[18],
                                                            &sssp_params[19],
                                                            &sssp_params[20],
                                                            &sssp_params[21],
                                                            &sssp_params[22],
                                                            &sssp_params[23]));

#else

// From Google documentation:
// Google Test may not support value-parameterized tests with some
// compilers. This dummy test keeps gtest_main linked in.
TEST_P(DummyTest, ValueParameterizedTestsAreNotSupportedOnThisPlatform) {}

#endif  // GTEST_HAS_PARAM_TEST
