/*
 * Unit tests for clustering coefficient algorithm.
 *
 * Created on: 2013-07-09
 * Authors: Sidney Pontes Filho
 *          Tahsin Arafat Reza
 *          Tanuj Kr Aasawat
 *
 */

// totem includes
#include "totem_common_unittest.h"

#if GTEST_HAS_PARAM_TEST

using ::testing::TestWithParam;
using ::testing::Values;

// The following implementation relies on
// TestWithParam<ClusteringCoefficientFunction> to test the two versions of 
// Clustering Coefficient implemented for: CPU and GPU.  Details on how to use 
// TestWithParam<T> can be found at:
// http:
// code.google.com/p/googletest/source/browse/trunk/samples/sample7_unittest.cc

typedef error_t(*ClusteringCoefficientFunction)(const graph_t*, weight_t**);

class ClusteringCoefficientTest : 
public TestWithParam<ClusteringCoefficientFunction> {
 public:
  virtual void SetUp() {
    // Ensure the minimum CUDA architecture is supported
    CUDA_CHECK_VERSION();
    clustering = GetParam();
    _graph = NULL; 
    _coefficients = NULL;
    _mem_type = TOTEM_MEM_HOST_PINNED;		
  }
  
  virtual void TearDown() {
    if(_graph) graph_finalize(_graph);
    if(_coefficients) totem_free(_coefficients, _mem_type);
  }

 protected:
   ClusteringCoefficientFunction clustering;
   graph_t* _graph;
   weight_t* _coefficients;
   totem_mem_t _mem_type;
};

// Tests ClusteringCoefficient for an empty graph.
TEST_P(ClusteringCoefficientTest, EmptyGraph) {
  graph_t graph;
  graph.directed = false;
  graph.vertex_count = 0;
  graph.edge_count = 0;
  EXPECT_EQ(FAILURE, clustering(&graph, &_coefficients));
}

// Tests ClusteringCoefficient for a single node graph.
TEST_P(ClusteringCoefficientTest, SingleNodeGrpah) {
  EXPECT_EQ(SUCCESS, graph_initialize(DATA_FOLDER("single_node.totem"),
                                      false, &_graph));
  
  CALL_SAFE(totem_malloc(_graph->vertex_count * sizeof(weight_t), _mem_type,
                         (void**)&_coefficients)); 
 
  EXPECT_EQ(SUCCESS, clustering(_graph, &_coefficients));
  EXPECT_FALSE(_coefficients == NULL);
  EXPECT_EQ((weight_t)0.0, _coefficients[0]);  
}

// Tests ClusteringCoefficient for an undirected complete graph with 
// 300 nodes.
TEST_P(ClusteringCoefficientTest, CompleteGraph300NodesUndirected) {
  EXPECT_EQ(SUCCESS,
            graph_initialize(DATA_FOLDER("complete_graph_300_nodes.totem"),
                             false, &_graph));
  
  CALL_SAFE(totem_malloc(_graph->vertex_count * sizeof(weight_t), _mem_type,
                         (void**)&_coefficients)); 
   
  EXPECT_EQ(SUCCESS, clustering(_graph, &_coefficients));
  EXPECT_FALSE(_coefficients == NULL);
  for(vid_t vertex = 0; vertex < _graph->vertex_count; vertex++){
    EXPECT_FLOAT_EQ(1.0, _coefficients[vertex]);
  }  
}

// Tests ClusteringCoefficient for an undirected grid graph with 15 nodes.
TEST_P(ClusteringCoefficientTest, GridGraph15NodesUndirected) {
  EXPECT_EQ(SUCCESS,
            graph_initialize(DATA_FOLDER("grid_graph_15_nodes_weight.totem"),
                             false, &_graph));

  CALL_SAFE(totem_malloc(_graph->vertex_count * sizeof(weight_t), _mem_type,
                         (void**)&_coefficients));

  EXPECT_EQ(SUCCESS, clustering(_graph, &_coefficients));
  EXPECT_FALSE(_coefficients == NULL);
  for(vid_t vertex = 0; vertex < _graph->vertex_count; vertex++){
    EXPECT_FLOAT_EQ(0.0, _coefficients[vertex]);
  }
}

// Tests ClusteringCoefficient for an undirected chain graph with 1000 nodes.
TEST_P(ClusteringCoefficientTest, ChainGraph1000NodesUndirected) {
  EXPECT_EQ(SUCCESS,
            graph_initialize(DATA_FOLDER("chain_1000_nodes.totem"),
                             false, &_graph));

  CALL_SAFE(totem_malloc(_graph->vertex_count * sizeof(weight_t), _mem_type, 
                         (void**)&_coefficients)); 
  
  EXPECT_EQ(SUCCESS, clustering(_graph, &_coefficients));
  EXPECT_FALSE(_coefficients == NULL);
  for(vid_t vertex = 0; vertex < _graph->vertex_count; vertex++){
    EXPECT_FLOAT_EQ(0.0, _coefficients[vertex]);
  }  
}

// Tests ClustreingCoefficinet for an undirected star graph with 1K nodes.
TEST_P(ClusteringCoefficientTest, StarGraph1000NodesUndirected) {
  EXPECT_EQ(SUCCESS,
            graph_initialize(DATA_FOLDER("star_1000_nodes.totem"),
                             false, &_graph));

  CALL_SAFE(totem_malloc(_graph->vertex_count * sizeof(weight_t), _mem_type, 
                         (void**)&_coefficients)); 
  
  EXPECT_EQ(SUCCESS, clustering(_graph, &_coefficients));
  EXPECT_FALSE(_coefficients == NULL);
  for(vid_t vertex = 0; vertex < _graph->vertex_count; vertex++){
    EXPECT_FLOAT_EQ(0.0, _coefficients[vertex]);
  }  
}

// Tests ClustreingCoefficinet for a graph with 1K disconnected nodes.
TEST_P(ClusteringCoefficientTest, DisconnectedGraph1000Nodes) {
  EXPECT_EQ(SUCCESS,
            graph_initialize(DATA_FOLDER("disconnected_1000_nodes.totem"),
                             false, &_graph));

  CALL_SAFE(totem_malloc(_graph->vertex_count * sizeof(weight_t), _mem_type, 
                         (void**)&_coefficients)); 
  
  EXPECT_EQ(SUCCESS, clustering(_graph, &_coefficients));
  EXPECT_FALSE(_coefficients == NULL);
  for(vid_t vertex = 0; vertex < _graph->vertex_count; vertex++){
    EXPECT_FLOAT_EQ(0.0, _coefficients[vertex]);
  }  
}

// Tests ClustreingCoefficinet for an undirected wheel graph with 1K 
// nodes. 
TEST_P(ClusteringCoefficientTest, WheelGraph1000NodesUndirected) {
  EXPECT_EQ(SUCCESS,
            graph_initialize(DATA_FOLDER("wheel_graph_1000_nodes.totem"),
                             false, &_graph));

  CALL_SAFE(totem_malloc(_graph->vertex_count * sizeof(weight_t), _mem_type, 
                         (void**)&_coefficients)); 
  
  EXPECT_EQ(SUCCESS, clustering(_graph, &_coefficients));
  EXPECT_FALSE(_coefficients == NULL);

  weight_t expected_cc_center_vertex = 
           2.0f / ((weight_t)_graph->vertex_count - 2.0f);
  EXPECT_FLOAT_EQ(expected_cc_center_vertex, _coefficients[0]);

  weight_t expected_cc_ring_vertex = 2.0f / 3.0f;
  for(vid_t vertex = 1; vertex < _graph->vertex_count; vertex++){
    EXPECT_FLOAT_EQ(expected_cc_ring_vertex, _coefficients[vertex]);
  }  
}

// From Google documentation:
// In order to run value-parameterized tests, we need to instantiate them,
// or bind them to a list of values which will be used as test parameters.
//
// Values() receives a list of parameters and the framework will execute the
// whole set of tests ClusteringCoefficientTest for each element of Values()
INSTANTIATE_TEST_CASE_P(ClusteringCoefficientGPUandCPUTest, 
                        ClusteringCoefficientTest, 
                        Values(&clustering_coefficient_cpu,
                        &clustering_coefficient_gpu,
                        &clustering_coefficient_sorted_neighbours_cpu,
                        &clustering_coefficient_sorted_neighbours_gpu));

#else

// From Google documentation:
// Google Test may not support value-parameterized tests with some
// compilers. This dummy test keeps gtest_main linked in.
TEST_P(DummyTest, ValueParameterizedTestsAreNotSupportedOnThisPlatform) {}

#endif  // GTEST_HAS_PARAM_TEST
