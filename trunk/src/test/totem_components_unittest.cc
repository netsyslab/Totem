/* TODO(lauro,abdullah,elizeu): Add license.
 *
 * Contains unit tests for an implementation of the weakly connected components
 * identification algorithm algorithm.
 *
 *  Created on: 2011-11-23
 *      Author: Abdullah Gharaibeh
 */

// totem includes
#include "totem_common_unittest.h"

#if GTEST_HAS_PARAM_TEST

using ::testing::TestWithParam;
using ::testing::Values;

// The following implementation relies on 
// TestWithParam<GetComponentsFunction> to test the two versions of 
// GetComponents implemented: CPU and GPU. Details on how to use 
// TestWithParam<T> can be found at:
// http://code.google.com/p/googletest/source/browse/trunk/samples/sample7_unittest.cc

typedef error_t(*GetComponentsFunction)(graph_t*, component_set_t**);

class GetComponentsTest : public TestWithParam<GetComponentsFunction> {
 public:
  virtual void SetUp() {
    // Ensure the minimum CUDA architecture is supported
    CUDA_CHECK_VERSION();
    get_components = GetParam();
    graph = NULL;
    comp_set = NULL;
  }

  virtual void TearDown() {
    if (graph) graph_finalize(graph);
    if (comp_set) finalize_component_set(comp_set);
  }
 protected:
   GetComponentsFunction get_components;
   graph_t* graph;
   component_set_t* comp_set;
};

// Tests GetComponents for empty graphs.
TEST_P(GetComponentsTest, Empty) {
  graph_t empty_graph;
  empty_graph.directed = false;
  empty_graph.vertex_count = 0;
  empty_graph.edge_count = 0;
  EXPECT_EQ(FAILURE, get_components(&empty_graph, &comp_set));
}

// Tests GetComponents for single node graphs.
TEST_P(GetComponentsTest, SingleNode) {
  graph_initialize(DATA_FOLDER("single_node.totem"), false, &graph);
  EXPECT_EQ(SUCCESS, get_components(graph, &comp_set));
  EXPECT_FALSE(NULL == comp_set);
  EXPECT_EQ((uint32_t)1, comp_set->count);
  EXPECT_EQ((uint32_t)1, comp_set->vertex_count[0]);
  EXPECT_EQ((uint32_t)0, comp_set->edge_count[0]);
  EXPECT_EQ((uint32_t)0, comp_set->marker[0]);
  EXPECT_EQ((uint32_t)0, comp_set->biggest);
}

TEST_P(GetComponentsTest, SingleNodeLoop) {
  graph_initialize(DATA_FOLDER("single_node_loop.totem"), false, &graph);
  EXPECT_EQ(SUCCESS, get_components(graph, &comp_set));
  EXPECT_FALSE(NULL == comp_set);
  EXPECT_EQ((uint32_t)1, comp_set->count);
  EXPECT_EQ((uint32_t)1, comp_set->vertex_count[0]);
  EXPECT_EQ((uint32_t)1, comp_set->edge_count[0]);
  EXPECT_EQ((uint32_t)0, comp_set->marker[0]);
  EXPECT_EQ((uint32_t)0, comp_set->biggest);
}

// Tests GetComponents for graphs with nodes and no edges.
TEST_P(GetComponentsTest, EmptyEdges) {
  graph_initialize(DATA_FOLDER("disconnected_1000_nodes.totem"), false, &graph);
  EXPECT_EQ(SUCCESS, get_components(graph, &comp_set));
  EXPECT_FALSE(NULL == comp_set);
  EXPECT_EQ((uint32_t)1000, comp_set->count);
  for (id_t comp = 0; comp < comp_set->count; comp++) {
    EXPECT_EQ((uint32_t)1, comp_set->vertex_count[0]);
    EXPECT_EQ((uint32_t)0, comp_set->edge_count[0]);
  }
  for (id_t vid = 0; vid < graph->vertex_count; vid++) {
    // each vertex will be a componenet on its own
    EXPECT_EQ((uint32_t)vid, comp_set->marker[vid]);
  }
  // All components are of the same size. The biggest component is the one with 
  // the lowest id
  EXPECT_EQ(comp_set->biggest, (id_t)0);
}

// Tests GetComponents for a chain of 1000 nodes.
TEST_P(GetComponentsTest, Chain) {
  graph_initialize(DATA_FOLDER("chain_1000_nodes.totem"), false, &graph);
  EXPECT_EQ(SUCCESS, get_components(graph, &comp_set));
  EXPECT_FALSE(NULL == comp_set);
  EXPECT_EQ(comp_set->count, (id_t)1);
  EXPECT_EQ(comp_set->vertex_count[0], graph->vertex_count);
  EXPECT_EQ(comp_set->edge_count[0], graph->edge_count);  
  for (id_t vid = 0; vid < graph->vertex_count; vid++) {
    EXPECT_EQ(comp_set->marker[vid], (id_t)0);
  }
  EXPECT_EQ(comp_set->biggest, (id_t)0);
}

// Tests GetComponents for a graph of four chains. The first two have 10 
// vertices the thrid has 11 while the last has 9.
TEST_P(GetComponentsTest, MultiChain) {
  graph_initialize(DATA_FOLDER("chain_4_comp_40_nodes.totem"), false, &graph);
  EXPECT_EQ(SUCCESS, get_components(graph, &comp_set));
  EXPECT_FALSE(NULL == comp_set);
  EXPECT_EQ(comp_set->count, (id_t)4);
  for (id_t comp = 0; comp < comp_set->count; comp++) {
    id_t vertex_count = comp == comp_set->count - 1 ? 
      9 : comp == comp_set->count -2 ? 11 : 10;
    EXPECT_EQ(comp_set->vertex_count[comp], vertex_count);
    EXPECT_EQ(comp_set->edge_count[comp], vertex_count * 2 - 2);
  }
  
  for (id_t vid = 0; vid < graph->vertex_count; vid++) {
    // the second to last chain is one vertex longer
    id_t comp = vid == 30 ? (comp_set->count - 2) : vid / 10; 
    EXPECT_EQ(comp_set->marker[vid], comp);
  }
  // the biggest component is the second to last one
  EXPECT_EQ(comp_set->biggest, comp_set->count - 2);
}

// Tests GetComponents for a complete graph of 300 nodes.
TEST_P(GetComponentsTest, CompleteGraph) {
  graph_initialize(DATA_FOLDER("complete_graph_300_nodes.totem"), false,
                   &graph);
  EXPECT_EQ(SUCCESS, get_components(graph, &comp_set));
  EXPECT_FALSE(NULL == comp_set);
  EXPECT_EQ(comp_set->count, (id_t)1);
  EXPECT_EQ(comp_set->vertex_count[0], graph->vertex_count);
  EXPECT_EQ(comp_set->edge_count[0], graph->edge_count);
  
  for (id_t vid = 0; vid < graph->vertex_count; vid++) {
    EXPECT_EQ(comp_set->marker[vid], (id_t)0);
  }
  EXPECT_EQ(comp_set->biggest, (id_t)0);
}

// Tests GetComponents for a complete graph of 1000 nodes.
TEST_P(GetComponentsTest, Star) {
  graph_initialize(DATA_FOLDER("star_1000_nodes.totem"), false, &graph);
  EXPECT_EQ(SUCCESS, get_components(graph, &comp_set));
  EXPECT_FALSE(NULL == comp_set);
  EXPECT_EQ(comp_set->count, (id_t)1);
  EXPECT_EQ(comp_set->vertex_count[0], graph->vertex_count);
  EXPECT_EQ(comp_set->edge_count[0], graph->edge_count);
  
  for (id_t vid = 0; vid < graph->vertex_count; vid++) {
    EXPECT_EQ(comp_set->marker[vid], (id_t)0);
  }
  EXPECT_EQ(comp_set->biggest, (id_t)0);
}

// TODO(lauro): Add test cases for not well defined structures.

// From Google documentation:
// In order to run value-parameterized tests, we need to instantiate them,
// or bind them to a list of values which will be used as test parameters.
//
// Values() receives a list of parameters and the framework will execute the
// whole set of tests GetComponentsTest for each element of Values()
INSTANTIATE_TEST_CASE_P(GetComponentsGPUAndCPUTest, GetComponentsTest, 
                        Values(&get_components_cpu));

#else

// From Google documentation:
// Google Test may not support value-parameterized tests with some
// compilers. This dummy test keeps gtest_main linked in.
TEST_P(DummyTest, ValueParameterizedTestsAreNotSupportedOnThisPlatform) {}

#endif  // GTEST_HAS_PARAM_TEST
