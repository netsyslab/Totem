/* TODO(lauro,abdullah,elizeu): Add license.
 *
 * Contains unit tests for totem helper functions.
 *
 *  Created on: 2011-03-09
 *      Author: Lauro Beltrão Costa
 */

// totem includes
#include "totem_common_unittest.h"

class GraphHelper : public ::testing::Test {
 protected:
  id_t* partitions_;
  partition_set_t* partition_set_;
  virtual void SetUp() {
    // Ensure the minimum CUDA architecture is supported
    CUDA_CHECK_VERSION();
    partitions_ = NULL;
    partition_set_ = NULL;
  }
  virtual void TearDown() {
    if (partitions_ != NULL) {
      free(partitions_);
    }
    if (partition_set_ != NULL) {
      partition_set_finalize(partition_set_);
    }
  }
};

// Tests for initialize helper function.
TEST_F(GraphHelper, Initialize) {
  graph_t* graph;
  EXPECT_EQ(SUCCESS, graph_initialize(DATA_FOLDER("single_node.totem"),
                                      false, &graph));
  EXPECT_EQ((uint32_t)1, graph->vertex_count);
  EXPECT_EQ((uint32_t)0, graph->edge_count);
  EXPECT_EQ(SUCCESS, graph_finalize(graph));

  EXPECT_EQ(SUCCESS,
            graph_initialize(DATA_FOLDER("single_node_loop.totem"),
                             false, &graph));
  EXPECT_EQ((uint32_t)1, graph->vertex_count);
  EXPECT_EQ((uint32_t)1, graph->edge_count);
  EXPECT_EQ(SUCCESS, graph_finalize(graph));

  EXPECT_EQ(SUCCESS, graph_initialize(DATA_FOLDER("chain_1000_nodes.totem"),
                                      false, &graph));
  EXPECT_EQ((uint32_t)1000, graph->vertex_count);
  EXPECT_EQ((uint32_t)1998, graph->edge_count);
  EXPECT_EQ(SUCCESS, graph_finalize(graph));

  EXPECT_EQ(SUCCESS,
            graph_initialize(DATA_FOLDER("complete_graph_300_nodes.totem"),
                             false, &graph));
  EXPECT_EQ((uint32_t)300, graph->vertex_count);
  EXPECT_EQ((uint32_t)89700, graph->edge_count);
  EXPECT_EQ(SUCCESS, graph_finalize(graph));

  EXPECT_EQ(SUCCESS,
            graph_initialize(DATA_FOLDER("disconnected_1000_nodes.totem"),
                             false, &graph));
  EXPECT_EQ((uint32_t)1000, graph->vertex_count);
  EXPECT_EQ((uint32_t)0, graph->edge_count);
  EXPECT_EQ(SUCCESS, graph_finalize(graph));


  EXPECT_EQ(SUCCESS,
            graph_initialize(DATA_FOLDER("star_1000_nodes.totem"),
                             false, &graph));
  EXPECT_EQ((uint32_t)1000, graph->vertex_count);
  EXPECT_EQ((uint32_t)1998, graph->edge_count);
  EXPECT_EQ(SUCCESS, graph_finalize(graph));

  // TODO(lauro, abdullah): Add more cases and test other fields.
}

TEST_F(GraphHelper, SubGraph) {
  graph_t* graph;
  graph_t* subgraph;
  bool* mask;


  // single node graph
  subgraph = NULL;
  graph_initialize(DATA_FOLDER("single_node.totem"), false, &graph);
  mask = (bool*)malloc(graph->vertex_count * sizeof(bool));
  // include the only vertex
  mask[0] = true;
  EXPECT_EQ(SUCCESS, get_subgraph(graph, mask, &subgraph));
  EXPECT_EQ((uint32_t)1, subgraph->vertex_count);
  EXPECT_EQ((uint32_t)0, subgraph->edge_count);
  EXPECT_EQ(SUCCESS, graph_finalize(subgraph));
  // exclude the only vertex
  mask[0] = false;
  EXPECT_EQ(SUCCESS, get_subgraph(graph, mask, &subgraph));
  EXPECT_EQ((uint32_t)0, subgraph->vertex_count);
  EXPECT_EQ((uint32_t)0, subgraph->edge_count);
  EXPECT_EQ(SUCCESS, graph_finalize(subgraph));
  // cleanup
  free(mask);
  graph_finalize(graph);


  // single node loop graph
  subgraph = NULL;
  graph_initialize(DATA_FOLDER("single_node_loop.totem"), false, &graph);
  mask = (bool*)malloc(graph->vertex_count * sizeof(bool));
  // include the only vertex
  mask[0] = true;
  EXPECT_EQ(SUCCESS, get_subgraph(graph, mask, &subgraph));
  EXPECT_EQ((uint32_t)1, subgraph->vertex_count);
  EXPECT_EQ((uint32_t)1, subgraph->edge_count);
  EXPECT_EQ(SUCCESS, graph_finalize(subgraph));
  // exclude the only vertex
  mask[0] = false;
  EXPECT_EQ(SUCCESS, get_subgraph(graph, mask, &subgraph));
  EXPECT_EQ((uint32_t)0, subgraph->vertex_count);
  EXPECT_EQ((uint32_t)0, subgraph->edge_count);
  EXPECT_EQ(SUCCESS, graph_finalize(subgraph));
  // cleanup
  free(mask);
  graph_finalize(graph);


  // chain graph
  subgraph = NULL;
  graph_initialize(DATA_FOLDER("chain_1000_nodes.totem"), false, &graph);
  mask = (bool*)malloc(graph->vertex_count * sizeof(bool));
  // first continuous half
  memset(mask, false, graph->vertex_count);
  for (uint32_t i = 0; i < graph->vertex_count/2; i++) mask[i] = true;
  EXPECT_EQ(SUCCESS, get_subgraph(graph, mask, &subgraph));
  EXPECT_EQ((uint32_t)500, subgraph->vertex_count);
  EXPECT_EQ((uint32_t)(500 * 2 - 2), subgraph->edge_count);
  EXPECT_EQ(SUCCESS, graph_finalize(subgraph));
  // alternating half
  memset(mask, false, graph->vertex_count);
  for (uint32_t i = 0; i < graph->vertex_count; i+=2) mask[i] = true;
  EXPECT_EQ(SUCCESS, get_subgraph(graph, mask, &subgraph));
  EXPECT_EQ((uint32_t)500, subgraph->vertex_count);
  EXPECT_EQ((uint32_t)0, subgraph->edge_count);
  EXPECT_EQ(SUCCESS, graph_finalize(subgraph));
  // clean up
  free(mask);
  graph_finalize(graph);


  // complete graph
  subgraph = NULL;
  graph_initialize(DATA_FOLDER("complete_graph_300_nodes.totem"), 
                   false, &graph);
  mask = (bool*)malloc(graph->vertex_count * sizeof(bool));
  // alternating half
  memset(mask, false, graph->vertex_count);
  for (uint32_t i = 0; i < graph->vertex_count; i+=2) mask[i] = true;
  EXPECT_EQ(SUCCESS, get_subgraph(graph, mask, &subgraph));
  EXPECT_EQ((uint32_t)150, subgraph->vertex_count);
  EXPECT_EQ((uint32_t)(150 * 149), subgraph->edge_count);
  EXPECT_EQ(SUCCESS, graph_finalize(subgraph));
  // cleanup
  free(mask);
  graph_finalize(graph);


  // diconnected graph
  subgraph = NULL;
  graph_initialize(DATA_FOLDER("disconnected_1000_nodes.totem"), false, &graph);
  mask = (bool*)malloc(graph->vertex_count * sizeof(bool));
  // alternating half
  memset(mask, false, graph->vertex_count);
  for (uint32_t i = 0; i < graph->vertex_count; i+=2) mask[i] = true;
  EXPECT_EQ(SUCCESS, get_subgraph(graph, mask, &subgraph));
  EXPECT_EQ((uint32_t)500, subgraph->vertex_count);
  EXPECT_EQ((uint32_t)0, subgraph->edge_count);
  EXPECT_EQ(SUCCESS, graph_finalize(subgraph));
  // cleaup
  free(mask);
  graph_finalize(graph);


  // star graph
  subgraph = NULL;
  graph_initialize(DATA_FOLDER("star_1000_nodes.totem"), false, &graph);
  mask = (bool*)malloc(graph->vertex_count * sizeof(bool));
  memset(mask, false, graph->vertex_count);
  // half the graph including the hub
  for (uint32_t i = 0; i < graph->vertex_count; i+=2) mask[i] = true;
  EXPECT_EQ(SUCCESS, get_subgraph(graph, mask, &subgraph));
  EXPECT_EQ((uint32_t)500, subgraph->vertex_count);
  EXPECT_EQ((uint32_t)(500 - 1) * 2, subgraph->edge_count);
  EXPECT_EQ(SUCCESS, graph_finalize(subgraph));
  // exclude the hub
  mask[0] = false;
  EXPECT_EQ(SUCCESS, get_subgraph(graph, mask, &subgraph));
  EXPECT_EQ((uint32_t)499, subgraph->vertex_count);
  EXPECT_EQ((uint32_t)0, subgraph->edge_count);
  EXPECT_EQ(SUCCESS, graph_finalize(subgraph));
  // cleanup
  free(mask);
  graph_finalize(graph);

  // TODO(lauro, abdullah): Add more cases and test other fields.
}


TEST_F(GraphHelper, AtomicOperations) {

  // the following is used in all tests
  srand (time(NULL));
  int buf_count = 1000;
  int* buf = (int*)malloc(buf_count * sizeof(int));
  for (int i = 0; i < buf_count; i++) {
    buf[i] = rand() % 100;
  }

  // Atomic floating add
  // Note that for floating point operations, the order of adding a set of items
  // affects the final sum due to rounding errors, hence in this test we use 
  // only integers to avoid this problem
  // single precision
  float sum_float = 0;
  for (int i = 0; i < buf_count; i++) {
    sum_float += (float)buf[i];
  }
  float p_sum_float = 0;
#pragma omp parallel for
  for (int i = 0; i < buf_count; i++) {    
    __sync_fetch_and_add_float(&p_sum_float, (float)buf[i]);
  }
  EXPECT_EQ(p_sum_float, sum_float);

  // double precision
  double sum_double = 0;
  for (int i = 0; i < buf_count; i++) {
    sum_double += (double)buf[i];
  }
  double p_sum_double = 0;
#pragma omp parallel for
  for (int i = 0; i < buf_count; i++) {
    __sync_fetch_and_add_double(&p_sum_double, (double)buf[i]);
  }
  EXPECT_EQ(p_sum_double, sum_double);


  // Atomic integer and floating point min
  // integer
  int min_int = 0;
  for (int i = 0; i < buf_count; i++) {
    min_int = min_int > buf[i] ? buf[i] : min_int;
  }
  int p_min_int = 0;
#pragma omp parallel for
  for (int i = 0; i < buf_count; i++) {
    __sync_fetch_and_min(&p_min_int, buf[i]);
  }
  EXPECT_EQ(p_min_int, min_int);

  // single precision
  float factor = .67;
  float min_float = 0;
  for (int i = 0; i < buf_count; i++) {
    float value = (float)buf[i] * factor;
    min_float = min_float > value ? value : min_float;
  }
  float p_min_float = 0;
#pragma omp parallel for
  for (int i = 0; i < buf_count; i++) {
    float value = (float)buf[i] * factor;
    __sync_fetch_and_min_float(&p_min_float, value);
  }
  EXPECT_EQ(p_min_float, min_float);

  // double precision
  double min_double = 0;
  for (int i = 0; i < buf_count; i++) {
    double value = (double)buf[i] * factor;
    min_double = min_double > value ? value : min_double;
  }
  double p_min_double = 0;
#pragma omp parallel for
  for (int i = 0; i < buf_count; i++) {
    double value = (double)buf[i] * factor;
    __sync_fetch_and_min_double(&p_min_double, value);
  }
  EXPECT_EQ(p_min_double, min_double);
}
