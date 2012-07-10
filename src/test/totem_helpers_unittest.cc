/* TODO(lauro,abdullah,elizeu): Add license.
 *
 * Contains unit tests for totem helper functions.
 *
 *  Created on: 2011-03-09
 *      Author: Lauro Beltrão Costa
 *              Abdullah Gharaibeh
 */

// totem includes
#include "totem_bitmap.h"
#include "totem_common_unittest.h"

class GraphHelper : public ::testing::Test {
 protected:
  virtual void SetUp() {
    // Ensure the minimum CUDA architecture is supported
    CUDA_CHECK_VERSION();
  }
  void CompareBinary(graph_t* graph, const char* filename) {
    graph_t* graph_bin;
    EXPECT_EQ(SUCCESS, graph_store_binary(graph, filename));
    EXPECT_EQ(SUCCESS, graph_initialize(filename,
                                        graph->weighted, &graph_bin));
    EXPECT_EQ(graph->vertex_count, graph_bin->vertex_count);
    EXPECT_EQ(graph->edge_count, graph_bin->edge_count);
    EXPECT_EQ(graph->directed, graph_bin->directed);
    EXPECT_EQ(graph->weighted, graph_bin->weighted);
    EXPECT_EQ(graph->valued, graph_bin->valued);
    for (id_t vid = 0; vid < graph->vertex_count; vid++) {
      for (id_t i = graph->vertices[vid]; i < graph->vertices[vid + 1]; i++) {
        EXPECT_EQ(graph->edges[i], graph_bin->edges[i]);
        if (graph->weighted) {
          EXPECT_EQ(graph->weights[i], graph_bin->weights[i]);
        }
      }
    }
    EXPECT_EQ(SUCCESS, graph_finalize(graph_bin));
  }
};

// Tests for initialize helper function.
TEST_F(GraphHelper, Initialize) {
  graph_t* graph;
  EXPECT_EQ(SUCCESS, graph_initialize(DATA_FOLDER("single_node.totem"),
                                      false, &graph));
  EXPECT_EQ((uint32_t)1, graph->vertex_count);
  EXPECT_EQ((uint32_t)0, graph->edge_count);
  EXPECT_TRUE(graph->directed);
  EXPECT_FALSE(graph->weighted);
  EXPECT_FALSE(graph->valued);
  CompareBinary(graph, TEMP_FOLDER("single_node.tbin"));
  EXPECT_EQ(SUCCESS, graph_finalize(graph));

  EXPECT_EQ(SUCCESS,
            graph_initialize(DATA_FOLDER("single_node_loop.totem"),
                             false, &graph));
  EXPECT_EQ((uint32_t)1, graph->vertex_count);
  EXPECT_EQ((uint32_t)1, graph->edge_count);
  EXPECT_TRUE(graph->directed);
  EXPECT_FALSE(graph->weighted);
  EXPECT_FALSE(graph->valued);
  CompareBinary(graph, TEMP_FOLDER("single_node_loop.tbin"));
  EXPECT_EQ(SUCCESS, graph_finalize(graph));

  EXPECT_EQ(SUCCESS, graph_initialize(DATA_FOLDER("chain_1000_nodes.totem"),
                                      false, &graph));
  EXPECT_EQ((uint32_t)1000, graph->vertex_count);
  EXPECT_EQ((uint32_t)1998, graph->edge_count);
  EXPECT_FALSE(graph->directed);
  EXPECT_FALSE(graph->weighted);
  EXPECT_FALSE(graph->valued);
  CompareBinary(graph, TEMP_FOLDER("chain_1000_nodes.tbin"));
  EXPECT_EQ(SUCCESS, graph_finalize(graph));

  EXPECT_EQ(SUCCESS,
            graph_initialize(DATA_FOLDER("complete_graph_300_nodes_weight.totem"),
                             true, &graph));
  EXPECT_EQ((uint32_t)300, graph->vertex_count);
  EXPECT_EQ((uint32_t)89700, graph->edge_count);
  EXPECT_FALSE(graph->directed);
  EXPECT_TRUE(graph->weighted);
  EXPECT_FALSE(graph->valued);
  CompareBinary(graph, TEMP_FOLDER("complete_graph_300_nodes_weight.tbin"));
  EXPECT_EQ(SUCCESS, graph_finalize(graph));

  EXPECT_EQ(SUCCESS,
            graph_initialize(DATA_FOLDER("disconnected_1000_nodes.totem"),
                             false, &graph));
  EXPECT_EQ((uint32_t)1000, graph->vertex_count);
  EXPECT_EQ((uint32_t)0, graph->edge_count);
  EXPECT_FALSE(graph->directed);
  EXPECT_FALSE(graph->weighted);
  EXPECT_FALSE(graph->valued);
  CompareBinary(graph, TEMP_FOLDER("disconnected_1000_nodes.tbin"));
  EXPECT_EQ(SUCCESS, graph_finalize(graph));

  EXPECT_EQ(SUCCESS,
            graph_initialize(DATA_FOLDER("star_1000_nodes.totem"),
                             false, &graph));
  EXPECT_EQ((uint32_t)1000, graph->vertex_count);
  EXPECT_EQ((uint32_t)1998, graph->edge_count);
  EXPECT_FALSE(graph->directed);
  EXPECT_FALSE(graph->weighted);
  EXPECT_FALSE(graph->valued);
  CompareBinary(graph, TEMP_FOLDER("star_1000_nodes.tbin"));
  EXPECT_EQ(SUCCESS, graph_finalize(graph));
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
}


TEST_F(GraphHelper, AtomicOperations) {
  // the following are used in all tests
  srand (time(NULL));
  int buf_count = 1000;
  // we use three different arrays instead of one just to avoid casting
  // frequently to make the code more readable
  int* buf = (int*)malloc(buf_count * sizeof(int));
  float* buf_f = (float*)malloc(buf_count * sizeof(float));
  double* buf_d = (double*)malloc(buf_count * sizeof(double));
  for (int i = 0; i < buf_count; i++) {
    buf[i] = rand() % 100;
    // add 0.5 to cover the fractional part in the test
    buf_f[i] = (float)((float)buf[i] + 0.5);
    buf_d[i] = (double)((double)buf[i] + 0.5);
  }

  // Atomic floating add
  // Note that for floating point operations, the order of adding a set of items
  // affects the final sum due to rounding errors, hence in this test we use
  // only integers to avoid this problem
  // single precision
  float sum_float = 0;
  for (int i = 0; i < buf_count; i++) {
    sum_float += buf_f[i];
  }
  float p_sum_float = 0;
#pragma omp parallel for
  for (int i = 0; i < buf_count; i++) {
    __sync_fetch_and_add_float(&p_sum_float, buf_f[i]);
  }
  EXPECT_FLOAT_EQ(p_sum_float, sum_float);

  // double precision
  double sum_double = 0;
  for (int i = 0; i < buf_count; i++) {
    sum_double += buf_d[i];
  }
  double p_sum_double = 0;
#pragma omp parallel for
  for (int i = 0; i < buf_count; i++) {
    __sync_fetch_and_add_double(&p_sum_double, buf_d[i]);
  }
  EXPECT_DOUBLE_EQ(p_sum_double, sum_double);

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
  float min_float = 0;
  for (int i = 0; i < buf_count; i++) {
    min_float = min_float > buf_f[i] ? buf_f[i] : min_float;
  }
  float p_min_float = 0;
#pragma omp parallel for
  for (int i = 0; i < buf_count; i++) {
    __sync_fetch_and_min_float(&p_min_float, buf_f[i]);
  }
  EXPECT_FLOAT_EQ(p_min_float, min_float);

  // double precision
  double min_double = 0;
  for (int i = 0; i < buf_count; i++) {
    min_double = min_double > buf_d[i] ? buf_d[i] : min_double;
  }
  double p_min_double = 0;
#pragma omp parallel for
  for (int i = 0; i < buf_count; i++) {
    __sync_fetch_and_min_double(&p_min_double, buf_d[i]);
  }
  EXPECT_DOUBLE_EQ(p_min_double, min_double);

  free(buf);
  free(buf_f);
  free(buf_d);
}


TEST_F(GraphHelper, Bitmap) {
  // Initialize the bitmap
  const id_t bitmap_len = 10006;
  bitmap_t bitmap = bitmap_init(bitmap_len);
  
  const id_t bits_set_count = bitmap_len/2;
  id_t bits_set[bits_set_count];
  // Initialize the bits to be set. The first two are initialized statically to
  // the first and last bits, the rest is initialized randomly.
  bits_set[0] = 0;
  bits_set[1] = bits_set_count - 1;
  srand (time(NULL));
  for (id_t i = 2; i < bits_set_count; i++) {
    bits_set[i] = rand() % bitmap_len;
  }

  // Get the bits that are not set
  id_t bits_not_set[bitmap_len];
  id_t bits_not_set_count = 0;
  for (id_t i = 0; i < bitmap_len; i++) {
    // Search for the bit in the bits_set array
    id_t j = 0;
    for (; j < bits_set_count; j++) {
      if (bits_set[j] == i) {
        break;
      }
    }
    if (j == bits_set_count) {
      // This bit is not set, add it to the bits_not_set array
      bits_not_set[bits_not_set_count++] = i;
    }
  }

  // First set the bits in the bits_set array
  #ifdef _OPENMP
  #pragma omp parallel for
  #endif // _OPENMP
  for (id_t i = 0; i < bits_set_count; i++) {
    bitmap_set(bitmap, bits_set[i]);
  }

  // Try to set again
  #ifdef _OPENMP
  #pragma omp parallel for
  #endif // _OPENMP
  for (id_t i = 0; i < bits_set_count; i++) {
    EXPECT_FALSE(bitmap_set(bitmap, bits_set[i]));
  }

  // Second check if the bits are set correctly
  for (id_t i = 0; i < bits_set_count; i++) {
    EXPECT_TRUE(bitmap_is_set(bitmap, bits_set[i]));
  }
  for (id_t i = 0; i < bits_not_set_count; i++) {
    EXPECT_FALSE(bitmap_is_set(bitmap, bits_not_set[i]));
  }

  bitmap_finalize(bitmap);
}
