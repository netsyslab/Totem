/* TODO(lauro,abdullah,elizeu): Add license.
 *
 * Contains unit tests for an implementation of the breadth-first search (BFS)
 * graph search algorithm.
 *
 *  Created on: 2011-03-08
 *      Author: Lauro Beltrão Costa
 */

// totem includes
#include "totem_common_unittest.h"

// Tests BFS for empty graphs.
TEST(BFSTest, Empty) {
  graph_t graph;
  graph.directed = false;
  graph.vertex_count = 0;
  graph.edge_count = 0;

  uint32_t* cost;
  EXPECT_EQ(FAILURE, bfs_gpu(0, &graph, &cost));
  EXPECT_EQ(NULL, cost);

  EXPECT_EQ(FAILURE, bfs_gpu(99, &graph, &cost));
  EXPECT_EQ(NULL, cost);
}

// Tests BFS for single node graphs.
TEST(BFSTest, SingleNode) {
  graph_t* graph;
  graph_initialize(DATA_FOLDER("single_node.totem"), false, &graph);

  uint32_t* cost;
  EXPECT_EQ(SUCCESS, bfs_gpu(0, graph, &cost));
  EXPECT_FALSE(NULL == cost);
  EXPECT_EQ((uint32_t)0, cost[0]);
  mem_free(cost);

  EXPECT_EQ(FAILURE, bfs_gpu(1, graph, &cost));
  EXPECT_EQ(NULL, cost);
  graph_finalize(graph);

  graph_initialize(DATA_FOLDER("single_node_loop.totem"), false, &graph);
  EXPECT_EQ(SUCCESS, bfs_gpu(0, graph, &cost));
  EXPECT_FALSE(NULL == cost);
  EXPECT_EQ((uint32_t)0, cost[0]);
  mem_free(cost);

  EXPECT_EQ(FAILURE, bfs_gpu(1, graph, &cost));
  EXPECT_EQ(NULL, cost);
  graph_finalize(graph);
}

// Tests BFS for a chain of 1000 nodes.
TEST(BFSTest, Chain) {
  graph_t* graph;
  graph_initialize(DATA_FOLDER("chain_1000_nodes.totem"), false, &graph);

  // First vertex as source
  id_t source = 0;
  uint32_t* cost;
  EXPECT_EQ(SUCCESS, bfs_gpu(source, graph, &cost));
  EXPECT_FALSE(NULL == cost);
  for(id_t vertex = source; vertex < graph->vertex_count; vertex++){
    EXPECT_EQ(vertex, cost[vertex]);
  }
  mem_free(cost);

  // Last vertex as source
  source = graph->vertex_count - 1;
  EXPECT_EQ(SUCCESS, bfs_gpu(source, graph, &cost));
  for(id_t vertex = source; vertex < graph->vertex_count; vertex++){
    EXPECT_EQ(source - vertex, cost[vertex]);
  }
  mem_free(cost);

  // A vertex in the middle as source
  source = 199;
  EXPECT_EQ(SUCCESS, bfs_gpu(source, graph, &cost));
  for(id_t vertex = 0; vertex < graph->vertex_count; vertex++) {
    EXPECT_EQ((uint32_t)abs(source - vertex), cost[vertex]);
  }
  mem_free(cost);

  // Non existent vertex source
  EXPECT_EQ(FAILURE, bfs_gpu(graph->vertex_count, graph, &cost));
  EXPECT_EQ(NULL, cost);

  graph_finalize(graph);
}

// Tests BFS for a complete graph of 300 nodes.
TEST(BFSTest, CompleteGraph) {
  graph_t* graph;
  graph_initialize(DATA_FOLDER("complete_graph_300_nodes.totem"), false,
                   &graph);

  // First vertex as source
  id_t source = 0;
  uint32_t* cost;
  EXPECT_EQ(SUCCESS, bfs_gpu(source, graph, &cost));
  EXPECT_EQ((uint32_t)0, cost[source]);
  for(id_t vertex = source + 1; vertex < graph->vertex_count; vertex++){
    EXPECT_EQ((uint32_t)1, cost[vertex]);
  }
  mem_free(cost);

  // Last vertex as source
  source = graph->vertex_count - 1;
  EXPECT_EQ(SUCCESS, bfs_gpu(source, graph, &cost));
  EXPECT_EQ((uint32_t)0, cost[source]);
  for(id_t vertex = 0; vertex < source; vertex++) {
    EXPECT_EQ((uint32_t)1, cost[vertex]);
  }
  mem_free(cost);

  // A vertex source in the middle
  source = 199;
  EXPECT_EQ(SUCCESS, bfs_gpu(source, graph, &cost));
  for(id_t vertex = 0; vertex < graph->vertex_count; vertex++) {
    EXPECT_EQ((uint32_t)((source == vertex)?0:1), cost[vertex]);
  }
  mem_free(cost);

  // Non existent vertex source
  EXPECT_EQ(FAILURE, bfs_gpu(graph->vertex_count, graph, &cost));
  EXPECT_EQ(NULL, cost);

  graph_finalize(graph);
}

// TODO(lauro): Add test cases for not well defined structures.
