/* TODO(lauro,abdullah,elizeu): Add license.
 *
 * Contains unit tests for an implementation of the PageRank graph algorithm.
 *
 *  Created on: 2011-03-22
 *      Author: Abdullah Gharaibeh
 */

// totem includes
#include "totem_common_unittest.h"

// Tests PageRank for empty graphs.
TEST(PageRankTest, Empty) {
  graph_t graph;
  graph.directed = false;
  graph.vertex_count = 0;
  graph.edge_count = 0;

  float* rank = NULL;
  EXPECT_EQ(FAILURE, page_rank_cpu(&graph, &rank));
}

// Tests PageRank for single node graphs.
TEST(PageRankTest, SingleNode) {
  graph_t* graph;
  EXPECT_EQ(SUCCESS, graph_initialize(DATA_FOLDER("single_node.totem"), 
                                      false, &graph));

  float* rank = NULL;
  EXPECT_EQ(SUCCESS, page_rank_cpu(graph, &rank));
  EXPECT_FALSE(rank == NULL);
  EXPECT_EQ(1, rank[0]);
  mem_free(rank);

  EXPECT_EQ(SUCCESS, graph_finalize(graph));
}

// Tests PageRank for a chain of 1000 nodes.
TEST(PageRankTest, Chain) {
  graph_t* graph;
  EXPECT_EQ(SUCCESS, graph_initialize(DATA_FOLDER("chain_1000_nodes.totem"),
                                      false, &graph));

  float* rank = NULL;
  EXPECT_EQ(SUCCESS, page_rank_cpu(graph, &rank));
  EXPECT_FALSE(rank == NULL);
  for(id_t vertex = 0; vertex < graph->vertex_count/2; vertex++){
    EXPECT_EQ(rank[vertex], rank[graph->vertex_count - vertex - 1]);
  }
  mem_free(rank);

  EXPECT_EQ(SUCCESS, graph_finalize(graph));
}

// Tests PageRank for a complete graph of 300 nodes.
TEST(PageRankTest, CompleteGraph) {
  graph_t* graph;
  EXPECT_EQ(SUCCESS, 
            graph_initialize(DATA_FOLDER("complete_graph_300_nodes.totem"), 
                             false, &graph));

  float* rank = NULL;
  EXPECT_EQ(SUCCESS, page_rank_cpu(graph, &rank));
  EXPECT_FALSE(rank == NULL);
  for(id_t vertex = 0; vertex < graph->vertex_count; vertex++){
    EXPECT_EQ(rank[0], rank[vertex]);
  }
  mem_free(rank);

  EXPECT_EQ(SUCCESS, graph_finalize(graph));
}

// Tests PageRank for a complete graph of 300 nodes.
TEST(PageRankTest, Star) {
  graph_t* graph;
  EXPECT_EQ(SUCCESS, 
            graph_initialize(DATA_FOLDER("star_1000_nodes.totem"), 
                             false, &graph));

  float* rank = NULL;
  EXPECT_EQ(SUCCESS, page_rank_cpu(graph, &rank));
  EXPECT_FALSE(rank == NULL);
  for(id_t vertex = 1; vertex < graph->vertex_count; vertex++){
    EXPECT_EQ(rank[1], rank[vertex]);
    EXPECT_GT(rank[0], rank[vertex]);
  }
  mem_free(rank);

  EXPECT_EQ(SUCCESS, graph_finalize(graph));
}

// TODO(abdullah): Add test cases for not well defined structures.
