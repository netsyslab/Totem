/* TODO(lauro,abdullah,elizeu): Add license.
 *
 * Contains unit tests for totem helper functions.
 *
 *  Created on: 2011-03-09
 *      Author: Lauro Beltrão Costa
 */

// totem includes
#include "totem_common_unittest.h"

// Tests for initialize helper function.
TEST(GraphHelper, Initialize) {
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

TEST(GraphHelper, SubGraph) {
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
