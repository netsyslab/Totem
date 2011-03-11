/* TODO(lauro,abdullah,elizeu): Add license.
 *
 * Contains unit tests for totem helper functions.
 *
 *  Created on: 2011-03-09
 *      Author: Lauro Beltrão Costa
 */

// test includes
#include "gtest/gtest.h"

// totem includes
#include "totem_comdef.h"

// functionality tested
#include "totem_graph.h"

// Tests for initialize helper function.
TEST(GraphHelper, Initialize) {
  graph_t* graph;

  graph_initialize("data/single_node.totem", false, &graph);
  EXPECT_EQ(1, graph->vertex_count);
  EXPECT_EQ(0, graph->edge_count);
  graph_finalize(graph);

  graph_initialize("data/single_node_loop.totem", false, &graph);
  EXPECT_EQ(1, graph->vertex_count);
  EXPECT_EQ(1, graph->edge_count);
  graph_finalize(graph);

  graph_initialize("data/chain_1000_nodes.totem", false, &graph);
  EXPECT_EQ(1000, graph->vertex_count);
  EXPECT_EQ(1998, graph->edge_count);
  graph_finalize(graph);

  graph_initialize("data/complete_graph_300_nodes.totem", false, &graph);
  EXPECT_EQ(300, graph->vertex_count);
  EXPECT_EQ(89700, graph->edge_count);
  graph_finalize(graph);

  // TODO(lauro, abdullah): Add more cases and test other fields.
}
