/**
 * Implements the graph interface defined in totem_graph.h
 *
 *  Created on: 2011-02-28
 *  Author: Abdullah Gharaibeh
 */

// totem includes
#include "totem_graph.h"

error_t graph_initialize(const char* graph_file, bool with_weights, 
			 graph_t** graph) {
  return FAILURE;
}

error_t graph_finalize(graph_t* graph) {
  assert(graph);
  assert(graph->vertices);
  assert(graph->edges);

  free(graph->vertices);
  free(graph->edges);
  if (graph->with_weights) {
    assert(graph->weights);
    free(graph->weights);
  }
  free(graph);

  return SUCCESS;
}
