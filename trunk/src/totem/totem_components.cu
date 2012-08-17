
/* TODO(lauro,abdullah,elizeu): Add license.
 *
 * Implements an algorithm to identify the weakly connected components of a
 * graph. The algorithm is based on BFS.
 *
 *  Created on: 2011-11-23
 *      Author: Abdullah Gharaibeh
 */

#include "totem_comdef.h"
#include "totem_graph.h"
#include "totem_mem.h"

PRIVATE error_t allocate_component_set(graph_t* graph, vid_t comp_count, 
                                       component_set_t** comp_set) {
  *comp_set = (component_set_t*)calloc(1, sizeof(component_set_t));
  assert(*comp_set);
  (*comp_set)->graph = graph;
  (*comp_set)->marker = (vid_t*)malloc((graph)->vertex_count * sizeof(vid_t));
  memset((*comp_set)->marker, 0xFF, (graph)->vertex_count * sizeof(vid_t));
  if (comp_count != 0) {
    (*comp_set)->vertex_count = (vid_t*)calloc((comp_count), sizeof(vid_t));
    (*comp_set)->edge_count = (eid_t*)calloc((comp_count), sizeof(eid_t));
    (*comp_set)->count = comp_count;
    (*comp_set)->biggest = 0;
  }
  return SUCCESS;
}

/**
 * Checks for input parameters and special cases. This is invoked at the
 * beginning of public interface
*/
PRIVATE
error_t check_special_cases(graph_t* graph, component_set_t** comp_set, 
                            bool* finished) {
  *finished = true;
  if (graph == NULL) {
    return FAILURE;
  } else if (graph->vertex_count == 0) {
    return FAILURE;
  } else if (graph->vertex_count > 0 && graph->edge_count == 0) {
    allocate_component_set(graph, graph->vertex_count, comp_set);
    for (vid_t vid = 0; vid < graph->vertex_count; vid++) {
      (*comp_set)->marker[vid] = vid;
      (*comp_set)->vertex_count[vid] = 1;
      (*comp_set)->edge_count[vid] = 0;
    }
    return SUCCESS;
  }
  *finished = false;
  return SUCCESS;
}


/**
 * performs BFS traversal starting from src and marks all visited nodes
 * with component id comp.
 * @param[in] graph
 * @param[in] src the vertex at which BFS starts the traversal
 * @param[in] marker vertices visited will be marked with comp
 * @param[in] comp the id of the current component
 */
PRIVATE void mark_component(const graph_t* graph, vid_t src, vid_t* marker, 
                            vid_t comp) {
  // TODO(abdullah): use bfs_* functions implemented in totem_bfs.cu to minimize
  // code maintenance overhead. The difference between this bfs-like
  // implementation and the ones in totem_bfs.cu is that this one marks the
  // vertices with their component id on the fly which has the potential to
  // improve performance in the case of graphs with large number of components.
  // Also, it assumes that all the vertices less than src has already been
  // visited, hence it skips iterating over them. An advantage of using the
  // bfs_* functions is modularity. One way to enable such a thing in the
  // original bfs implementation is to have callbacks in them.

  assert(graph && (src < graph->vertex_count) && (marker[src] == INFINITE));
  marker[src] = comp;
  // single vertex component
  if ((graph->vertices[src + 1] - graph->vertices[src]) == 0) {
    return;
  }

  // while the current level has vertices to be processed
  bool finished = false;
  for (vid_t level = comp; !finished; level++) {
    finished = true;
    OMP(omp parallel for)
    for (vid_t vid = src; vid < graph->vertex_count; vid++) {
      // the assumption is that all the vertices less than src has alredy been
      // marked, therefore we can safely skip them and start the loop from src.
      if (marker[vid] != level) continue;
      marker[vid] = comp;
      for (eid_t i = graph->vertices[vid]; i < graph->vertices[vid + 1]; i++) {
        const vid_t nbr = graph->edges[i];
        if (marker[nbr] == INFINITE) {
          finished = false;
          marker[nbr] = level + 1;
        }
      }
    }
  }
}

error_t get_components_cpu(graph_t* graph, component_set_t** comp_set_ret) {

  assert(graph);
  // Check for special cases
  bool finished = false;
  error_t rc = check_special_cases(graph, comp_set_ret, &finished);
  if (finished) return rc;

  vid_t comp_count = 0;
  component_set_t* comp_set;
  allocate_component_set(graph, comp_count, &comp_set);
  for (vid_t vid = 0; vid < graph->vertex_count; vid++) {
    if (comp_set->marker[vid] == INFINITE) {
      mark_component(graph, vid, comp_set->marker, comp_count);
      comp_count++;
    }
  }
  comp_set->count = comp_count;

  // compute the vertex and edge count of each component
  comp_set->vertex_count = (vid_t*)calloc(comp_count, sizeof(vid_t));
  comp_set->edge_count   = (eid_t*)calloc(comp_count, sizeof(eid_t));
  OMP(omp parallel for)
  for (vid_t vid = 0; vid < graph->vertex_count; vid++) {
    vid_t comp = comp_set->marker[vid];
    __sync_fetch_and_add(&(comp_set->vertex_count[comp]), 1);
    vid_t nbr_count = graph->vertices[vid + 1] - graph->vertices[vid];
    __sync_fetch_and_add(&(comp_set->edge_count[comp]), nbr_count);
  }

  // identify the biggest component
  comp_set->biggest = 0;
  for (vid_t comp = 1; comp < comp_set->count; comp++) {
    if (comp_set->vertex_count[comp] >
        comp_set->vertex_count[comp_set->biggest]) {
      comp_set->biggest = comp;
    }
  }

  *comp_set_ret = comp_set;
  return SUCCESS;
}

error_t finalize_component_set(component_set_t* comp_set) {
  if (!comp_set) return FAILURE;
  if (comp_set->marker) free(comp_set->marker);
  if (comp_set->vertex_count) free(comp_set->vertex_count);
  if (comp_set->edge_count) free(comp_set->edge_count);
  free(comp_set);
  return SUCCESS;
}
