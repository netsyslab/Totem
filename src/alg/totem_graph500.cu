/* This file contains an implementation of the breadth-first search (BFS) graph
 * search algorithm based on the algorithms in [Hong2011PPoPP, Hong2011PACT].
 * [Hong2011PPoPP] S. Hong,  S.K. Kim, T. Oguntebi and K. Olukotun, 
 *   "Accelerating CUDA graph algorithms at maximum warp" in PPoPP 2011.
 * [Hong2011PACT] S. Hong, T. Oguntebi and K. Olukotun, "Efficient parallel 
 *   graph exploration on multi-core cpu and gpu" in PACT 2011.
 *
 *  Created on: 2013-05-27
 *      Author: Abdullah Gharaibeh
 */

#include "totem_alg.h"

PRIVATE error_t check_special_cases(graph_t* graph, vid_t src, bfs_tree_t* tree,
                                    bool* finished) {
  *finished = true;
  if((graph == NULL) || (src >= graph->vertex_count) || (tree == NULL)) {
    return FAILURE;
  } else if(graph->vertex_count == 1) {
    tree[0] = src;
    return SUCCESS;
  } else if(graph->edge_count == 0) {
    // Initialize tree to INFINITE and self to the source node
    totem_memset(tree, (bfs_tree_t)-1, graph->vertex_count, TOTEM_MEM_HOST);
    tree[src] = src;
    return SUCCESS;
  }
  *finished = false;
  return SUCCESS;
}

PRIVATE bitmap_t initialize_cpu(graph_t* graph, vid_t src, bfs_tree_t* tree, 
                                cost_t** cost) {
  // Initialize cost to INFINITE and create the vertices bitmap
  CALL_SAFE(totem_malloc(graph->vertex_count * sizeof(cost_t), TOTEM_MEM_HOST, 
                         (void**)cost));
  totem_memset(*cost, INF_COST, graph->vertex_count, TOTEM_MEM_HOST);
  totem_memset(tree, (bfs_tree_t)-1, graph->vertex_count, 
               TOTEM_MEM_HOST);
  bitmap_t visited = bitmap_init_cpu(graph->vertex_count);
  
  // Initialize the state of the source vertex
  (*cost)[src] = 0;
  bitmap_set_cpu(visited, src);
  tree[src] = src;
  return visited;
}

__host__
error_t graph500_cpu(graph_t* graph, vid_t src, bfs_tree_t* tree) {
  // Check for special cases
  bool finished = false;
  error_t rc = check_special_cases(graph, src, tree, &finished);
  if (finished) return rc;

  cost_t* cost = NULL;
  bitmap_t visited = initialize_cpu(graph, src, tree, &cost);

  finished = false;
  // Within the following code segment, all threads execute in parallel the 
  // same code (similar to a cuda kernel)
  OMP(omp parallel)
  {
    // level is a local variable to each thread, having a separate copy per
    // thread reduces the overhead of cache coherency protocol
    cost_t level = 0;
    // while the current level has vertices to be processed.
    while (!finished) {
      // This barrier ensures that all threads checked the while condition above
      // using the same "finished" value that resulted from the previous
      // iteration before it is initialized again for the next one.
      OMP(omp barrier)

      // This "single" clause ensures that only one thread sets the variable. 
      // Note that this close has an implicit barrier (i.e., all threads will
      // block until the variable is set by the responsible thread)
      OMP(omp single) {
        finished = true;
      }
      // The "for" clause instructs openmp to run the loop in parallel. Each
      // thread will be assigned a chunk of work depending on the chosen OMP
      // scheduling algorithm. The reduction clause defines a private temporary
      // variable for each thread, reduces them in the end using an "and" 
      // operator and stores the value in "finished". This improves performance
      // by reducing cache coherency overhead. The "runtime" scheduling clause
      // defer the choice of thread scheduling algorithm to the client, 
      // either via OS environment variable or omp_set_schedule interface.
      OMP(omp for schedule(runtime) reduction(& : finished))
      for (vid_t vertex_id = 0; vertex_id < graph->vertex_count; vertex_id++) {
        if (cost[vertex_id] != level) continue;
        for (eid_t i = graph->vertices[vertex_id];
             i < graph->vertices[vertex_id + 1]; i++) {
          const vid_t neighbor_id = graph->edges[i];
          if (!bitmap_is_set(visited, neighbor_id)) {
            if (bitmap_set_cpu(visited, neighbor_id)) {
              finished = false;
              cost[neighbor_id] = level + 1;
              tree[neighbor_id] = vertex_id;
            }
          }
        }
      }
      level++;
    }
  } // omp parallel
  bitmap_finalize_cpu(visited);
  totem_free(cost, TOTEM_MEM_HOST);
  return SUCCESS;
}

