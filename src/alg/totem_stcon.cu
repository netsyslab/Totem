/* TODO(lauro,abdullah,elizeu): Add license.
 *
 * This file contains an implementation of the ST Connectivity.
 *
 *  Created on: 2011-04-01
 *      Author: Lauro Beltrão Costa
 */

// totem includes
#include "totem_alg.h"

__host__
error_t stcon_gpu(const graph_t* graph, vid_t source_id, vid_t destination_id,
                  bool* connected) {
  if((graph == NULL) || (source_id >= graph->vertex_count)
     || (destination_id >= graph->vertex_count)) {
    *connected = false;
    return FAILURE;
  }
  if( source_id == destination_id ) {
    *connected = true;
    return SUCCESS;
  }

  dim3 blocks;
  dim3 threads_per_block;
  KERNEL_CONFIGURE(graph->vertex_count, blocks, threads_per_block);
  //TODO(abdullah, lauro) handle the case (vertex_count > number of threads).
  assert(graph->vertex_count <= MAX_THREAD_COUNT);

  // Create graph on GPU memory.
  graph_t* graph_d;
  CHK_SUCCESS(graph_initialize_device(graph, &graph_d), err);

  // TODO(lauro): Finish stcon_gpu implementation.

  graph_finalize_device(graph_d);
  *connected = false;
  return FAILURE;

  // error handlers
  err:
    *connected = false;
    return FAILURE;
}

__host__
error_t stcon_cpu(const graph_t* graph, vid_t source_id, vid_t destination_id,
                  bool* connected) {
  if((graph == NULL) || (source_id >= graph->vertex_count)
     || (destination_id >= graph->vertex_count)) {
    *connected = false;
    return FAILURE;
  }
  if( source_id == destination_id ) {
    *connected = true;
    return SUCCESS;
  }

  // TODO(lauro): Finish stcon_cpu implementation.

  *connected = false;
  return FAILURE;
}
