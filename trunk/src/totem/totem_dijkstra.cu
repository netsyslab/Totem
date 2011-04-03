/**
 * // TODO(elizeu): Add license.
 *
 * Implements Dijkstra's single source shortest path algorithm. This
 * implementation is based on the algorithms presented by [Harish07]
 * P. Harish and P. Narayanan, "Accelerating large graph algorithms on the GPU
 * using CUDA," in High Performance Computing - HiPC 2007, LNCS v. 4873, ch. 21,
 * doi: http://dx.doi.org/10.1007/978-3-540-77220-0_21
 *
 *  Created on: 2011-03-04
 *      Author: Elizeu Santos-Neto (elizeus@ece.ubc.ca)
 */

// system includes
#include <cuda.h>

// totem includes
#include "totem_comdef.h"
#include "totem_comkernel.cuh"
#include "totem_graph.h"
#include "totem_mem.h"

/**
 * Tests whether any of the array elements is set to true.
 * @param[in] array a boolean array
 * @param[in] size the number of element in the array
 * @param[out] result indicates whether the kernel found a true element.
 */
__global__
void has_true_kernel(bool* array, uint32_t size, bool* result) {
  const id_t vertex_id = THREAD_GLOBAL_INDEX;
  *result = false;
  if (vertex_id >= size) {    
    return;
  }
  if (array[vertex_id]) {
    *result = true;
  }
}

/**
 * Computes the new distances for each neighbor in the graph.
 * @param[in] graph the input graph used to compute the distances
 * @param[in] to_update an array to indicate which nodes will update distances
 * @param[in] distances an array that contains the current state of distances
 * @param[in] mutex a mutex variable used to implement an atomicMin.
 * @param[out] new_distances an array with distances updated in this round
 */
__global__
void dijkstra_kernel(graph_t graph, bool* to_update, weight_t* distances,
                     weight_t* new_distances, uint32_t* mutex) {

  // get direct access to graph members
  id_t  vertex_count = graph.vertex_count;
  id_t* vertices     = graph.vertices;
  id_t* edges        = graph.edges;
  weight_t* weights  = graph.weights;

  // TODO(abdullah): May be there is an opportunity for optimization here.
  //                 Threads (vertices) that do not have the to_update set will
  //                 exit at this point, and they will stay idle until their
  //                 mates are done. I was wondering how we can keep them busy.
  //                 For BFS, the Stanford paper uses a work queue and they get
  //                 marginal improvement. It is not clear if SSSP will have the
  //                 same behavior.
  const id_t vertex_id = THREAD_GLOBAL_INDEX;
  if ((vertex_id >= vertex_count) || !to_update[vertex_id]) {
    return;
  }
  to_update[vertex_id] = false;

  id_t* neighbors = &(edges[vertices[vertex_id]]);
  weight_t* local_weights = &(weights[vertices[vertex_id]]);
  uint64_t neighbor_count = vertices[vertex_id + 1] - vertices[vertex_id];
  weight_t distance_to_vertex = distances[vertex_id];

  for (id_t i = 0; i < neighbor_count; i++) {
    id_t neighbor_id = neighbors[i];
    weight_t current_distance = distance_to_vertex + local_weights[i];
    // TODO(elizeu): This mutex is inefficient, as it serializes all threads.
    //               One approach to solve this is to have one mutex per vertex
    //               that indicates whether the position in the new_distance
    //               array regarding that vertex is locked or open.
    while(!atomicCAS(mutex, 1, 0));
    weight_t* new_distance = &(new_distances[neighbor_id]);
    if (current_distance < *new_distance) {
      *new_distance = current_distance;
    }
    atomicCAS(mutex, 0, 1);
  } // for
}

/**
 * Make the new distances permanent if the new distances are smaller than
 * current distances.
 * @param[in] graph the input graph used to compute the distances
 * @param[in] to_update an array to indicate which nodes will update distances
 * @param[in] distances an array that contains the current state of distances
 * @param[in] mutex a mutex variable used to implement an atomicMin.
 * @param[out] new_distances an array with distances updated in this round
 */
__global__
void dijkstra_final_kernel(graph_t graph, bool* to_update, weight_t* distances,
                           weight_t* new_distances) {
  const uint32_t vertex_id = THREAD_GLOBAL_INDEX;
  if (vertex_id >= graph.vertex_count) {
    return;
  }
  if (new_distances[vertex_id] < distances[vertex_id]) {
    distances[vertex_id] = new_distances[vertex_id];
    to_update[vertex_id] = true;
  }
  new_distances[vertex_id] = distances[vertex_id];
}

error_t dijkstra_gpu(graph_t* graph, id_t source_id,
                     weight_t** shortest_distances) {
  // TODO(elizeu): Move input validations to a common separate function.
  // Validate input parameters
  if ((graph == NULL) || !graph->weighted
      || (source_id >= graph->vertex_count)) {
    *shortest_distances = NULL;
    return FAILURE;
  } else if(graph->vertex_count == 1) {
    *shortest_distances = (weight_t*)mem_alloc(sizeof(weight_t));
    (*shortest_distances)[0] = 0;
    return SUCCESS;
  }

  // Check whether the graph has vertices, but an empty edge set.
  if ((graph->vertex_count > 0) && (graph->edge_count == 0)) {
    *shortest_distances =
      (weight_t*)mem_alloc(graph->vertex_count * sizeof(weight_t));
    for (id_t node_id = 0; node_id < graph->vertex_count; node_id++) {
      (*shortest_distances)[node_id] = WEIGHT_MAX;
    }
    (*shortest_distances)[source_id] =  (weight_t)0.0;
    return SUCCESS;
  }

  // Kernel configuration parameters.
  dim3 block_count;
  dim3 threads_per_block;

  graph_t* graph_d;
  // Allocate and transfer the vertex array to the device.
  CHK_SUCCESS(graph_initialize_device(graph, &graph_d), err);

  // The distance array from the source vertex to every nother in the graph.
  weight_t* distances_d;
  CHK_CU_SUCCESS(cudaMalloc((void **)&distances_d,
                       graph_d->vertex_count * sizeof(weight_t)),
            err_free_graph);

  // An array that contains the newly computed array of distances.
  weight_t* new_distances_d;
  CHK_CU_SUCCESS(cudaMalloc((void **)&new_distances_d,
                       graph_d->vertex_count * sizeof(weight_t)),
            err_free_distances);

  // An entry in this array indicate whether the corresponding vertex should
  // try to compute new distances.
  bool* changed_d;
  CHK_CU_SUCCESS(cudaMalloc((void **)&changed_d, graph_d->vertex_count
                       * sizeof(bool)), err_free_new_distances);

  // Compute the number of blocks.
  KERNEL_CONFIGURE(graph_d->vertex_count, block_count, threads_per_block);

  // Initialize the mutex used in the kernel to avoid the race condition.
  // TODO(elizeu): We may want to move this feature into a separate file if
  //               atomic-*() functions that receive floating point arguments
  //               become common.
  uint32_t* mutex_d;
  CHK_CU_SUCCESS(cudaMalloc((void **)&mutex_d, sizeof(uint32_t)),
                 err_free_new_distances);

  // Initialize the mutex.
  CHK_CU_SUCCESS(cudaMemset(mutex_d, 1, sizeof(uint32_t)), err_free_mutex);

  // Set all distances to infinite.
  memset_device<<<block_count, threads_per_block>>>(distances_d, WEIGHT_MAX,
                                                    graph_d->vertex_count);
  memset_device<<<block_count, threads_per_block>>>(new_distances_d, WEIGHT_MAX,
                                                    graph_d->vertex_count);

  // Set the distance to the source to zero.
  CHK_CU_SUCCESS(cudaMemset(&(distances_d[source_id]), (weight_t)0,
                 sizeof(weight_t)), err_free_mutex);

  // Activate the source vertex to compute distances.
  CHK_CU_SUCCESS(cudaMemset(&(changed_d[source_id]), true, sizeof(bool)),
                 err_free_mutex);

  // Compute the distances update
  bool has_true;
  has_true = true;
  bool* has_true_d;
  CHK_CU_SUCCESS(cudaMalloc((void **)&has_true_d, sizeof(bool)),
            err_free_has_true);
  CHK_CU_SUCCESS(cudaMemset(has_true_d, false, sizeof(bool)),
            err_free_has_true);

  while (has_true) {
    dijkstra_kernel<<<block_count, threads_per_block>>>
      (*graph_d, changed_d, distances_d, new_distances_d, mutex_d);
    dijkstra_final_kernel<<<block_count, threads_per_block>>>
      (*graph_d, changed_d, distances_d, new_distances_d);
    has_true_kernel<<<block_count, threads_per_block>>>
      (changed_d, graph_d->vertex_count, has_true_d);
    CHK_CU_SUCCESS(cudaMemcpy(&has_true, has_true_d, sizeof(bool),
                              cudaMemcpyDeviceToHost), err_free_has_true);
  }

  // Copy the pointer to the output parameter
  weight_t* local_shortest_distances;
  local_shortest_distances =
    (weight_t*)mem_alloc(graph_d->vertex_count * sizeof(weight_t));
  CHK_CU_SUCCESS(cudaMemcpy(local_shortest_distances, distances_d,
                       graph_d->vertex_count * sizeof(weight_t),
                       cudaMemcpyDeviceToHost),
                       err_free_has_true);
  *shortest_distances = local_shortest_distances;

  // Release the allocated memory
  graph_finalize_device(graph_d);
  cudaFree(distances_d);
  cudaFree(changed_d);
  cudaFree(mutex_d);
  cudaFree(new_distances_d);

  return SUCCESS;

  // error handlers
  err_free_has_true:
    cudaFree(has_true_d);
  err_free_mutex:
    cudaFree(mutex_d);
  err_free_new_distances:
    cudaFree(new_distances_d);
  err_free_distances:
    cudaFree(distances_d);
  err_free_graph:
    graph_finalize_device(graph_d);
  err:
    return FAILURE;
}

__host__
error_t dijkstra_cpu(graph_t* graph, id_t source_id,
                     weight_t** shortest_distances) {
  // Validate input parameters
  if ((graph == NULL) || !graph->weighted
      || (source_id >= graph->vertex_count)) {
    *shortest_distances = NULL;
    return FAILURE;
  } else if (graph->vertex_count == 1) {
    *shortest_distances = (weight_t*)mem_alloc(sizeof(weight_t));
    (*shortest_distances)[0] = 0;
    return SUCCESS;
  }

  // Initialize the shortest_distances to infinite  
  *shortest_distances =
    (weight_t*)mem_alloc(graph->vertex_count * sizeof(weight_t));

  // An entry in this array indicate whether the corresponding vertex should
  // try to update the current distances.
  bool* to_update = (bool *)mem_alloc(graph->vertex_count * sizeof(bool));

  for (id_t vertex_id = 0; vertex_id < graph->vertex_count; vertex_id++) {
    (*shortest_distances)[vertex_id] = WEIGHT_MAX;
    to_update[vertex_id] = false;
  }
  (*shortest_distances)[source_id] =  (weight_t)0.0;
  to_update[source_id] = true;

  // Check whether the graph has vertices, but an empty edge set.
  if ((graph->vertex_count > 0) && (graph->edge_count == 0)) {
    mem_free(to_update);
    return SUCCESS;
  }

  // get direct access to graph members
  id_t  vertex_count = graph->vertex_count;
  id_t* vertices     = graph->vertices;
  id_t* edges        = graph->edges;
  weight_t* weights  = graph->weights;

  // Initialize the mutex.
  // TODO(elizeu): This line generates a "unreferenced variable" warning,
  //               even though the variable is referenced implicitely by
  //               omp below. We need to find a way to disable/enable it.
  int mutex = 0;
  mutex = mutex + 0;

  bool changed = true;
  while (changed) {
    changed = false;

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif // _OPENMP
    for (id_t vertex_id = 0; vertex_id < vertex_count; vertex_id++) {
      if (!to_update[vertex_id]) {
        continue;
      }
      to_update[vertex_id] = false;

      id_t* neighbors = &edges[vertices[vertex_id]];
      weight_t* local_weights = &weights[vertices[vertex_id]];
      uint64_t neighbor_count = vertices[vertex_id + 1] - vertices[vertex_id];

      for (id_t i = 0; i < neighbor_count; i++) {
        id_t neighbor_id = neighbors[i];
        // TODO(elizeu): This global lock may be inefficient. One approach to
        //               solve this is to have one lock per vertex.
        #ifdef _OPENMP
        #pragma omp critical (mutex)
        {
        #endif // _OPENMP
        weight_t current_distance =
            (*shortest_distances)[vertex_id] + local_weights[i];
        if ((*shortest_distances)[neighbor_id] > current_distance) {
          (*shortest_distances)[neighbor_id] = current_distance;
          to_update[neighbor_id] = true;
          changed = true;
        }
        #ifdef _OPENMP
        } // critical
        #endif // _OPENMP
      } // for
    } // for
  } // while
  return SUCCESS;
}

