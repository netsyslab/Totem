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

// totem includes
#include "totem_bitmap.cuh"
#include "totem_comdef.h"
#include "totem_comkernel.cuh"
#include "totem_graph.h"
#include "totem_mem.h"

/**
   This structure is used by the virtual warp-based implementation. It stores a
   batch of work. It is typically allocated on shared memory and is processed by
   a single virtual warp.
 */
 // TODO(elizeu): Consider moving the edges weights of the neighbors to this
 // structure. It might be tricky to keep alignment or even to fit the data
 // into shared memory.
typedef struct {
  id_t vertices[VWARP_BATCH_SIZE + 1];
  weight_t distances[VWARP_BATCH_SIZE];
  bool to_update[VWARP_BATCH_SIZE];
  // the following ensures 64-bit alignment, it assumes that the cost and
  // vertices arrays are of 32-bit elements.
  // TODO(abdullah) a portable way to do this (what if id_t is 64-bit?)
  int pad;
} vwarp_mem_t;


/**
 * Checks for input parameters and special cases. This is invoked at the
 * beginning of public interfaces (GPU and CPU)
*/
PRIVATE
error_t check_special_cases(const graph_t* graph, id_t source_id,
                            weight_t **shortest_distances, bool* finished) {
  *finished = true;
  if ((graph == NULL) || !graph->weighted
      || (source_id >= graph->vertex_count)) {
    *shortest_distances = NULL;
    return FAILURE;
  } else if (graph->vertex_count == 1) {
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

  *finished = false;
  return SUCCESS;
}


/**
 * A common initialization function for GPU implementations of Dijkstra's
 * algorithm. It allocates memory in the device and initalizes state on the GPU.
*/
PRIVATE
error_t initialize_gpu(const graph_t* graph, id_t source_id,
                       uint64_t distance_length, graph_t** graph_d,
                       bool** changed_d, bool** has_true_d,
                       weight_t** distances_d, weight_t** new_distances_d) {

  // Kernel configuration parameters.
  dim3 block_count;
  dim3 threads_per_block;

  // Allocate and transfer the vertex array to the device.
  CHK_SUCCESS(graph_initialize_device(graph, graph_d), err);

  // The distance array from the source vertex to every node in the graph.
  CHK_CU_SUCCESS(cudaMalloc((void**)distances_d,
                            distance_length * sizeof(weight_t)),
                 err_free_graph);

  // An array that contains the newly computed array of distances.
  CHK_CU_SUCCESS(cudaMalloc((void**)new_distances_d,
                            distance_length * sizeof(weight_t)),
                 err_free_distances);

  // An entry in this array indicate whether the corresponding vertex should
  // try to compute new distances.
  CHK_CU_SUCCESS(cudaMalloc((void **)changed_d, distance_length * sizeof(bool)),
                 err_free_new_distances);

  // Compute the number of blocks.
  KERNEL_CONFIGURE(distance_length, block_count, threads_per_block);

  // Set all distances to infinite.
  memset_device<<<block_count, threads_per_block>>>
      (*distances_d, WEIGHT_MAX, distance_length);
  memset_device<<<block_count, threads_per_block>>>
      (*new_distances_d, WEIGHT_MAX, distance_length);

  // Set the distance to the source to zero.
  CHK_CU_SUCCESS(cudaMemset(&((*distances_d)[source_id]), (weight_t)0,
                 sizeof(weight_t)), err_free_new_distances);

  // Activate the source vertex to compute distances.
  CHK_CU_SUCCESS(cudaMemset(*changed_d, false, distance_length * sizeof(bool)),
                 err_free_new_distances);
  CHK_CU_SUCCESS(cudaMemset(&((*changed_d)[source_id]), true, sizeof(bool)),
                 err_free_new_distances);

  // Initialize the flags that indicate whether the distances were updated
  CHK_CU_SUCCESS(cudaMalloc((void **)has_true_d, sizeof(bool)),
            err_free_new_distances);
  CHK_CU_SUCCESS(cudaMemset(*has_true_d, false, sizeof(bool)),
            err_free_all);

  return SUCCESS;

  // error handlers
  err_free_all:
    cudaFree(*has_true_d);
  err_free_new_distances:
    cudaFree(*new_distances_d);
  err_free_distances:
    cudaFree(*distances_d);
  err_free_graph:
    graph_finalize_device(*graph_d);
  err:
    return FAILURE;
}


/**
 * A common finalize function for GPU implementations. It allocates the host
 * output buffer, moves the final results from GPU to the host buffers and
 * frees up some resources.
*/
PRIVATE
error_t finalize_gpu(graph_t* graph_d, weight_t* distances_d, bool* changed_d,
                     weight_t* new_distances_d, weight_t** shortest_distances) {

  // Copy the pointer to the output parameter
  *shortest_distances =
      (weight_t*)mem_alloc(graph_d->vertex_count * sizeof(weight_t));
  CHK_CU_SUCCESS(cudaMemcpy(*shortest_distances, distances_d,
                            graph_d->vertex_count * sizeof(weight_t),
                            cudaMemcpyDeviceToHost), err);

  // Release the allocated memory
  cudaFree(distances_d);
  cudaFree(changed_d);
  cudaFree(new_distances_d);
  graph_finalize_device(graph_d);
  return SUCCESS;

 err:
  mem_free(shortest_distances);
  *shortest_distances = NULL;
  return FAILURE;
}

/**
 * Computes the new distances for each neighbor in the graph.
 * @param[in] graph the input graph used to compute the distances
 * @param[in] to_update an array to indicate which nodes will update distances
 * @param[in] distances an array that contains the current state of distances
 * @param[out] new_distances an array with distances updated in this round
 */
__global__
void dijkstra_kernel(graph_t graph, bool* to_update, weight_t* distances,
                     weight_t* new_distances) {

  // get direct access to graph members
  id_t  vertex_count = graph.vertex_count;
  id_t* vertices     = graph.vertices;
  id_t* edges        = graph.edges;
  weight_t* weights  = graph.weights;

  const id_t vertex_id = THREAD_GLOBAL_INDEX;
  if ((vertex_id >= vertex_count) || !to_update[vertex_id]) {
    return;
  }

  id_t* neighbors = &(edges[vertices[vertex_id]]);
  weight_t* local_weights = &(weights[vertices[vertex_id]]);
  id_t neighbor_count = vertices[vertex_id + 1] - vertices[vertex_id];
  weight_t distance_to_vertex = distances[vertex_id];

  for (id_t i = 0; i < neighbor_count; i++) {
    id_t neighbor_id = neighbors[i];
    weight_t current_distance = distance_to_vertex + local_weights[i];
    atomicMin(&(new_distances[neighbor_id]), current_distance);
  } // for
}


/**
 * The neighbors processing function. This function computes the distance from
 * the source node to the each of the neighbors to the current vertex. The
 * assumption is that the threads of a warp invoke this function to process the
 * warp's batch of work. In each iteration of the for loop, each thread
 * processes a neighbor. For example, thread 0 in the warp processes neighbors
 * at indices 0, VWARP_WARP_SIZE, (2 * VWARP_WARP_SIZE) etc. in the edges array,
 * while thread 1 in the warp processes neighbors 1, (1 + VWARP_WARP_SIZE),
 * (1 + 2 * VWARP_WARP_SIZE) and so on.
*/
inline __device__
void vwarp_process_neighbors(int warp_offset, id_t neighbor_count,
                             id_t* neighbors, weight_t* weights,
                             weight_t distance_to_vertex,
                             weight_t* new_distances) {
  for(int i = warp_offset; i < neighbor_count; i += VWARP_WARP_SIZE) {
    id_t neighbor_id = neighbors[i];
    weight_t current_distance = distance_to_vertex + weights[i];
    atomicMin(&(new_distances[neighbor_id]), current_distance);
  } // for
}


/**
 * An implementation of the Dijkstra kernel that implements the virtual warp
 * technique.
 */
__global__
void vwarp_dijkstra_kernel(graph_t graph, bool* to_update, weight_t* distances,
                           weight_t* new_distances, uint32_t thread_count) {

  if (THREAD_GLOBAL_INDEX >= thread_count) return;

  int warp_offset = THREAD_GLOBAL_INDEX % VWARP_WARP_SIZE;
  int warp_id     = THREAD_GLOBAL_INDEX / VWARP_WARP_SIZE;

  __shared__ vwarp_mem_t shared_memory[(MAX_THREADS_PER_BLOCK
                                        / VWARP_WARP_SIZE)];
  vwarp_mem_t* my_space = shared_memory + (THREAD_GRID_INDEX / VWARP_WARP_SIZE);

  // copy my work to local space
  int v_ = warp_id * VWARP_BATCH_SIZE;
  vwarp_memcpy(my_space->distances, &distances[v_], VWARP_BATCH_SIZE,
               warp_offset);
  vwarp_memcpy(my_space->vertices, &(graph.vertices[v_]), VWARP_BATCH_SIZE + 1,
               warp_offset);
  vwarp_memcpy(my_space->to_update, &(to_update[v_]), VWARP_BATCH_SIZE,
               warp_offset);

  // iterate over my work
  for(uint32_t v = 0; v < VWARP_BATCH_SIZE; v++) {
    weight_t distance_to_vertex = my_space->distances[v];
    if (my_space->to_update[v]) {
      id_t* neighbors = &(graph.edges[my_space->vertices[v]]);
      weight_t* local_weights = &(graph.weights[my_space->vertices[v]]);
      id_t neighbor_count = my_space->vertices[v + 1] - my_space->vertices[v];
      vwarp_process_neighbors(warp_offset, neighbor_count, neighbors,
                              local_weights, distance_to_vertex, new_distances);
    }
  }
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
                           weight_t* new_distances, bool* has_true) {
  const uint32_t vertex_id = THREAD_GLOBAL_INDEX;
  if (vertex_id >= graph.vertex_count) {
    return;
  }
  if (new_distances[vertex_id] < distances[vertex_id]) {
    distances[vertex_id] = new_distances[vertex_id];
    to_update[vertex_id] = true;
    *has_true = true;
  }
  new_distances[vertex_id] = distances[vertex_id];
}

error_t dijkstra_gpu(const graph_t* graph, id_t source_id,
                     weight_t** shortest_distances) {
  // Check for special cases
  bool finished = false;
  error_t rc = check_special_cases(graph, source_id, shortest_distances,
                                   &finished);
  if (finished) return rc;

  // Allocate and initialize GPU state
  bool *changed_d;
  bool *has_true_d;
  graph_t* graph_d;
  weight_t* distances_d;
  weight_t* new_distances_d;
  CHK_SUCCESS(initialize_gpu(graph, source_id, graph->vertex_count, &graph_d,
                             &changed_d, &has_true_d, &distances_d,
                             &new_distances_d), err);

  {
  dim3 block_count, threads_per_block;
  KERNEL_CONFIGURE(graph->vertex_count, block_count, threads_per_block);
  bool has_true = true;
  while (has_true) {
    dijkstra_kernel<<<block_count, threads_per_block>>>
      (*graph_d, changed_d, distances_d, new_distances_d);
    CHK_CU_SUCCESS(cudaMemset(changed_d, false, graph->vertex_count *
                              sizeof(bool)), err_free_all);
    CHK_CU_SUCCESS(cudaMemset(has_true_d, false, sizeof(bool)), err_free_all);
    dijkstra_final_kernel<<<block_count, threads_per_block>>>
      (*graph_d, changed_d, distances_d, new_distances_d, has_true_d);
    CHK_CU_SUCCESS(cudaMemcpy(&has_true, has_true_d, sizeof(bool),
                              cudaMemcpyDeviceToHost), err_free_all);
  }
  }

  // Copy the output shortest distances from the device mem to the host
  // Finalize GPU
  CHK_SUCCESS(finalize_gpu(graph_d, distances_d, changed_d, new_distances_d,
                      shortest_distances), err_free_all);
  return SUCCESS;

  // error handlers
  err_free_all:
   cudaFree(changed_d);
   cudaFree(has_true_d);
   cudaFree(distances_d);
   cudaFree(new_distances_d);
   graph_finalize_device(graph_d);
  err:
    return FAILURE;
}

error_t dijkstra_vwarp_gpu(const graph_t* graph, id_t source_id,
                           weight_t** shortest_distances) {

  // Check for special cases
  bool finished = false;
  error_t rc = check_special_cases(graph, source_id, shortest_distances,
                                   &finished);
  if (finished) return rc;

  // Allocate and initialize GPU state
  bool *changed_d;
  bool *has_true_d;
  graph_t* graph_d;
  weight_t* distances_d;
  weight_t* new_distances_d;
  uint64_t distance_length;
  distance_length = VWARP_BATCH_SIZE * VWARP_BATCH_COUNT(graph->vertex_count);
  CHK_SUCCESS(initialize_gpu(graph, source_id, distance_length, &graph_d,
                             &changed_d, &has_true_d, &distances_d,
                             &new_distances_d), err);

  {
  bool has_true = true;
  dim3 block_count, threads_per_block;
  int thread_count = VWARP_WARP_SIZE * VWARP_BATCH_COUNT(graph->vertex_count);
  KERNEL_CONFIGURE(thread_count, block_count, threads_per_block);
  cudaFuncSetCacheConfig(vwarp_dijkstra_kernel, cudaFuncCachePreferShared);
  dim3 block_count_final, threads_per_block_final;
  KERNEL_CONFIGURE(graph->vertex_count, block_count_final,
                   threads_per_block_final);
  while (has_true) {
    vwarp_dijkstra_kernel<<<block_count, threads_per_block>>>
      (*graph_d, changed_d, distances_d, new_distances_d, thread_count);
    CHK_CU_SUCCESS(cudaMemset(changed_d, false, distance_length * sizeof(bool)),
                   err_free_all);
    CHK_CU_SUCCESS(cudaMemset(has_true_d, false, sizeof(bool)), err_free_all);
    dijkstra_final_kernel<<<block_count_final, threads_per_block_final>>>
      (*graph_d, changed_d, distances_d, new_distances_d, has_true_d);
    CHK_CU_SUCCESS(cudaMemcpy(&has_true, has_true_d, sizeof(bool),
                              cudaMemcpyDeviceToHost), err_free_all);
  }
  }

  // Finalize GPU
  CHK_SUCCESS(finalize_gpu(graph_d, distances_d, changed_d, new_distances_d,
                      shortest_distances), err_free_all);
  return SUCCESS;

  // error handlers
  err_free_all:
   cudaFree(changed_d);
   cudaFree(has_true_d);
   cudaFree(distances_d);
   cudaFree(new_distances_d);
   graph_finalize_device(graph_d);
  err:
    return FAILURE;
}

__host__ error_t dijkstra_cpu(const graph_t* graph, id_t source_id,
                              weight_t** shortest_distances) {
  // Check for special cases
  bool finished = false;
  error_t rc = check_special_cases(graph, source_id, shortest_distances,
                                   &finished);
  if (finished) return rc;

  // Initialize the shortest_distances to infinite
  *shortest_distances = 
    (weight_t*)mem_alloc(graph->vertex_count * sizeof(weight_t));
  OMP(omp parallel for)
  for (id_t vertex_id = 0; vertex_id < graph->vertex_count; vertex_id++) {
    (*shortest_distances)[vertex_id] = WEIGHT_MAX;
  }

  // An entry in this bitmap indicates whether the corresponding vertex is
  // active and that it should try to update the distances of its neighbors
  bitmap_t active = bitmap_init_cpu(graph->vertex_count);

  // Initialize the distance of the source vertex
  (*shortest_distances)[source_id] =  (weight_t)0.0;
  bitmap_set_cpu(active, source_id);

  finished = false;
  while (!finished) {
    finished = true;
    OMP(omp parallel for reduction(& : finished))
    for (id_t vertex_id = 0; vertex_id < graph->vertex_count; vertex_id++) {
      if (!bitmap_is_set(active, vertex_id)) {
        continue;
      }
      bitmap_unset_cpu(active, vertex_id);
      
      for (id_t i = graph->vertices[vertex_id]; 
           i < graph->vertices[vertex_id + 1]; i++) {
        const id_t neighbor_id = graph->edges[i];
        weight_t new_distance = (*shortest_distances)[vertex_id] + 
          graph->weights[i];
        weight_t old_distance =
          __sync_fetch_and_min_float(&((*shortest_distances)[neighbor_id]),
                                       new_distance);
        if (new_distance < old_distance) {
          bitmap_set_cpu(active, neighbor_id);
          finished = false;
        }
      } // for
    } // for
  } // while
  bitmap_finalize_cpu(active);
  return SUCCESS;
}

