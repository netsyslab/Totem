/* TODO(lauro,abdullah,elizeu): Add license.
 *
 * All pairs shortest path algorithm based on the Floyd-Warshall algorithm for
 * the CPU implementation.
 * According to [Harish07], running Dijkstra's algorithm for every vertex on the
 * graph, as compared to a parallel Floyd-Warshall algorithm, was both more
 * memory efficient (O(V) as compared to O(V^2)) and faster.
 *
 *  Created on: 2011-11-04
 *      Author: Greg Redekop
 */

// system includes
#include <cuda.h>

// totem includes
#include "totem_comdef.h"
#include "totem_comkernel.cuh"
#include "totem_graph.h"
#include "totem_mem.h"

// Externed function declarations for dijkstra kernel functions
__global__ void has_true_kernel(bool*, uint32_t, bool*);
__global__ void dijkstra_kernel(graph_t, bool*, weight_t*, weight_t*);
__global__ void dijkstra_final_kernel(graph_t, bool*, weight_t*, weight_t*);


/**
 * Checks for input parameters and special cases. This is invoked at the
 * beginning of public interfaces (GPU and CPU)
 */
PRIVATE
error_t check_special_cases(graph_t* graph, weight_t **distances,
                            bool* finished) {

  *finished = true;
  if ((graph == NULL) || !graph->weighted || graph->vertex_count <= 0) {
    *distances = NULL;
    return FAILURE;
  }
  else if (graph->vertex_count == 1) {
    *distances = (weight_t*)mem_alloc(sizeof(weight_t));
    (*distances)[0] = 0;
    return SUCCESS;
  }

  id_t v_count = graph->vertex_count;
  // Check whether the graph has vertices, but an empty edge set.
  if ((v_count > 0) && (graph->edge_count == 0)) {
    *distances = (weight_t*)mem_alloc(v_count * v_count * sizeof(weight_t));

    for (id_t src = 0; src < v_count; src++) {
      // Initialize path lengths to WEIGHT_MAX.
      weight_t* base = &(*distances)[src * v_count];
      for (id_t dest = 0; dest < v_count; dest++) {
        base[dest] = (weight_t)WEIGHT_MAX;
      }
      // 0 distance to oneself
      base[src] = 0;
    }
    return SUCCESS;
  }

  *finished = false;
  return SUCCESS;
}

/**
 * An initialization function for the GPU implementation of APSP. This function
 * allocates memory for use with Dijkstra's algorithm.
*/
PRIVATE
error_t initialize_gpu(graph_t* graph, uint64_t distance_length,
                       graph_t** graph_d, weight_t** distances_d,
                       weight_t** new_distances_d, bool** changed_d,
                       bool** has_true_d) {
  // Initialize the graph
  CHK_SUCCESS(graph_initialize_device(graph, graph_d), err);

  // The distance array from the source vertex to every node in the graph.
  CHK_CU_SUCCESS(cudaMalloc((void**)distances_d, distance_length *
                            sizeof(weight_t)), err_free_graph);

  // An array that contains the newly computed array of distances.
  CHK_CU_SUCCESS(cudaMalloc((void**)new_distances_d, distance_length *
                            sizeof(weight_t)), err_free_distances);
  // An entry in this array indicate whether the corresponding vertex should
  // try to compute new distances.
  CHK_CU_SUCCESS(cudaMalloc((void **)changed_d, distance_length * sizeof(bool)),
                 err_free_new_distances);
  // Initialize the flags that indicate whether the distances were updated
  CHK_CU_SUCCESS(cudaMalloc((void **)has_true_d, sizeof(bool)),
                 err_free_all);

  return SUCCESS;

  // error handlers
  err_free_all:
    cudaFree(*changed_d);
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
 * An initialization function for GPU implementations of APSP. This function
 * initializes state for running an iteration of Dijkstra's algorithm on a
 * source vertex.
*/
PRIVATE
error_t initialize_source_gpu(id_t source_id, uint64_t distance_length,
                              bool** changed_d, bool** has_true_d,
                              weight_t** distances_d,
                              weight_t** new_distances_d) {

  // Kernel configuration parameters.
  dim3 block_count;
  dim3 threads_per_block;

  // Compute the number of blocks.
  KERNEL_CONFIGURE(distance_length, block_count, threads_per_block);

  // Set all distances to infinite.
  memset_device<<<block_count, threads_per_block>>>
      (*distances_d, WEIGHT_MAX, distance_length);
  memset_device<<<block_count, threads_per_block>>>
      (*new_distances_d, WEIGHT_MAX, distance_length);

  // Set the distance to the source to zero.
  CHK_CU_SUCCESS(cudaMemset(&((*distances_d)[source_id]), (weight_t)0,
                 sizeof(weight_t)), err);

  // Activate the source vertex to compute distances.
  CHK_CU_SUCCESS(cudaMemset(&((*changed_d)[source_id]), true, sizeof(bool)),
                 err);

  // Initialize the flags that indicate whether the distances were updated
  CHK_CU_SUCCESS(cudaMemset(*has_true_d, false, sizeof(bool)), err);

  return SUCCESS;

  // error handlers
  err:
    return FAILURE;
}


/**
 * A common finalize function for GPU implementations. It frees up the GPU
 * resources.
*/
PRIVATE
void finalize_gpu(graph_t* graph, graph_t* graph_d, weight_t* distances_d,
                  bool* changed_d, weight_t* new_distances_d,
                  bool* has_true_d) {

  // Finalize GPU
  graph_finalize_device(graph_d);
  // Release the allocated memory
  cudaFree(distances_d);
  cudaFree(changed_d);
  cudaFree(new_distances_d);
  cudaFree(has_true_d);
}


/**
 * All Pairs Shortest Path algorithm on the GPU.
 */
error_t apsp_gpu(graph_t* graph, weight_t** path_ret) {
  bool finished;
  error_t ret_val = check_special_cases(graph, path_ret, &finished);
  if (finished) return ret_val;

  // The distances array mimics a static array to avoid the overhead of
  // creating an array of pointers. Thus, accessing index [i][j] will be
  // done as distances[(i * vertex_count) + j]
  weight_t* distances = (weight_t*)mem_alloc(graph->vertex_count *
                                             graph->vertex_count *
                                             sizeof(weight_t));

  // Allocate and initialize GPU state
  bool *changed_d;
  bool *has_true_d;
  graph_t* graph_d;
  weight_t* distances_d;
  weight_t* new_distances_d;
  CHK_SUCCESS(initialize_gpu(graph, graph->vertex_count, &graph_d, &distances_d,
                             &new_distances_d, &changed_d, &has_true_d), err);

  {
  dim3 block_count, threads_per_block;
  KERNEL_CONFIGURE(graph->vertex_count, block_count, threads_per_block);
  for (id_t source_id = 0; source_id < graph->vertex_count; source_id++) {
    // Run SSSP for the selected vertex
    CHK_SUCCESS(initialize_source_gpu(source_id, graph->vertex_count,
                                      &changed_d, &has_true_d, &distances_d,
                                      &new_distances_d), err_free_all);
    bool has_true = true;
    while (has_true) {
      dijkstra_kernel<<<block_count, threads_per_block>>>
        (*graph_d, changed_d, distances_d, new_distances_d);
      dijkstra_final_kernel<<<block_count, threads_per_block>>>
        (*graph_d, changed_d, distances_d, new_distances_d);
      has_true_kernel<<<block_count, threads_per_block>>>
        (changed_d, graph_d->vertex_count, has_true_d);
      CHK_CU_SUCCESS(cudaMemcpy(&has_true, has_true_d, sizeof(bool),
                                cudaMemcpyDeviceToHost), err_free_all);
    }
    // Copy the output shortest distances from the device mem to the host
    weight_t* base = &(distances[(source_id * graph->vertex_count)]);
    CHK_CU_SUCCESS(cudaMemcpy(base, distances_d, graph->vertex_count *
                              sizeof(weight_t), cudaMemcpyDeviceToHost),
                   err_free_all);

  }}

  // Free GPU resources
  finalize_gpu(graph, graph_d, distances_d, changed_d, new_distances_d,
               has_true_d);
  // Return the constructed shortest distances array
  *path_ret = distances;
  return SUCCESS;

  // error handlers
  err_free_all:
   cudaFree(changed_d);
   cudaFree(has_true_d);
   cudaFree(distances_d);
   cudaFree(new_distances_d);
   graph_finalize_device(graph_d);
  err:
   mem_free(distances);
   *path_ret = NULL;
   return FAILURE;
}

/**
 * CPU implementation of the All-Pairs Shortest Path algorithm.
 */
error_t apsp_cpu(graph_t* graph, weight_t** path_ret) {
  bool finished;
  error_t ret_val = check_special_cases(graph, path_ret, &finished);
  if (finished) return ret_val;

  uint32_t v_count = graph->vertex_count;
  // The distances array mimics a static array to avoid the overhead of
  // creating an array of pointers. Thus, accessing index [i][j] will be
  // done as distances[(i * v_count) + j]
  weight_t* distances = (weight_t*)mem_alloc(v_count * v_count *
                                             sizeof(weight_t));
  // Initialize the path cost from the edge list
  for (id_t src = 0; src < v_count; src++) {
    weight_t* base = &distances[src * v_count];
    // Initialize path lengths to WEIGHT_MAX.
    for (id_t dest = 0; dest < v_count; dest++) {
      base[dest] = (weight_t)WEIGHT_MAX;
    }
    for (id_t edge = graph->vertices[src]; edge < graph->vertices[src + 1];
         edge++) {
      base[graph->edges[edge]] = graph->weights[edge];
    }
    // 0 distance to oneself
    base[src] = 0;
  }

  // Run the main loop |V| times to converge.
  for (id_t mid = 0; mid < v_count; mid++) {
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif // _OPENMP
    for (id_t src = 0; src < v_count; src++) {
      weight_t* base = &distances[src * v_count];
      weight_t* mid_base = &distances[mid * v_count];
      for (id_t dest = 0; dest < v_count; dest++) {
        base[dest] = min(base[dest], base[mid] + mid_base[dest]);
      }
    }
  }

  *path_ret = distances;
  return SUCCESS;
}
