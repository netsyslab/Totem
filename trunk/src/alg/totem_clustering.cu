/**
 *
 *  Implements Clustering Coefficient algorithm for GPU.
 *
 *  Created on: 2014-02-03
 *  Author: Tahsin Arafat Reza 
 */

// totem includes
#include "totem_alg.h"
#include "totem_centrality.h"

/**
 * Checks for input parameters and special cases. This is invoked at the
 * beginning of public interfaces (CPU and GPU).
 *
 * @param[in] graph
 * @param[in] finished
 * @param[in] coefficients
 */
PRIVATE
error_t check_special_cases(const graph_t* graph, bool* finished,
                            weight_t** coefficients) {
  // Check whether the graph is null or vertex set is empty 
  if ((graph == NULL) || (graph->vertex_count == 0)
      || (coefficients == NULL)) {
    return FAILURE;
  }

  // Check whether the edge set is empty
  if (graph->edge_count == 0) {
    totem_malloc(graph->vertex_count * sizeof(weight_t), TOTEM_MEM_HOST_PINNED, 
                 (void**)coefficients);
    memset(*coefficients, (weight_t)0.0, graph->vertex_count
           * sizeof(weight_t));
    return SUCCESS;
  }

  *finished = false;
  return SUCCESS;
}

/**
 * Initialize GPU.
 *
 * @param[in] graph
 * @param[in] distance_length
 * @param[out] graph_d
 * @param[out] coefficients_d
 */
PRIVATE
error_t initialize_gpu(const graph_t* graph, graph_t** graph_d, 
                       weight_t** coefficients_d) { 

  totem_mem_t type = TOTEM_MEM_DEVICE;
  
  // Transfer the graph to the device memory
  CHK_SUCCESS(graph_initialize_device(graph, graph_d), err);

  // Allocate memory for the coefficients array
  CHK_SUCCESS(totem_malloc(graph->vertex_count * sizeof(weight_t), type,
                           (void**)coefficients_d), err_free_graph);
  
  // Set cofficients to zero
  totem_memset(*coefficients_d, (weight_t)0.0, graph->vertex_count, type);

  return SUCCESS;

  // Error handlers
  err_free_graph:
    graph_finalize_device(*graph_d);
  err:
    return FAILURE;    
}

/**
 * Finalize GPU.
 *
 * @param[in] graph
 * @param[in] distance_length
 * @param[out] graph_d
 * @param[out] coefficients_d
 */
PRIVATE
error_t finalize_gpu(graph_t* graph_d, weight_t* coefficients_d, 
                     weight_t* coefficients) {

  // Copy the pointer to the output paramenter
  CHK_CU_SUCCESS(cudaMemcpy(coefficients, coefficients_d, 
    graph_d->vertex_count * sizeof(weight_t), cudaMemcpyDeviceToHost), err); 
  
  // Release allocated memory for results
  totem_free(coefficients_d, TOTEM_MEM_DEVICE);
  graph_finalize_device(graph_d);
  return SUCCESS;
  
  // Error handlers
  err:
    return FAILURE;  
}

/**
 * Implements the GPU kernel function.
 *
 * @param[in] graph the input graph
 * @param[out] clustering_coefficients array containing 
 * computed coefficients 
 */
__global__
void clustering_coefficient_kernel(graph_t graph, 
                                   weight_t* clustering_coefficients) {
  const vid_t v = THREAD_GLOBAL_INDEX;
  if (v >= graph.vertex_count) return;
  
  vid_t triangle_count_v = 0;
   
  for (eid_t e = graph.vertices[v]; e < graph.vertices[v + 1]; e++) {
    vid_t e_v = graph.edges[e];  // v's neighbour  
    for (eid_t f = graph.vertices[e_v]; f < graph.vertices[e_v + 1]; f++) {
      vid_t f_e_v = graph.edges[f];  // Neighbour of v's neighbour  
      for (eid_t u = graph.vertices[v]; u < graph.vertices[v + 1]; u++) {
        vid_t u_v = graph.edges[u];  // v's neighbour 
        if (u_v != e_v && u_v == f_e_v) {  // Common neighbour verification  
          triangle_count_v++;  
        }  
      } // for 
    } // for 
  } // for

  vid_t degree_v = graph.vertices[v + 1] - graph.vertices[v]; 
  weight_t cc = 0.0f;  // Clustering Coefficient 
  weight_t triangles_v = (weight_t)triangle_count_v/2.0f;  
  if (triangles_v > 0.0f) {
    cc = (2.0f * triangles_v) / (((weight_t)degree_v - 1.0f ) * 
                                 (weight_t)degree_v);
  }
  clustering_coefficients[v] = cc; 
}
 
/**
 * Implements the GPU-only clustering coefficient algorithm.
 *
 * Given a graph \f$G = (V, E)\f$, the clustering coefficient (\f$CC\f$) 
 * of a vertex \f$v\inV\f$ with degree \f$d\f$ is defined as 
 * \f$CC = 2*T / d(d-1)f$, where \f$Tf$ is the number of triangles 
 * incident on \f$v\f$.    
 *
 * This algorithm utilizes GPU cores according to kernel launch 
 * configuration (number of blocks, threads per block etc.). Each vertex 
 * performs computation (defined by the kernel function) in a unique thread. 
 * Computaions of triangle counting and clustering coeffienct involves 
 * writing to thread local variables only. Therefore, vertices can carryout 
 * computation in parallel without interrupting each other. Each vertex 
 * iterates through the list of its neighbours (and the list of neighbours 
 * of each of its neighbours) in a sequential manner, to verify presence of 
 * a common neigbour; hence, count trainagles and calculate clustering 
 * coefficeint.
 *
 * @param[in] graph the input graph
 * @param[out] coefficients array containing computed coefficients 
 */
error_t clustering_coefficient_gpu(const graph_t* graph, 
                                   weight_t** coefficients) {
  
  // Check inputs
  bool finished = true;
  error_t rc = check_special_cases(graph, &finished, coefficients);
  if (finished) return rc;  

  // Initialize GPU states  
  graph_t* graph_d;
  weight_t* coefficients_d;
  CHK_SUCCESS(initialize_gpu(graph, &graph_d, &coefficients_d), err);
  {
  dim3 block_count, threads_per_block;
  KERNEL_CONFIGURE(graph->vertex_count, block_count, threads_per_block);
  clustering_coefficient_kernel<<<block_count, threads_per_block>>>
    (*graph_d, coefficients_d);
  }
  
  // Finalize GPU states
  // Copy the calculated coefficients from the 
  // device memory to the host memory  
  CHK_SUCCESS(finalize_gpu(graph_d, coefficients_d, *coefficients), 
    err_free_all);
 
  return SUCCESS;

  // Error handlers
  err_free_all:
    totem_free(coefficients_d, TOTEM_MEM_DEVICE);
    graph_finalize_device(graph_d);
  err:
    return FAILURE;  
}

/**
 * Implements the CPU-only clustering coefficient algorithm.
 *
 * Given a graph \f$G = (V, E)\f$, the clustering coefficient (\f$CC\f$) 
 * of a vertex \f$v\inV\f$ with degree \f$d\f$ is defined as 
 * \f$CC = 2*T / d(d-1)f$, where \f$Tf$ is the number of triangles 
 * incident on \f$v\f$.    
 *
 * The outmost loop is parallelized with OpenMP. Each vertex performs 
 * computation in a unique thread. Computaions of triangle
 * counting and clustering coeffienct involves writing to thread 
 * local variables only. Therefore, vertices can carryout computation in 
 * parallel without interrupting each other. Each vertex iterates through 
 * the list of its neighbours (and the list of neighbours of each of its 
 * neighbours) in a sequential manner, to verify presence of a common neigbour; 
 * hence, count trainagles and calculate clustering coefficeint.           
 *
 * @param[in] graph the input graph
 * @param[out] coefficients array containing computed coefficients 
 */
error_t clustering_coefficient_cpu(const graph_t* graph,
                                   weight_t** coefficients) {
  // Check inputs
  bool finished = true;
  error_t rc = check_special_cases(graph, &finished, coefficients);
  if (finished) return rc;

  // Allocate memory for the results
  weight_t* clustering_coefficients = NULL;
  totem_malloc(graph->vertex_count * sizeof(weight_t), TOTEM_MEM_HOST_PINNED, 
               (void**)&clustering_coefficients);
  
  memset(clustering_coefficients, 0.0, graph->vertex_count * sizeof(weight_t));

  OMP(omp parallel for)
  for (vid_t v = 0; v < graph->vertex_count; v++) {
    vid_t triangle_count_v = 0;
    for (eid_t e = graph->vertices[v]; e < graph->vertices[v + 1]; e++) {
      vid_t e_v = graph->edges[e];  // v's neighbour  
      for (eid_t f = graph->vertices[e_v]; f < graph->vertices[e_v + 1]; f++) {
        vid_t f_e_v = graph->edges[f];  // Neighbour of v's neighbour  
        for (eid_t u = graph->vertices[v]; u < graph->vertices[v + 1]; u++) {
          vid_t u_v = graph->edges[u]; // v's neighbour  
          if (u_v != e_v && u_v == f_e_v) {  // Common neighbour verification  
            triangle_count_v++;
          }
        } // for
      } // for
    } // for

    vid_t degree_v = graph->vertices[v + 1] - graph->vertices[v]; 

    clustering_coefficients[v] = 0.0f;

    weight_t triangles_v = (weight_t)triangle_count_v/2.0f; 

    if (triangles_v > 0.0f) {      
      clustering_coefficients[v] = 
        (2.0f * triangles_v) / (((weight_t)degree_v - 1.0f) * 
                                (weight_t)degree_v);
    }
  } // parallel for 
   
  *coefficients = clustering_coefficients;
  return SUCCESS;
}
