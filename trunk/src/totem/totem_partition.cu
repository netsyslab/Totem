/**
 * Implements the graph partitionining interface defined in totem_partition.h
 *
 *  Created on: 2011-12-29
 *  Author: Abdullah Gharaibeh
 */

// totem includes
#include "totem_comkernel.cuh"
#include "totem_mem.h"
#include "totem_partition.h"
#include "totem_util.h"

error_t partition_modularity(graph_t* graph, partition_set_t* partition_set,
                             double* modularity) {
  assert(graph && partition_set);
  if ((graph->edge_count == 0) || (partition_set->partition_count <= 1)) {
    *modularity = 0;
    return SUCCESS;
  }
  // The final modularity value
  double Q = 0.0;
  for (int p = 0; p < partition_set->partition_count; p++) {
    uint32_t local_edges = 0;
    uint32_t remote_edges = 0;
    partition_t* partition = &partition_set->partitions[p];
    graph_t* subgraph = &partition->subgraph;
    for (id_t v = 0; v < subgraph->vertex_count; v++) {
      for (id_t e = subgraph->vertices[v];
           e < subgraph->vertices[v + 1]; e++) {
        if ((uint64_t)p == GET_PARTITION_ID(subgraph->edges[e])) {
          local_edges++;
        } else {
          remote_edges++;
        }
      }
    }
    double local = local_edges / (double)graph->edge_count;
    double remote = (remote_edges * remote_edges)
                    / (double)(graph->edge_count * graph->edge_count);
    Q += local - remote;
  }
  *modularity = Q;
  return SUCCESS;
}

PRIVATE error_t partition_check(graph_t* graph, int partition_count, 
                                double* partition_fraction,
                                id_t** partition_labels) {
  *partition_labels = NULL;
  if (graph == NULL || (partition_count <= 0) || (graph->vertex_count == 0)) {
    return FAILURE;
  }
  if (graph == NULL) {
    // TODO(elizeu): Use Lauro's beautiful logging library.
    printf("ERROR: Graph object is NULL, cannot proceed with partitioning.\n");
    return FAILURE;
  }
  // The requested number of partitions should be positive
  if ((partition_count <= 0) || (graph->vertex_count == 0)) {
    printf("ERROR: Invalid number of partitions or empty graph: %d (|V|),"
           " %d (partitions).\n", graph->vertex_count, partition_count);
    return FAILURE;
  }

  if (partition_fraction != NULL) {
    // Ensure the partition fractions are >= 0.0 and add up to 1.0
    double sum = 0.0;
    for (int par_id = 0; par_id < partition_count; par_id++) {
      sum += partition_fraction[par_id];
      if (partition_fraction[par_id] < 0.0) {
        return FAILURE;
      }
    }
    // The following trick is to avoid getting stuck in precision errors
    sum = (int)(sum * 100.0);
    if (sum > 101 || sum < 99) {
      return FAILURE;
    }
  }
  return SUCCESS;
}

PRIVATE error_t partition_random(graph_t* graph, int partition_count,
                                 uint32_t seed, id_t** partition_labels) {
  // Allocate the partition vector
  id_t* partitions = (id_t*)malloc((graph->vertex_count) * sizeof(id_t));

  // Initialize the random number generator
  srand(seed);

  for (uint64_t vertex_id = 0; vertex_id < graph->vertex_count; vertex_id++) {
    // Assign each vertex to a random partition within the range
    // (0, PARTITION_COUNT - 1)
    partitions[vertex_id] = rand() % partition_count;
  }
  *partition_labels = partitions;
  return SUCCESS;
}

error_t partition_random(graph_t* graph, int partition_count,
                         double* partition_fraction, uint32_t seed, 
                         id_t** partition_labels) {
  // Check pre-conditions
  if (partition_check(graph, partition_count, partition_fraction,
                      partition_labels) == FAILURE) {
    return FAILURE;
  }

  // Check if the client is asking for equal divide among partitions
  if (partition_fraction == NULL) {
    return partition_random(graph, partition_count, seed, partition_labels);
  }

  // Allocate the partition vector
  id_t* partitions = (id_t*)malloc(graph->vertex_count * sizeof(id_t));
  assert(partitions != NULL);

  // Initialize the random number generator
  srand(seed);

  // Allocate all the partition ids to the id vector
  id_t v = 0;
  for (int pid = 0; pid < partition_count; pid++) {
    uint64_t end = (pid == partition_count - 1) ? graph->vertex_count :
      v + ((double)graph->vertex_count * partition_fraction[pid]);
    for (; v < end; v++) {
      partitions[v] = pid;
    }
  }

  /* Randomize the vector to achieve a random distribution. This is using the
   * Fisher-Yates "Random permutation" algorithm */
  for (uint64_t i = graph->vertex_count - 1; i > 0; i--) {
    uint64_t j = rand() % (i + 1);
    id_t temp = partitions[i];
    partitions[i] = partitions[j];
    partitions[j] = temp;
  }

  *partition_labels = partitions;
  return SUCCESS;
}

PRIVATE error_t init_allocate_struct_space(graph_t* graph, int pcount,
                                           size_t msg_size,
                                           partition_set_t** pset) {
  *pset = (partition_set_t*)calloc(1, sizeof(partition_set_t));
  assert(*pset);
  (*pset)->partitions = (partition_t*)calloc(pcount, sizeof(partition_t));
  assert((*pset)->partitions);
  (*pset)->id_in_partition = (id_t*)calloc(graph->vertex_count, sizeof(id_t));
  assert((*pset)->id_in_partition);
  (*pset)->graph = graph;
  (*pset)->partition_count = pcount;
  (*pset)->msg_size = msg_size;
  (*pset)->weighted = graph->weighted;
  return SUCCESS;
}

PRIVATE void init_compute_partitions_sizes(partition_set_t* pset,
                                           id_t* plabels) {
  graph_t* graph = pset->graph;
  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (id_t vid = 0; vid < graph->vertex_count; vid++) {
    id_t nbr_count = graph->vertices[vid + 1] - graph->vertices[vid];
    int pid = plabels[vid];
    partition_t* partition = &(pset->partitions[pid]);
    __sync_fetch_and_add(&(partition->subgraph.vertex_count), 1);
    __sync_fetch_and_add(&(partition->subgraph.edge_count), nbr_count);
  }
}

PRIVATE void init_allocate_partitions_space(partition_set_t* pset) {
  for (int pid = 0; pid < pset->partition_count; pid++) {
    partition_t* partition = &pset->partitions[pid];
    graph_t* subgraph = &partition->subgraph;
    if (subgraph->vertex_count > 0) {
      subgraph->vertices =
        (id_t*)malloc(sizeof(id_t) * (subgraph->vertex_count + 1));
      assert(subgraph->vertices);
      partition->map = (id_t*)calloc(subgraph->vertex_count, sizeof(id_t));
      if (subgraph->edge_count > 0) {
        subgraph->edges = (id_t*)malloc(sizeof(id_t) * subgraph->edge_count);
        assert(subgraph->edges);
        if (pset->graph->weighted) {
          subgraph->weights = (weight_t*)malloc(sizeof(weight_t) *
                                                subgraph->edge_count);
          assert(subgraph->weights);
        }
      }
    }
  }
}

PRIVATE void init_build_map(partition_set_t* pset, id_t* plabels) {
  // Reset the vertex and edge count, will be set again while building the map
  for (int pid = 0; pid < pset->partition_count; pid++) {
    pset->partitions[pid].subgraph.vertex_count = 0;
  }
  for (id_t vid = 0; vid < pset->graph->vertex_count; vid++) {
    id_t pid = plabels[vid];
    graph_t* subgraph = &pset->partitions[pid].subgraph;
     // forward map
    pset->id_in_partition[vid] = SET_PARTITION_ID(subgraph->vertex_count, pid);
    pset->partitions[pid].map[subgraph->vertex_count] = vid; // reverse map
    subgraph->vertex_count++;
  }
}

PRIVATE void init_build_partitions(partition_set_t* pset, id_t* plabels,
                                   processor_t* pproc) {
  // build the map. The map maps the old vertex id to its new id in the
  // partition. This is necessary because the vertices assigned to a
  // partition will be renamed so that the ids are contiguous from 0 to
  // partition->subgraph.vertex_count - 1.
  init_build_map(pset, plabels);

  // Set the processor type and reset the vertex count, will be set again next
  for (int pid = 0; pid < pset->partition_count; pid++) {
    pset->partitions[pid].id = pid;
    pset->partitions[pid].processor = pproc[pid];
    pset->partitions[pid].subgraph.edge_count = 0;
    pset->partitions[pid].subgraph.vertex_count = 0;
  }

  // Construct the partitions vertex, edge and weight lists
  {
  graph_t* graph = pset->graph;
  for (id_t vid = 0; vid < graph->vertex_count; vid++) {
    partition_t* partition = &pset->partitions[plabels[vid]];
    graph_t* subgraph = &partition->subgraph;
    subgraph->vertices[subgraph->vertex_count] =
      subgraph->edge_count;
    for (id_t i = graph->vertices[vid]; i < graph->vertices[vid + 1]; i++) {
      subgraph->edges[subgraph->edge_count] =
        pset->id_in_partition[graph->edges[i]];
      if (graph->weighted) {
        subgraph->weights[subgraph->edge_count] =
          graph->weights[i];
      }
      subgraph->edge_count++;
    }
    subgraph->vertices[subgraph->vertex_count + 1] =
      subgraph->edge_count;
    subgraph->vertex_count++;
  }
  }
}

PRIVATE void init_sort_nbrs(partition_set_t* pset) {
  uint32_t pcount = pset->partition_count;
  for (uint32_t pid = 0; pid < pcount; pid++) {
    graph_t* subgraph = &pset->partitions[pid].subgraph;
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (id_t v = 0; v < subgraph->vertex_count; v++) {
      id_t* nbrs = &subgraph->edges[subgraph->vertices[v]];
      qsort(nbrs, subgraph->vertices[v+1] - subgraph->vertices[v],
            sizeof(id_t), compare_ids);
    }
  }
}

PRIVATE void init_build_partitions_gpu(partition_set_t* pset) {
  uint32_t pcount = pset->partition_count;
  for (uint32_t pid = 0; pid < pcount; pid++) {
    partition_t* partition = &pset->partitions[pid];
    if (partition->processor.type != PROCESSOR_GPU) continue;
    CALL_CU_SAFE(cudaSetDevice(partition->processor.id));
    CALL_CU_SAFE(cudaStreamCreate(&partition->streams[0]));
    CALL_CU_SAFE(cudaStreamCreate(&partition->streams[1]));
    CALL_CU_SAFE(cudaEventCreate(&partition->event_start));
    CALL_CU_SAFE(cudaEventCreate(&partition->event_end));
    graph_t* subgraph_h = (graph_t*)malloc(sizeof(graph_t));
    assert(subgraph_h);
    memcpy(subgraph_h, &partition->subgraph, sizeof(graph_t));
    graph_t* subgraph_d = NULL;
    CALL_SAFE(graph_initialize_device(subgraph_h, &subgraph_d));
    graph_finalize(subgraph_h);
    memcpy(&partition->subgraph, subgraph_d, sizeof(graph_t));
    free(subgraph_d);
  }
}

error_t partition_set_initialize(graph_t* graph, id_t* plabels,
                                 processor_t* pproc, int pcount,
                                 size_t msg_size, partition_set_t** pset) {
  assert(graph && plabels && pproc);
  if (pcount > MAX_PARTITION_COUNT) return FAILURE;

  // Setup space and initialize the partition set data structure
  CHK_SUCCESS(init_allocate_struct_space(graph, pcount, msg_size, pset), err);

  // Get the partition sizes
  init_compute_partitions_sizes(*pset, plabels);

  // Allocate partitions space
  init_allocate_partitions_space(*pset);

  // Build the state of each partition
  init_build_partitions(*pset, plabels, pproc);

  // Sort nbrs of each each vertex to improve access locality
  init_sort_nbrs(*pset);

  // Initialize grooves' inbox and outbox state
  grooves_initialize(*pset);

  // Build the state on the GPU(s) for GPU residing partitions
  init_build_partitions_gpu(*pset);

  return SUCCESS;
 err:
  return FAILURE;
}

error_t partition_set_finalize(partition_set_t* pset) {
  assert(pset);
  assert(pset->partitions);
  for (int pid = 0; pid < pset->partition_count; pid++) {
    partition_t* partition = &pset->partitions[pid];
    graph_t* subgraph = &partition->subgraph;
    if (partition->processor.type == PROCESSOR_GPU) {
      CALL_CU_SAFE(cudaSetDevice(partition->processor.id));
      CALL_CU_SAFE(cudaStreamDestroy(partition->streams[0]));
      CALL_CU_SAFE(cudaStreamDestroy(partition->streams[1]));
      CALL_CU_SAFE(cudaEventDestroy(partition->event_start));
      CALL_CU_SAFE(cudaEventDestroy(partition->event_end));
      CALL_CU_SAFE(cudaFree(subgraph->edges));
      CALL_CU_SAFE(cudaFree(subgraph->vertices));
      if (subgraph->weighted && subgraph->weights)
        CALL_CU_SAFE(cudaFree(subgraph->weights));
    } else {
      assert(partition->processor.type == PROCESSOR_CPU);
      if (subgraph->vertices) free(subgraph->vertices);
      if (subgraph->edges) free(subgraph->edges);
      if (pset->weighted && subgraph->weights) {
        free(subgraph->weights);
      }
    }
    if (subgraph->vertices) free(partition->map);
  }
  grooves_finalize(pset);
  free(pset->partitions);
  free(pset->id_in_partition);
  free(pset);
  return SUCCESS;
}