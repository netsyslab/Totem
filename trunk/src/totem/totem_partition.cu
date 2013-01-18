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

/*
 * A vertex-degree data type used in the partitioning algorithms that depend 
 * on sorting the vertices by edge degree.
 */
typedef struct vdegree_s {
  vid_t id;     // vertex id
  vid_t degree; // vertex degree
} vdegree_t;


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
    eid_t local_edges = 0;
    eid_t remote_edges = 0;
    partition_t* partition = &partition_set->partitions[p];
    graph_t* subgraph = &partition->subgraph;
    for (vid_t v = 0; v < subgraph->vertex_count; v++) {
      for (eid_t e = subgraph->vertices[v];
           e < subgraph->vertices[v + 1]; e++) {
        if (p == GET_PARTITION_ID(subgraph->edges[e])) {
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
                                vid_t** partition_labels) {
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
                                 vid_t** partition_labels) {
  // Allocate the partition vector
  vid_t* partitions = (vid_t*)malloc((graph->vertex_count) * sizeof(vid_t));

  // Initialize the random number generator
  // TODO(abdullah): pass the seed as an argument to control the randomness
  //                 of the algorithm if the experiments show variability in 
  //                 performance or the characteristics of the partitions.
  srand(time(NULL));

  for (vid_t vertex_id = 0; vertex_id < graph->vertex_count; vertex_id++) {
    // Assign each vertex to a random partition within the range
    // (0, PARTITION_COUNT - 1)
    partitions[vertex_id] = rand() % partition_count;
  }
  *partition_labels = partitions;
  return SUCCESS;
}

error_t partition_random(graph_t* graph, int partition_count,
                         double* partition_fraction, vid_t** partition_labels) {
  // Check pre-conditions
  if (partition_check(graph, partition_count, partition_fraction,
                      partition_labels) == FAILURE) {
    return FAILURE;
  }

  // Check if the client is asking for equal divide among partitions
  if (partition_fraction == NULL) {
    return partition_random(graph, partition_count, partition_labels);
  }

  // Allocate the partition vector
  vid_t* partitions = (vid_t*)malloc(graph->vertex_count * sizeof(vid_t));
  assert(partitions != NULL);

  // Initialize the random number generator
  srand(time(NULL));

  // Allocate all the partition ids to the id vector
  vid_t v = 0;
  for (int pid = 0; pid < partition_count; pid++) {
    vid_t end = (pid == partition_count - 1) ? graph->vertex_count :
      v + ((double)graph->vertex_count * partition_fraction[pid]);
    for (; v < end; v++) {
      partitions[v] = pid;
    }
  }

  /* Randomize the vector to achieve a random distribution. This is using the
   * Fisher-Yates "Random permutation" algorithm */
  for (vid_t i = graph->vertex_count - 1; i > 0; i--) {
    vid_t j = rand() % (i + 1);
    vid_t temp = partitions[i];
    partitions[i] = partitions[j];
    partitions[j] = temp;
  }

  *partition_labels = partitions;
  return SUCCESS;
}

PRIVATE int compare_degrees_asc(const void *a, const void *b) {
  vdegree_t* d1 = (vdegree_t*)a;
  vdegree_t* d2 = (vdegree_t*)b;
  if (d1->degree < d2->degree) return -1;
  if (d1->degree == d2->degree) return 0;
  return 1;
}

PRIVATE int compare_degrees_dsc(const void *a, const void *b) {
  vdegree_t* d1 = (vdegree_t*)a;
  vdegree_t* d2 = (vdegree_t*)b;
  if (d1->degree > d2->degree) return -1;
  if (d1->degree == d2->degree) return 0;
  return 1;
}

PRIVATE 
error_t partition_by_sorted_degree(graph_t* graph, int partition_count, 
                                   bool asc, double* partition_fraction, 
                                   vid_t** partition_labels) {
  // Check pre-conditions
  if (partition_check(graph, partition_count, partition_fraction,
                      partition_labels) == FAILURE) {
    return FAILURE;
  }

  bool even_fractions = false;
  if (partition_fraction == NULL) {
    even_fractions = true;
    partition_fraction = (double*)calloc(partition_count, sizeof(double));
    for (int pid = 0; pid < partition_count; pid++) {
      partition_fraction[pid] = 1.0/(double)partition_count;
    }
  }

  // Prepare the degree-sorted list of vertices 
  vdegree_t* vd = (vdegree_t*)calloc(graph->vertex_count, sizeof(vdegree_t));
  assert(vd);
  for (vid_t v = 0; v < graph->vertex_count; v++) {
    vd[v].id = v;
    vd[v].degree = graph->vertices[v + 1] - graph->vertices[v];
  }
  if (asc) {
    qsort(vd, graph->vertex_count, sizeof(vdegree_t), compare_degrees_asc);
  } else {
    qsort(vd, graph->vertex_count, sizeof(vdegree_t), compare_degrees_dsc);
  }

  // Allocate the labels array
  *partition_labels = (vid_t*)calloc(graph->vertex_count, sizeof(vid_t));
  assert(*partition_labels);

  // Assign vertices to partitions. Considering the sum of edges and vertices as
  // the normalizing factor allows to create a partition with space complexity 
  // proportional to the requested fraction. This is important when creating 
  // partitions with few edges but many vertices (e.g., a partition dominated by
  // low-degree vertices). 
  double total_elements = (double)graph->vertex_count + 
    (double)graph->edge_count;
  vid_t index = 0;
  for (int pid = 0; pid < partition_count - 1; pid++) {
    double assigned = 0;
    while ((assigned / total_elements < partition_fraction[pid]) &&
           (index < graph->vertex_count)) {
      assigned += vd[index].degree + 1;
      (*partition_labels)[vd[index].id] = pid;
      index++;
    }
  }
  // Assign the rest to the last partition
  for (; index < graph->vertex_count; index++) {
    (*partition_labels)[vd[index].id] = partition_count - 1;
  }

  // Clean up
  if (even_fractions) {
    free(partition_fraction);
  }
  free(vd);
  return SUCCESS;
}

error_t partition_by_asc_sorted_degree(graph_t* graph, int partition_count,
                                       double* partition_fraction, 
                                       vid_t** partition_labels) {
  return partition_by_sorted_degree(graph, partition_count, true, 
                                    partition_fraction, partition_labels);
}

error_t partition_by_dsc_sorted_degree(graph_t* graph, int partition_count,
                                       double* partition_fraction, 
                                       vid_t** partition_labels) {
  return partition_by_sorted_degree(graph, partition_count, false, 
                                    partition_fraction, partition_labels);
}

PRIVATE error_t init_allocate_struct_space(graph_t* graph, int pcount,
                                           size_t push_msg_size,
                                           size_t pull_msg_size,
                                           partition_set_t** pset) {
  *pset = (partition_set_t*)calloc(1, sizeof(partition_set_t));
  assert(*pset);
  (*pset)->partitions = (partition_t*)calloc(pcount, sizeof(partition_t));
  assert((*pset)->partitions);
  (*pset)->id_in_partition = (vid_t*)calloc(graph->vertex_count, sizeof(vid_t));
  assert((*pset)->id_in_partition);
  (*pset)->graph = graph;
  (*pset)->partition_count = pcount;
  (*pset)->push_msg_size = push_msg_size;
  (*pset)->pull_msg_size = pull_msg_size;
  (*pset)->weighted = graph->weighted;
  return SUCCESS;
}

PRIVATE
void init_compute_partitions_sizes(partition_set_t* pset, vid_t* plabels) {
  graph_t* graph = pset->graph;
  OMP(omp parallel for)
  for (vid_t vid = 0; vid < graph->vertex_count; vid++) {
    vid_t nbr_count = graph->vertices[vid + 1] - graph->vertices[vid];
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
        (eid_t*)malloc(sizeof(eid_t) * (subgraph->vertex_count + 1));
      assert(subgraph->vertices);
      partition->map = (vid_t*)calloc(subgraph->vertex_count, sizeof(vid_t));
      if (subgraph->edge_count > 0) {
        subgraph->edges = (vid_t*)malloc(sizeof(vid_t) * subgraph->edge_count);
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

PRIVATE void init_build_map(partition_set_t* pset, vid_t* plabels) {
  // Reset the vertex and edge count, will be set again while building the map
  for (int pid = 0; pid < pset->partition_count; pid++) {
    pset->partitions[pid].subgraph.vertex_count = 0;
  }
  for (vid_t vid = 0; vid < pset->graph->vertex_count; vid++) {
    vid_t pid = plabels[vid];
    graph_t* subgraph = &pset->partitions[pid].subgraph;
     // forward map
    pset->id_in_partition[vid] = SET_PARTITION_ID(subgraph->vertex_count, pid);
    pset->partitions[pid].map[subgraph->vertex_count] = vid; // reverse map
    subgraph->vertex_count++;
  }
}

PRIVATE void init_build_partitions(partition_set_t* pset, vid_t* plabels,
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
  for (vid_t vid = 0; vid < graph->vertex_count; vid++) {
    partition_t* partition = &pset->partitions[plabels[vid]];
    graph_t* subgraph = &partition->subgraph;
    subgraph->vertices[subgraph->vertex_count] =
      subgraph->edge_count;
    for (eid_t i = graph->vertices[vid]; i < graph->vertices[vid + 1]; i++) {
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
    OMP(omp parallel for)
    for (vid_t v = 0; v < subgraph->vertex_count; v++) {
      vid_t* nbrs = &subgraph->edges[subgraph->vertices[v]];
      qsort(nbrs, subgraph->vertices[v+1] - subgraph->vertices[v],
            sizeof(vid_t), compare_ids);
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

error_t partition_set_initialize(graph_t* graph, vid_t* plabels,
                                 processor_t* pproc, int pcount,
                                 size_t push_msg_size, size_t pull_msg_size,
                                 partition_set_t** pset) {
  assert(graph && plabels && pproc);
  if (pcount > MAX_PARTITION_COUNT) return FAILURE;

  // Setup space and initialize the partition set data structure
  CHK_SUCCESS(init_allocate_struct_space(graph, pcount, push_msg_size, 
                                         pull_msg_size, pset), err);

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
