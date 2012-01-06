/**
 * Implements the graph partitionining interface defined in totem_partition.h
 *
 *  Created on: 2011-12-29
 *  Author: Abdullah Gharaibeh
 */

// totem includes
#include "totem_partition.h"
#include "totem_mem.h"

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
    for (id_t v = 0; v < partition->vertex_count; v++) {
      for (id_t e = partition->vertices[v]; e < partition->vertices[v + 1];
           e++) {
        if ((uint64_t)p == GET_PARTITION_ID(partition->edges[e])) {
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

error_t partition_random(graph_t* graph, int number_of_partitions,
                         unsigned int seed, id_t** partition_labels) {
  // Check pre-conditions
  if (graph == NULL) {
    // TODO(elizeu): Use Lauro's beautiful logging library.
    printf("ERROR: Graph object is NULL, cannot proceed with partitioning.\n");
    *partition_labels = NULL;
    return FAILURE;
  }
  // The requested number of partitions should be positive
  if ((number_of_partitions <= 0) || (graph->vertex_count == 0)) {
    printf("ERROR: Invalid number of partitions or empty graph: %d (|V|),",
           graph->vertex_count);
    printf(" %d (partitions).\n", number_of_partitions);
    *partition_labels = NULL;
    return FAILURE;
  }

  // Allocate the partition vector
  id_t* partitions = (id_t*)malloc((graph->vertex_count) * sizeof(id_t));

  // Initialize the random number generator
  srand(seed);

  for (uint64_t vertex_id = 0; vertex_id < graph->vertex_count; vertex_id++) {
    // Assign each vertex to a random partition within the range
    // (0, NUMBER_OF_PARTITIONS - 1)
    partitions[vertex_id] = rand() % number_of_partitions;
  }
  *partition_labels = partitions;
  return SUCCESS;
}

PRIVATE error_t init_allocate_struct_space(graph_t* graph, int pcount, 
                                           partition_set_t** pset) {
  *pset = (partition_set_t*)calloc(1, sizeof(partition_set_t));
  if (!*pset) return FAILURE;
  (*pset)->partitions = (partition_t*)calloc(pcount, sizeof(partition_t));
  if (!(*pset)->partitions) return FAILURE;
  (*pset)->graph = graph;
  (*pset)->partition_count = pcount;
  (*pset)->weighted = graph->weighted;  
  return SUCCESS;
}

PRIVATE void init_compute_partitions_sizes(partition_set_t* pset, 
                                           id_t* plabels) {
  graph_t* graph = pset->graph;
  for (id_t vid = 0; vid < graph->vertex_count; vid++) {
    id_t nbr_count = graph->vertices[vid + 1] - graph->vertices[vid];
    int pid = plabels[vid];
    partition_t* partition = &(pset->partitions[pid]);
    partition->vertex_count += 1;
    partition->edge_count += nbr_count;
  }
}

PRIVATE void init_allocate_partitions_space(partition_set_t* pset) {
  for (int pid = 0; pid < pset->partition_count; pid++) {
    partition_t* partition = &pset->partitions[pid];
    if (partition->vertex_count > 0) {
      partition->vertices = (id_t*)mem_alloc(sizeof(id_t) * 
                                             (partition->vertex_count + 1));
      if (partition->edge_count > 0) {
        partition->edges = (id_t*)mem_alloc(sizeof(id_t) * 
                                            partition->edge_count);
        if (pset->graph->weighted) {
          partition->weights = (weight_t*)mem_alloc(sizeof(weight_t) * 
                                                    partition->edge_count);
        }
      }
    }
  }
}

PRIVATE error_t init_build_map(partition_set_t* pset, id_t* plabels, 
                               id_t** map) {
  // Reset the vertex and edge count, will be set again while building the map
  for (int pid = 0; pid < pset->partition_count; pid++) {
    pset->partitions[pid].vertex_count = 0;
  }
  *map = (id_t*)calloc(pset->graph->vertex_count, sizeof(id_t));
  if (!*map) return FAILURE;
  for (id_t vid = 0; vid < pset->graph->vertex_count; vid++) {
    id_t pid = plabels[vid];
    partition_t* partition = &pset->partitions[pid];
    (*map)[vid] = partition->vertex_count;
    partition->vertex_count++;
  }
  return SUCCESS;
}

PRIVATE error_t init_build_partitions(partition_set_t* pset, id_t* plabels) {
  // build the map. The map maps the old vertex id to its new id in the 
  // partition. This is necessary because the vertices assigned to a 
  // partition will be renamed so that the ids are contiguous from 0 to
  // partition->vertex_count - 1.
  id_t* map;
  CHK_SUCCESS(init_build_map(pset, plabels, &map), err);

  // Reset the vertex count, will be set again next
  for (int pid = 0; pid < pset->partition_count; pid++) {
    pset->partitions[pid].edge_count = 0;
    pset->partitions[pid].vertex_count = 0;
  }

  // Construct the partitions vertex, edge and weight lists
  {
  graph_t* graph = pset->graph;
  for (id_t vid = 0; vid < graph->vertex_count; vid++) {
    partition_t* partition = &pset->partitions[plabels[vid]];
    partition->vertices[partition->vertex_count] = partition->edge_count;
    for (id_t i = graph->vertices[vid]; i < graph->vertices[vid + 1]; i++) {
      int nbr_pid = plabels[graph->edges[i]];
      id_t nbr_new_id = map[graph->edges[i]];
      partition->edges[partition->edge_count] = 
        SET_PARTITION_ID(nbr_new_id, nbr_pid);
      if (graph->weighted) {
        partition->weights[partition->edge_count] = graph->weights[i];
      }
      partition->edge_count++;
    }
    partition->vertices[partition->vertex_count + 1] = partition->edge_count;
    partition->vertex_count++;
  }
  }
  // clean up
  free(map);
  return SUCCESS;
 err:
  return FAILURE;
}

error_t partition_set_initialize(graph_t* graph, id_t* labels, 
                                 int partition_count, 
                                 partition_set_t** partition_set) {
  assert(graph && labels);
  if (partition_count > MAX_PARTITION_COUNT) return FAILURE;

  // Setup space and initialize the partition set data structure
  CHK_SUCCESS(init_allocate_struct_space(graph, partition_count, partition_set),
              err);

  // Get the partition sizes
  init_compute_partitions_sizes(*partition_set, labels);
  
  // Allocate partitions space
  init_allocate_partitions_space(*partition_set);

  // build the state of each partition
  CHK_SUCCESS(init_build_partitions(*partition_set, labels), err_free_space);
  
  return SUCCESS;
 err_free_space:
  partition_set_finalize(*partition_set);
 err:
  return FAILURE;
}

error_t partition_set_finalize(partition_set_t* partition_set) {
  assert(partition_set);
  assert(partition_set->partitions);
  for (int pid = 0; pid < partition_set->partition_count; pid++) {
    partition_t* partition = &partition_set->partitions[pid];
    if (partition->vertices) mem_free(partition->vertices);
    if (partition->edges) mem_free(partition->edges);
    if (partition_set->weighted && partition->weights) {
      mem_free(partition->weights);
    }
  }
  free(partition_set->partitions);
  free(partition_set);
  return SUCCESS;
}
