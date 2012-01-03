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

error_t partition_set_initialize(graph_t* graph, id_t* labels, 
                                 int partition_count, 
                                 partition_set_t** partition_set_ret) {
  assert(graph && labels);
  if (partition_count > MAX_PARTITION_COUNT) return FAILURE;

  // Setup space and initialize the partition set data structure
  partition_set_t* partition_set = 
    (partition_set_t*)calloc(1, sizeof(partition_set_t));
  if (!partition_set) return FAILURE;
  partition_set->partitions = 
    (partition_t*)calloc(partition_count, sizeof(partition_t));
  if (!partition_set->partitions) return FAILURE;

  // Get the partition sizes
  for (id_t vid = 0; vid < graph->vertex_count; vid++) {
    id_t nbr_count = graph->vertices[vid + 1] - graph->vertices[vid];
    int pid = labels[vid];
    partition_t* partition = &(partition_set->partitions[pid]);
    partition->vertex_count += 1;
    partition->edge_count += nbr_count;
  }
  
  // Allocate partitions space
  for (int pid = 0; pid < partition_count; pid++) {
    partition_t* partition = &partition_set->partitions[pid];
    if (partition->vertex_count > 0) {
      partition->vertices = (id_t*)mem_alloc(sizeof(id_t) * 
                                             (partition->vertex_count + 1));
      if (partition->edge_count > 0) {
        partition->edges = (id_t*)mem_alloc(sizeof(id_t) * 
                                            partition->edge_count);
        if (graph->weighted) {
          partition->weights =
            (weight_t*)mem_alloc(sizeof(weight_t) * partition->edge_count);
        }
      }
    }
    // Reset the vertex and edge count, will be set again while building the
    // partitions vertex and edge lists
    partition->edge_count = 0;
    partition->vertex_count = 0;
  }

  // build the map. The map maps the old vertex id to its new id in the 
  // partition. This is necessary because the vertices assigned to a 
  // partition will be renamed so that the ids are contiguous from 0 to
  // partition->vertex_count - 1.
  id_t* map = (id_t*)calloc(graph->vertex_count, sizeof(id_t));
  if (!map) return FAILURE;
  for (id_t vid = 0; vid < graph->vertex_count; vid++) {
    id_t pid = labels[vid];
    partition_t* partition = &partition_set->partitions[pid];
    map[vid] = partition->vertex_count;
    partition->vertex_count++;
  }

  // Reset the vertex count, will be set again next
  for (int pid = 0; pid < partition_count; pid++) {
    partition_set->partitions[pid].vertex_count = 0;
  }

  // Construct the partitions vertex, edge and weight lists 
  for (id_t vid = 0; vid < graph->vertex_count; vid++) {
    partition_t* partition = &partition_set->partitions[labels[vid]];
    partition->vertices[partition->vertex_count] = partition->edge_count;
    for (id_t i = graph->vertices[vid]; i < graph->vertices[vid + 1]; i++) {
      int nbr_pid = labels[graph->edges[i]];
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
  
  // clean up
  free(map);

  partition_set->graph = graph;
  partition_set->partition_count = partition_count;
  partition_set->weighted = graph->weighted;  
  *partition_set_ret = partition_set;
  return SUCCESS;
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
