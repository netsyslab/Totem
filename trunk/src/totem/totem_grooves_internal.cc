/**
 * Implements internal functionality to the Grooves module. The functions here
 * are purely C++. Splitting those functions from the main module allows 
 * accessing GCC and C++ compiler features not supported by nvcc, such as using
 * range-based for loops and std containers (e.g., map and set).
 *
 *  Created on: 2012-01-25
 *  Author: Abdullah Gharaibeh
 */

// system includes
#include <tbb/concurrent_unordered_set.h>

// totem includes
#include "totem_partition.h"
#include "totem_util.h"

PRIVATE void init_get_rmt_nbrs_list(partition_t* par, vid_t vcount,
                                    uint32_t pcount, vid_t** nbrs,
                                    int* count_per_par) {
  graph_t* subgraph = &(par->subgraph);
  // A set to identify the remote neighbors.
  tbb::concurrent_unordered_set<vid_t> nbrs_set;
  eid_t rmt_edge_count = 0;
  OMP(omp parallel for schedule(guided) reduction(+:rmt_edge_count))
  for (vid_t vid = 0; vid < subgraph->vertex_count; vid++) {
    for (eid_t i = subgraph->vertices[vid];
         i < subgraph->vertices[vid + 1]; i++) {
      vid_t nbr = subgraph->edges[i];
      uint32_t nbr_pid = GET_PARTITION_ID(nbr);
      if (nbr_pid != par->id) {
        rmt_edge_count++;
        nbrs_set.insert(nbr);
      }
    }
  }
  par->rmt_edge_count = rmt_edge_count;

  // Initialize the counters to zero.
  memset(count_per_par, 0, MAX_PARTITION_COUNT * sizeof(int));
  *nbrs = (vid_t*)calloc(nbrs_set.size(), sizeof(vid_t));
  par->rmt_vertex_count = 0;
  for (auto nbr : nbrs_set) {
    (*nbrs)[par->rmt_vertex_count++] = nbr;
    count_per_par[GET_PARTITION_ID(nbr)]++;
  }
}

PRIVATE void init_map_rmt_nbrs(partition_t* par, uint32_t pcount, 
                               vid_t* rmt_nbrs, int* count_per_par, 
                               hash_table_t** ht) {
  // Sort the ids. This significantly improves the performance of
  // memory scatter operations and prefetching by improving data locality
  // (hence cache hit rate).
  tbb::parallel_sort(rmt_nbrs, rmt_nbrs + par->rmt_vertex_count, 
                     compare_ids_tbb);

  // Build the map.
  CALL_SAFE(hash_table_initialize_cpu(par->rmt_vertex_count, ht));
  vid_t cur_rmt_v = 0;
  while (cur_rmt_v < par->rmt_vertex_count) {
    int count = count_per_par[GET_PARTITION_ID(rmt_nbrs[cur_rmt_v])];
    OMP(omp parallel for)
    for (int i = 0; i < count; i++) {
      CALL_SAFE(hash_table_put_cpu(*ht, rmt_nbrs[cur_rmt_v + i], i));
    }
    cur_rmt_v += count;
  }
}

PRIVATE void init_map_rmt_nbrs(partition_t* par, uint32_t pcount, vid_t* nbrs,
                               int* count_per_par, hash_table_t* ht,
                               vid_t** rmt_nbrs) {
  // Allocate the state, it is per remote partition.
  memset(rmt_nbrs, 0, MAX_PARTITION_COUNT * sizeof(vid_t*));
  for (int i = 0; i < MAX_PARTITION_COUNT; i++) {
    if (count_per_par[i]) {
      rmt_nbrs[i] = (vid_t*)calloc(count_per_par[i], sizeof(vid_t));
      assert(rmt_nbrs[i]);
    }
  }
  
  // Create the final mapping.
  OMP(omp parallel for schedule(guided))
  for (vid_t v = 0; v < par->rmt_vertex_count; v++) {
    int index; HT_LOOKUP(ht, nbrs[v], index);
    vid_t* pnbrs = rmt_nbrs[GET_PARTITION_ID(nbrs[v])];
    pnbrs[index] = GET_VERTEX_ID(nbrs[v]);
  }
}

PRIVATE void init_update_subgraph(partition_t* par, hash_table_t* ht) {
  graph_t* subgraph = &(par->subgraph);
  OMP(omp parallel for schedule(guided))
  for (vid_t vid = 0; vid < subgraph->vertex_count; vid++) {
    for (eid_t i = subgraph->vertices[vid];
         i < subgraph->vertices[vid + 1]; i++) {
      vid_t nbr = subgraph->edges[i];
      uint32_t nbr_pid = GET_PARTITION_ID(nbr);
      if (nbr_pid != par->id) {
        int new_nbr_id;
        HT_LOOKUP(ht, nbr, new_nbr_id);
        assert(new_nbr_id != -1);
        subgraph->edges[i] = SET_PARTITION_ID((vid_t)new_nbr_id, nbr_pid);
      }
    }
  }
}

void init_get_rmt_nbrs(partition_t* par, vid_t vcount, uint32_t pcount,
                       vid_t** rmt_nbrs, int* count_per_par) {
  // First, identify the remote neighbors to this partition. "nbrs" is a flat
  // array that stores the list of all remote neighbors, irrespective of which
  // remote partition they belong to. "count_per_par" array is the number of
  // remote neighbors per remote partition.
  vid_t* nbrs;
  init_get_rmt_nbrs_list(par, vcount, pcount, &nbrs, count_per_par);

  // Second, map the remote neighbors ids in this partition to a new contiguous
  // id space. This new id space is relevant to this partition only. Having
  // the remote vertices mapped to a contiguous id space enables accesing their
  // shadow state (i.e., the in/outbox tables) as an indexed array, hence a
  // performance advantage (compared, for example, to using the original ids as
  // keys to access the shadow state maintained in a hash table).

  // Note that we will have a contiguous id space per remote partition. This
  // makes it easier for grooves to communicate the box tables between
  // partitions without the need for any preprocessing.
  // More importantly, the mapping is sorted. For example, if this partition has
  // the following neighbors in another partition: {5, 10, 3}, then the mapping
  // will be {5:1, 10:2, 3:0}. This is important for the performance of memory
  // scatter operations and prefetching by improving data locality (hence cache
  // hit rate) when aggregating the data sent by a remote partition with the
  // partition's local state (i.e., engine_scatter_inbox_* functions defined
  // in totem_engine_internal.cuh).
  if (par->rmt_vertex_count) {
    // Create the hash table that maps the remote vertex ids to the new (local)
    // contiguous id space. "ht" is a hash table in which the keys are the
    // original remote vertex id (including the partition id), while the values
    // are the new id of the vertex.
    hash_table_t* ht;
    init_map_rmt_nbrs(par, pcount, nbrs, count_per_par, &ht);

    // Create a two dimensional map (the first dimension is the partition id,
    // while the second is the remote vertex id) from the previously created
    // hash table. The values stored in "rmt_nbrs" are the original remote
    // vertex ids, while the index that is used to access a value represents
    // the corresponding new id.
    init_map_rmt_nbrs(par, pcount, nbrs, count_per_par, ht, rmt_nbrs);

    // Finally, reflect the new mapping on the partition's graph data structure.
    init_update_subgraph(par, ht);
    hash_table_finalize_cpu(ht);
    free(nbrs);
  }
}
