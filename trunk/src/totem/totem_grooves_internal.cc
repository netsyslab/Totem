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

// Marks the remote neighbours of the specified partition. The marker is a two
// dimensional array, the first dimension is the remote partition id, while the
// second is the original remote vertex id.
PRIVATE void mark_rmt_nbrs(partition_t* partition, vid_t** marker) {
  const graph_t& subgraph = partition->subgraph;
  eid_t rmt_edge_count = 0;
  OMP(omp parallel for schedule(guided) reduction(+ : rmt_edge_count))
  for (vid_t v = 0; v < subgraph.vertex_count; v++) {
    for (eid_t i = subgraph.vertices[v]; i < subgraph.vertices[v + 1]; i++) {
      uint32_t nbr_pid = GET_PARTITION_ID(subgraph.edges[i]);
      if (nbr_pid == partition->id) { continue; }
      rmt_edge_count++;
      vid_t* rmt_partition_marker = marker[nbr_pid];
      rmt_partition_marker[GET_VERTEX_ID(subgraph.edges[i])] = 1;
    }
  }
  partition->rmt_edge_count = rmt_edge_count;
}

// Creates a map that maps the a remote vertex id with its new id used within
// the partition. The function assumes that the forward_map array is initialized
// such that if a vertex in rmt_subgraph is remote to the partition currently
// being processed, then the entry in the array indexed by the vertex id is set
// to 1, otherwise 0. This initialization is done via the mark_rmt_nbrs
// function. This function also returns the number of remote neighbours that
// belong to rmt_subgraph.
PRIVATE vid_t get_forward_rmt_nbrs_map(
    const graph_t& rmt_subgraph, vid_t* forward_map) {
  vid_t sum = 0;
  for (vid_t v = 0; v < rmt_subgraph.vertex_count; v++) {
    if (forward_map[v] == 0) { continue; }
    sum += forward_map[v];
    forward_map[v] = sum;
  }
  return sum;
}

// Creates a reverse remote neighbours map from the forward map.
PRIVATE void get_reverse_rmt_nbrs_map(
    const graph_t& rmt_subgraph, const vid_t* forward_map, vid_t* reverse_map) {
  OMP(omp parallel for schedule(guided))
  for (vid_t v = 0; v < rmt_subgraph.vertex_count; v++) {
    if (forward_map[v] != 0) { reverse_map[forward_map[v] - 1] = v; }
  }
}

// Creates the forwrard and reverse maps of a partition's remote neighbours.
// The number of remote neighbours per remote partition is also returned.
PRIVATE void get_rmt_nbrs_map(partition_set_t* pset, int pid,
                              vid_t** forward_map, vid_t** reverse_map,
                              vid_t* count_per_par) {
  memset(count_per_par, 0, MAX_PARTITION_COUNT * sizeof(vid_t));
  for (int p = 0; p < pset->partition_count; p++) {
    const graph_t& subgraph = pset->partitions[p].subgraph;
    if (subgraph.vertex_count == 0 || p == pid) { continue; }
    forward_map[p] = reinterpret_cast<vid_t*>(
        calloc(subgraph.vertex_count, sizeof(vid_t)));
    assert(forward_map[p]);
  }

  mark_rmt_nbrs(&pset->partitions[pid], forward_map);

  for (int p = 0; p < pset->partition_count; p++) {
    const graph_t& subgraph = pset->partitions[p].subgraph;
    if (subgraph.vertex_count == 0 || p == pid) { continue; }
    count_per_par[p] = get_forward_rmt_nbrs_map(subgraph, forward_map[p]);
    if (count_per_par[p] == 0) { continue; }
    pset->partitions[pid].rmt_vertex_count += count_per_par[p];
    reverse_map[p] = reinterpret_cast<vid_t*>(
        calloc(count_per_par[p], sizeof(vid_t)));
    get_reverse_rmt_nbrs_map(subgraph, forward_map[p], reverse_map[p]);
  }
}

// For each remote vertex in the partition's subgraph data structure, replace
// the id with an index that can be used to access its the remote vertex's
// outbox entry.
PRIVATE void update_subgraph(partition_t* par, vid_t** forward_map) {
  graph_t* subgraph = &par->subgraph;
  OMP(omp parallel for schedule(guided))
  for (vid_t v = 0; v < subgraph->vertex_count; v++) {
    for (eid_t i = subgraph->vertices[v]; i < subgraph->vertices[v + 1]; i++) {
      uint32_t nbr_pid = GET_PARTITION_ID(subgraph->edges[i]);
      if (nbr_pid != par->id) {
        vid_t nbr = GET_VERTEX_ID(subgraph->edges[i]);
        vid_t* map = forward_map[nbr_pid];
        vid_t local_id = map[nbr] - 1;
        subgraph->edges[i] = SET_PARTITION_ID(local_id, nbr_pid);
      }
    }
  }
}

// Maps the remote neighbors ids in this partition to a new contiguous id space.
// This new id space is relevant to this partition only. Having the remote
// vertices mapped to a contiguous id space enables accesing their shadow state
// (i.e., the in/outbox tables) as an indexed array, hence a performance
// advantage (compared, for example, to using the original ids as keys to access
// the shadow state maintained in a hash table).
//
// Note that we will have a contiguous id space per remote partition. This makes
// it easier for grooves to communicate the box tables between partitions
// without the need for any preprocessing.
//
// Note that the mapping is generated in order. For example, if this partition
// has the following neighbors in another partition: {5, 10, 3} then the mapping
// will be {5:1, 10:2, 3:0}. This is important for the performance of memory
// scatter operations and prefetching by improving data locality (hence cache
// hit rate) when aggregating the data sent by a remote partition with the
// partition's local state (i.e., engine_scatter_inbox_* functions defined in
// totem_engine_internal.cuh).
void init_get_rmt_nbrs(partition_set_t* pset, int pid,
                       vid_t** rmt_nbrs_map, vid_t* count_per_par) {
  // Maps the remote vertex ids to the new (local) contiguous id space. For each
  // remote partition there is an array where the index to it is the original
  // remote vertex id, while the values is the new id of the vertex. This map
  // is used to update the subgraph data structure of the partition with the
  // remote vertices new indecies.
  vid_t* forward_map[MAX_PARTITION_COUNT];
  get_rmt_nbrs_map(pset, pid, forward_map, rmt_nbrs_map, count_per_par);

  // Reflect the new mapping on the partition's graph data structure.
  partition_t* partition = &pset->partitions[pid];
  if (partition->rmt_vertex_count != 0) {
    update_subgraph(partition, forward_map);
  }

  for (int p = 0; p < pset->partition_count; p++) {
    const graph_t& rmt_subgraph = pset->partitions[p].subgraph;
    if (rmt_subgraph.vertex_count == 0 || p == pid) { continue; }
    free(forward_map[p]);
  }
}
