/**
 * Implements the Grooves interface.
 *
 *  Created on: 2012-01-25
 *  Author: Abdullah Gharaibeh
 */

// totem includes
#include "totem_comkernel.cuh"
#include "totem_grooves.h"
#include "totem_mem.h"
#include "totem_partition.h"
#include "totem_util.h"

PRIVATE
void init_get_rmt_nbrs_list(partition_t* par, uint32_t vcount, uint32_t pcount,
                            id_t** nbrs, int* count_per_par) {

  // Initialize the counters to zero
  memset(count_per_par, 0, MAX_PARTITION_COUNT * sizeof(int));

  // This is a temporary hash table to identify the remote neighbors.
  // It is initialized with conservative space such that it can accommodate
  // the extreme case where all vertices in other partitions are remote to
  // this partition
  graph_t* subg = &(par->subgraph);
  hash_table_t* ht;
  CALL_SAFE(hash_table_initialize_cpu(vcount - subg->vertex_count, &ht));
  // TODO (abdullah): parallelize this loop
  for (id_t vid = 0; vid < subg->vertex_count; vid++) {
    for (id_t i = subg->vertices[vid]; i < subg->vertices[vid + 1]; i++) {
      id_t nbr = subg->edges[i];
      int nbr_pid = GET_PARTITION_ID(nbr);
      if (nbr_pid != par->id) {
        par->rmt_edge_count++;
        bool found; HT_CHECK(ht, nbr, found);
        if (!found) {
          // new remote neighbour
          __sync_fetch_and_add(&count_per_par[nbr_pid], 1);
          CALL_SAFE(hash_table_put_cpu(ht, nbr, 1));
        }
      }
    }
  }
  CALL_SAFE(hash_table_get_keys_cpu(ht, nbrs, &par->rmt_vertex_count));
  hash_table_finalize_cpu(ht);
}

PRIVATE
void init_map_rmt_nbrs(partition_t* par, uint32_t pcount, id_t* nbrs,
                       int* count_per_par, hash_table_t** ht) {
  // Sort the ids. This significantly improves the performance of
  // memory scatter operations and prefetching by improving data locality
  // (hence cache hit rate)
  qsort(nbrs, par->rmt_vertex_count, sizeof(id_t), compare_ids);

  // Build the hash table map
  CALL_SAFE(hash_table_initialize_cpu(par->rmt_vertex_count, ht));
  id_t cur_rmt_v = 0;
  while (cur_rmt_v < par->rmt_vertex_count) {
    id_t count = count_per_par[GET_PARTITION_ID(nbrs[cur_rmt_v])];
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (id_t i = 0; i < count; i++) {
      CALL_SAFE(hash_table_put_cpu(*ht, nbrs[cur_rmt_v + i], i));
    }
    cur_rmt_v += count;
  }
}

PRIVATE
void init_map_rmt_nbrs(partition_t* par, uint32_t pcount, id_t* nbrs,
                       int* count_per_par, hash_table_t* ht,
                       id_t** rmt_nbrs) {
  // allocate the state, it is per remote partition
  memset(rmt_nbrs, 0, MAX_PARTITION_COUNT * sizeof(id_t*));
  for (int i = 0; i < MAX_PARTITION_COUNT; i++) {
    if (count_per_par[i]) {
      rmt_nbrs[i] = (id_t*)calloc(count_per_par[i], sizeof(id_t));
      assert(rmt_nbrs[i]);
    }
  }
  // create the final mapping
  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int v = 0; v < par->rmt_vertex_count; v++) {
    int index; HT_LOOKUP(ht, nbrs[v], index);
    id_t* pnbrs = rmt_nbrs[GET_PARTITION_ID(nbrs[v])];
    pnbrs[index] = GET_VERTEX_ID(nbrs[v]);
  }
}

PRIVATE
void init_update_subgraph(partition_t* par, hash_table_t* ht) {
  graph_t* subg = &(par->subgraph);
  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (id_t vid = 0; vid < subg->vertex_count; vid++) {
    for (id_t i = subg->vertices[vid]; i < subg->vertices[vid + 1]; i++) {
      id_t nbr = subg->edges[i];
      int nbr_pid = GET_PARTITION_ID(nbr);
      if (nbr_pid != par->id) {
        int new_nbr_id;
        HT_LOOKUP(ht, nbr, new_nbr_id);
        assert(new_nbr_id != -1);
        subg->edges[i] = SET_PARTITION_ID((id_t)new_nbr_id, nbr_pid);
      }
    }
  }
}

PRIVATE
void init_get_rmt_nbrs(partition_t* par, uint32_t vcount, uint32_t pcount,
                       id_t** rmt_nbrs, int* count_per_par) {
  // First, identify the remote neighbors to this partition. "nbrs" is a flat
  // array that stores the list of all remote neighbors, irrespective of which
  // remote partition they belong to. "count_per_par" array is the number of
  // remote neighbors per remote partition
  id_t* nbrs;
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

PRIVATE void init_table_gpu(grooves_box_table_t* btable, uint32_t pcount,
                            size_t msg_size, grooves_box_table_t** btable_d,
                            grooves_box_table_t** btable_h) {
  *btable_h = (grooves_box_table_t*)calloc(pcount, sizeof(grooves_box_table_t));
  memcpy(*btable_h, btable, pcount * sizeof(grooves_box_table_t));
  // initialize the tables on the gpu
  for (uint32_t pid = 0; pid < pcount; pid++) {
    int count = (*btable_h)[pid].count;
    if (count) {
      CALL_CU_SAFE(cudaMalloc((void**)&((*btable_h)[pid].rmt_nbrs),
                              count * sizeof(id_t)));
      CALL_CU_SAFE(cudaMemcpy((*btable_h)[pid].rmt_nbrs,
                              btable[pid].rmt_nbrs, count * sizeof(id_t),
                              cudaMemcpyHostToDevice));
      CALL_CU_SAFE(cudaMalloc((void**)&((*btable_h)[pid].values),
                              count * msg_size));
    }
  }

  // transfer the table array
  CALL_CU_SAFE(cudaMalloc((void**)(btable_d), pcount *
                          sizeof(grooves_box_table_t)));
  CALL_CU_SAFE(cudaMemcpy(*btable_d, (*btable_h),
                          pcount * sizeof(grooves_box_table_t),
                          cudaMemcpyHostToDevice));
}

PRIVATE void init_outbox_table(partition_t* partition, uint32_t pcount,
                               id_t** rmt_nbrs, int* count_per_par,
                               size_t msg_size) {
  grooves_box_table_t* outbox = partition->outbox;
  uint32_t pid = partition->id;
  for (int rmt_pid = (pid + 1) % pcount; rmt_pid != pid;
       rmt_pid = (rmt_pid + 1) % pcount) {
    outbox[rmt_pid].count = count_per_par[rmt_pid];
    if (outbox[rmt_pid].count) {
      assert(rmt_nbrs[rmt_pid]);
      outbox[rmt_pid].rmt_nbrs = rmt_nbrs[rmt_pid];
      if (partition->processor.type == PROCESSOR_CPU) {
        // Allocate the values array for the cpu-based partitions. The gpu-based
        // partitions will have their values array allocated later when their
        // state is initialized on the gpu
        outbox[rmt_pid].values = mem_alloc(outbox[rmt_pid].count * msg_size);
      }
    }
  }
}

PRIVATE void init_outbox(partition_set_t* pset) {
  uint32_t pcount = pset->partition_count;
  for (int pid = 0; pid < pcount; pid++) {
    partition_t* partition = &pset->partitions[pid];

    // each remote partition has a slot in the outbox array
    partition->outbox =
      (grooves_box_table_t*)calloc(pcount, sizeof(grooves_box_table_t));

    if (!partition->subgraph.vertex_count ||
        !partition->subgraph.edge_count) continue;

    // identify the remote nbrs and their count per remote partition
    id_t* rmt_nbrs[MAX_PARTITION_COUNT];
    int count_per_par[MAX_PARTITION_COUNT];
    init_get_rmt_nbrs(partition, pset->graph->vertex_count, pcount, rmt_nbrs,
                      count_per_par);
    // build the outbox
    if (partition->rmt_vertex_count) {
      // build the outbox tables for this partition
      init_outbox_table(partition, pcount, rmt_nbrs, count_per_par,
                        pset->msg_size);
    }
  }
}

PRIVATE void init_inbox(partition_set_t* pset) {
  uint32_t pcount = pset->partition_count;
  for (int pid = 0; pid < pcount; pid++) {
    partition_t* partition = &pset->partitions[pid];

    // each remote partition has a slot in the inbox array
    partition->inbox = 
      (grooves_box_table_t*)calloc(pcount, sizeof(grooves_box_table_t));

    if (!partition->subgraph.vertex_count ||
        !partition->subgraph.edge_count) continue;

    for (int src_pid = (pid + 1) % pcount; src_pid != pid;
         src_pid = (src_pid + 1) % pcount) {
      partition_t* remote_par = &pset->partitions[src_pid];
      // An inbox in a partition is an outbox in the source partition.
      // Therefore, we just need to copy the state of the already built
      // source partition's outbox into the destination partition's inbox.
      // This includes copying a reference to the hash table that maintains
      // the set of boundary vertices (vertices that belong to this partition,
      // and are the destination of a remote edge that originates in another
      // partition, and are maintained in the outbox of that other partition).
      partition->inbox[src_pid] = remote_par->outbox[pid];
      if (remote_par->processor.type == PROCESSOR_GPU) {
        // if the remote processor is GPU, then a values array for this inbox
        // needs to be allocated on the host
        partition->inbox[src_pid].values =
          mem_alloc(partition->inbox[src_pid].count * pset->msg_size);
      }
    }
  }
}

PRIVATE void init_gpu_enable_peer_access(uint32_t pid, partition_set_t* pset) {
  uint32_t pcount = pset->partition_count;
  partition_t* partition = &pset->partitions[pid];
  for (int remote_pid = (pid + 1) % pcount; remote_pid != pid;
       remote_pid = (remote_pid + 1) % pcount) {
    partition_t* remote_par = &pset->partitions[remote_pid];
    if (remote_par->processor.type == PROCESSOR_GPU &&
        remote_par->processor.id != partition->processor.id) {
      CALL_CU_SAFE(cudaDeviceEnablePeerAccess(remote_par->processor.id, 0));
    }
  }
}

PRIVATE void init_gpu_state(partition_set_t* pset) {
  uint32_t pcount = pset->partition_count;

  // The following array will maintain pointers to the gpu-partitions' ouboxes
  // state on the host after they are copied to the gpu. These references are
  // maintained in order to free their state safely.
  // Outboxes are shared by the destination partitions as an inbox. Outboxes
  // that are shared between a gpu partition and a cpu one will not be freed. It
  // will be freed at finalization as part of finalizing the inboxes of the
  // destination cpu partition. However, outboxes shared between two gpu
  // partitions will be freed right after they are copied to the gpu (they will
  // be copied once as an outbox in the source partitions and as an inbox to the
  // destination).
  grooves_box_table_t* host_outboxes[MAX_PARTITION_COUNT];
  for (int pid = 0; pid < pcount; pid++) {
    partition_t* partition = &pset->partitions[pid];
    if (partition->processor.type == PROCESSOR_GPU) {
      // set device context, create the tables for this gpu
      CALL_CU_SAFE(cudaSetDevice(partition->processor.id));
      grooves_box_table_t* outbox_h = NULL;
      init_table_gpu(partition->outbox, pcount, pset->msg_size,
                     &partition->outbox_d, &outbox_h);
      host_outboxes[pid] = partition->outbox;
      partition->outbox = outbox_h;

      grooves_box_table_t* inbox_h = NULL;
      init_table_gpu(partition->inbox, pcount, pset->msg_size,
                     &partition->inbox_d, &inbox_h);
      free(partition->inbox);
      partition->inbox = inbox_h;
      init_gpu_enable_peer_access(pid, pset);
    }
  }

  // Clean up the state on the host. As mentioned before, only the outboxes
  // that are shared between two gpu-based partitions are freed.
  for (int pid = 0; pid < pcount; pid++) {
    partition_t* partition = &pset->partitions[pid];
    grooves_box_table_t* outbox = host_outboxes[pid];
    if (partition->processor.type == PROCESSOR_GPU) {
      for (int rmt_pid = 0; rmt_pid < pcount; rmt_pid++) {
        if (rmt_pid == pid) continue;
        partition_t* remote_par = &pset->partitions[rmt_pid];
        if (remote_par->processor.type == PROCESSOR_GPU &&
            outbox[rmt_pid].count) {
          free(outbox[rmt_pid].rmt_nbrs);
        }
      }
      free(host_outboxes[pid]);
    }
  }
}

error_t grooves_initialize(partition_set_t* pset) {
  if (pset->partition_count > 1) {
    init_outbox(pset);
    init_inbox(pset);
    init_gpu_state(pset);
  }
  return SUCCESS;
}

PRIVATE void finalize_table_gpu(grooves_box_table_t* btable_d,
                                grooves_box_table_t* btable_h,
                                uint32_t pcount) {
  CALL_CU_SAFE(cudaFree(btable_d));
  // finalize the tables on the gpu
  for (uint32_t pid = 0; pid < pcount; pid++) {
    if (btable_h[pid].count) {
      CALL_CU_SAFE(cudaFree(btable_h[pid].rmt_nbrs));
      CALL_CU_SAFE(cudaFree(btable_h[pid].values));
    }
  }
  free(btable_h);
}

PRIVATE void finalize_gpu_disable_peer_access(uint32_t pid,
                                              partition_set_t* pset) {
  uint32_t pcount = pset->partition_count;
  partition_t* partition = &pset->partitions[pid];
  for (int remote_pid = (pid + 1) % pcount; remote_pid != pid;
       remote_pid = (remote_pid + 1) % pcount) {
    partition_t* remote_par = &pset->partitions[remote_pid];
    if (remote_par->processor.type == PROCESSOR_GPU &&
        remote_par->processor.id != partition->processor.id) {
      CALL_CU_SAFE(cudaDeviceDisablePeerAccess(remote_par->processor.id));
    }
  }
}

PRIVATE void finalize_outbox(partition_set_t* pset) {
  uint32_t pcount = pset->partition_count;
  for (int pid = 0; pid < pcount; pid++) {
    partition_t* partition = &pset->partitions[pid];
    assert(partition->outbox);
    if (partition->processor.type == PROCESSOR_GPU) {
      CALL_CU_SAFE(cudaSetDevice(partition->processor.id));
      finalize_gpu_disable_peer_access(pid, pset);
      finalize_table_gpu(partition->outbox_d, partition->outbox, pcount);
    } else {
      assert(partition->processor.type == PROCESSOR_CPU);
      for (uint32_t rmt_pid = 0; rmt_pid < pcount; rmt_pid++) {
        if (partition->outbox[rmt_pid].count) {
          free(partition->outbox[rmt_pid].rmt_nbrs);
          mem_free(partition->outbox[rmt_pid].values);
        }
      }
      free(partition->outbox);
    }
  }
}

PRIVATE void finalize_inbox(partition_set_t* pset) {
  uint32_t pcount = pset->partition_count;
  for (int pid = 0; pid < pcount; pid++) {
    partition_t* partition = &pset->partitions[pid];
    assert(partition->inbox);
    if (partition->processor.type == PROCESSOR_GPU) {
      CALL_CU_SAFE(cudaSetDevice(partition->processor.id));
      finalize_table_gpu(partition->inbox_d, partition->inbox, pcount);
    } else {
      assert(partition->processor.type == PROCESSOR_CPU);
      for (int rmt_pid = 0; rmt_pid < pcount; rmt_pid++) {
        partition_t* remote_par = &pset->partitions[rmt_pid];
        // free only the inboxes that are the destination of an outbox of a gpu-
        // partition. Others that are destinations to a cpu-partition will be
        // freed as an outbox in the source partition.
        if (remote_par->processor.type == PROCESSOR_GPU &&
            partition->inbox[rmt_pid].count) {
          free(partition->inbox[rmt_pid].rmt_nbrs);
          mem_free(partition->inbox[rmt_pid].values);
        }
      }
      free(partition->inbox);
    }
  }
}

error_t grooves_finalize(partition_set_t* pset) {
  if (pset->partition_count > 1) {
    finalize_outbox(pset);
    finalize_inbox(pset);
  }
  return SUCCESS;
}

error_t grooves_launch_communications(partition_set_t* pset) {
  uint32_t pcount = pset->partition_count;
  for (int src_pid = 0; src_pid < pcount; src_pid++) {
    for (int dst_pid = (src_pid + 1) % pcount; dst_pid != src_pid;
         dst_pid = (dst_pid + 1) % pcount) {
      // if both partitions are on the host, then, by design the source
      // partition's outbox is shared with the destination partition's inbox,
      // hence no need to copy data
      if ((pset->partitions[src_pid].processor.type == PROCESSOR_CPU) &&
          (pset->partitions[dst_pid].processor.type == PROCESSOR_CPU)) continue;

      cudaStream_t* stream = &pset->partitions[src_pid].streams[0];
      grooves_box_table_t* src_box = &pset->partitions[src_pid].outbox[dst_pid];
      // if the two partitions share nothing, then we have nothing to do
      if (!src_box->count) continue;

      if (pset->partitions[dst_pid].processor.type == PROCESSOR_GPU) {
        stream = &pset->partitions[dst_pid].streams[0];
      }
      grooves_box_table_t* dst_box = &pset->partitions[dst_pid].inbox[src_pid];
      assert(src_box->count == dst_box->count);
      CALL_CU_SAFE(cudaMemcpyAsync(dst_box->values, src_box->values,
                                   dst_box->count * pset->msg_size,
                                   cudaMemcpyDefault, *stream));
    }
  }
  return SUCCESS;
}

error_t grooves_synchronize(partition_set_t* pset) {
  for (int pid = 0; pid < pset->partition_count; pid++) {
    partition_t* partition = &pset->partitions[pid];
    if (partition->processor.type == PROCESSOR_CPU) continue;
    CALL_CU_SAFE(cudaStreamSynchronize(partition->streams[0]));
  }
  return SUCCESS;
}
