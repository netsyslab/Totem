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
void init_get_rmt_nbrs(partition_t* par, uint32_t vcount, uint32_t pcount,
                       id_t** nbrs, int** count_per_par) {
  graph_t* subg = &(par->subgraph);

  // This is a temporary hash table to identify the remote neighbors.
  // It is initialized with conservative space such that it can accommodate
  // the extreme case where all vertices in other partitions are remote to
  // this partition
  hash_table_t* ht;
  CALL_SAFE(hash_table_initialize_cpu(vcount - subg->vertex_count, &ht));
  *count_per_par = (int*)calloc(pcount - 1, sizeof(int));
  for (id_t vid = 0; vid < subg->vertex_count; vid++) {
    for (id_t i = subg->vertices[vid]; i < subg->vertices[vid + 1]; i++) {
      id_t nbr = subg->edges[i];
      int nbr_pid = GET_PARTITION_ID(nbr);
      if (nbr_pid != par->id) {
        par->rmt_edge_count++;
        bool found; HT_CHECK(ht, nbr, found);
        if (!found) {
          // new remote neighbour
          int bindex = GROOVES_BOX_INDEX(nbr_pid, par->id, pcount);
          __sync_fetch_and_add(&(*count_per_par)[bindex], 1);
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
  int cur_pid = GET_PARTITION_ID(nbrs[0]);
  uint32_t count = 0;
  for (int v = 0; v < par->rmt_vertex_count; v++) {
    int my_pid = GET_PARTITION_ID(nbrs[v]);
    if (my_pid != cur_pid) {
      assert(count == count_per_par[GROOVES_BOX_INDEX(cur_pid, par->id, 
                                                      pcount)]);
      assert(my_pid > cur_pid);
      cur_pid = my_pid;
      count = 0;
    }
    CALL_SAFE(hash_table_put_cpu(*ht, nbrs[v], count));
    count++;
  }
}

PRIVATE 
void init_map_rmt_nbrs(partition_t* par, uint32_t pcount, id_t* nbrs, 
                       int* count_per_par, hash_table_t* ht, 
                       id_t***rmt_nbrs) {
  // allocate the state, it is per remote partition
  *rmt_nbrs = (id_t**)calloc(pcount - 1, sizeof(id_t*));
  for (int i = 0; i < pcount - 1; i++) {
    if (count_per_par[i]) {
      (*rmt_nbrs)[i] = (id_t*)calloc(count_per_par[i], sizeof(id_t));
      assert((*rmt_nbrs)[i]);
    }
  }
  // create the final mapping
  for (int v = 0; v < par->rmt_vertex_count; v++) {
    int index; HT_LOOKUP(ht, nbrs[v], index);
    id_t* pnbrs = (*rmt_nbrs)[GROOVES_BOX_INDEX(GET_PARTITION_ID(nbrs[v]),
                                                par->id, pcount)];
    pnbrs[index] = GET_VERTEX_ID(nbrs[v]);
  }
}

PRIVATE
void init_update_subgraph(partition_t* par, hash_table_t* ht) {
  graph_t* subg = &(par->subgraph);
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
                       id_t*** rmt_nbrs, int** count_per_par) {
  // First, identify the remote neighbors to this partition. "nbrs" is a flat 
  // array that stores the list of all remote neighbors, irrespective of which
  // remote partition they belong to. "count_per_par" array is the number of
  // remote neighbors per remote partition (hence, its length is pcount - 1)
  id_t* nbrs;
  init_get_rmt_nbrs(par, vcount, pcount, &nbrs, count_per_par);

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
    init_map_rmt_nbrs(par, pcount, nbrs, *count_per_par, &ht);

    // Create a two dimensional map (the first dimension is the partition id, 
    // while the second is the remote vertex id) from the previously created
    // hash table. The values stored in "rmt_nbrs" are the original remote
    // vertex ids, while the index that is used to access a value represents
    // the corresponding new id.
    init_map_rmt_nbrs(par, pcount, nbrs, *count_per_par, ht, rmt_nbrs);

    // Finally, reflect the new mapping on the partition's graph data structure.
    init_update_subgraph(par, ht);
    hash_table_finalize_cpu(ht);
    free(nbrs);
  }
}

PRIVATE void init_table_gpu(grooves_box_table_t* btable, uint32_t bcount, 
                            size_t msg_size, grooves_box_table_t** btable_d,
                            grooves_box_table_t** btable_h) {
  *btable_h = (grooves_box_table_t*)calloc(bcount, sizeof(grooves_box_table_t));
  memcpy(*btable_h, btable, bcount * sizeof(grooves_box_table_t));
  // initialize the tables on the gpu  
  for (uint32_t bindex = 0; bindex < bcount; bindex++) {
    int count = (*btable_h)[bindex].count;
    if (count) {
      CALL_CU_SAFE(cudaMalloc((void**)&((*btable_h)[bindex].rmt_nbrs), 
                              count * sizeof(id_t)));
      CALL_CU_SAFE(cudaMemcpy((*btable_h)[bindex].rmt_nbrs, 
                              btable[bindex].rmt_nbrs, count * sizeof(id_t),
                              cudaMemcpyHostToDevice));
      CALL_CU_SAFE(cudaMalloc((void**)&((*btable_h)[bindex].values), 
                              count * msg_size));
    }
  }

  // transfer the table array
  CALL_CU_SAFE(cudaMalloc((void**)(btable_d), bcount * 
                          sizeof(grooves_box_table_t)));
  CALL_CU_SAFE(cudaMemcpy(*btable_d, (*btable_h), 
                          bcount * sizeof(grooves_box_table_t), 
                          cudaMemcpyHostToDevice));
}

PRIVATE void init_outbox_table(partition_t* partition, uint32_t pcount,
                               id_t** rmt_nbrs, int* count_per_par, 
                               size_t msg_size) {
  grooves_box_table_t* outbox = partition->outbox;
  uint32_t pid = partition->id;
  for (int rmt_pid = (pid + 1) % pcount; rmt_pid != pid; 
       rmt_pid = (rmt_pid + 1) % pcount) {
    int bindex = GROOVES_BOX_INDEX(rmt_pid, pid, pcount);
    outbox[bindex].count = count_per_par[bindex];
    if (outbox[bindex].count) {
      assert(rmt_nbrs[bindex]);
      outbox[bindex].rmt_nbrs = rmt_nbrs[bindex];
      if (partition->processor.type == PROCESSOR_CPU) {
        // Allocate the values array for the cpu-based partitions. The gpu-based
        // partitions will have their values array allocated later when their
        // state is initialized on the gpu
        outbox[bindex].values = mem_alloc(outbox[bindex].count * msg_size);
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
      (grooves_box_table_t*)calloc(pcount - 1, sizeof(grooves_box_table_t));

    if (!partition->subgraph.vertex_count || 
        !partition->subgraph.edge_count) continue;

    // identify the remote nbrs and their count per remote partition
    id_t** rmt_nbrs      = NULL;
    int*   count_per_par = NULL;
    init_get_rmt_nbrs(partition, pset->graph->vertex_count, pcount, &rmt_nbrs,
                      &count_per_par);
    // build the outbox
    if (partition->rmt_vertex_count) {
      assert(rmt_nbrs && count_per_par);
      // build the outbox tables for this partition
      init_outbox_table(partition, pcount, rmt_nbrs, count_per_par, 
                        pset->msg_size);
      free(rmt_nbrs);
    }
    free(count_per_par);
  }
}

PRIVATE void init_inbox(partition_set_t* pset) {
  uint32_t pcount = pset->partition_count;
  for (int pid = 0; pid < pcount; pid++) {
    partition_t* partition = &pset->partitions[pid];

    // each remote partition has a slot in the inbox array
    partition->inbox = 
      (grooves_box_table_t*)calloc(pcount - 1, sizeof(grooves_box_table_t));

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
      int src_bindex = GROOVES_BOX_INDEX(pid, src_pid, pcount);
      int dst_bindex = GROOVES_BOX_INDEX(src_pid, pid, pcount);      
      partition->inbox[dst_bindex] = remote_par->outbox[src_bindex];
      if (remote_par->processor.type == PROCESSOR_GPU) {
        // if the remote processor is GPU, then a values array for this inbox
        // needs to be allocated on the host
        partition->inbox[dst_bindex].values = 
          mem_alloc(partition->inbox[dst_bindex].count * pset->msg_size);
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
  grooves_box_table_t** host_outboxes = 
    (grooves_box_table_t**)calloc(pcount, sizeof(grooves_box_table_t*));

  for (int pid = 0; pid < pcount; pid++) {
    partition_t* partition = &pset->partitions[pid];
    if (partition->processor.type == PROCESSOR_GPU) {
      // set device context, create the streams and the tables for this gpu
      CALL_CU_SAFE(cudaSetDevice(partition->processor.id));
      CALL_CU_SAFE(cudaStreamCreate(&partition->streams[0]));
      CALL_CU_SAFE(cudaStreamCreate(&partition->streams[1]));
      grooves_box_table_t* outbox_h = NULL;
      init_table_gpu(partition->outbox, pcount - 1, pset->msg_size, 
                     &partition->outbox_d, &outbox_h);
      host_outboxes[pid] = partition->outbox;
      partition->outbox = outbox_h;

      grooves_box_table_t* inbox_h = NULL;
      init_table_gpu(partition->inbox, pcount - 1, pset->msg_size, 
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
      for (int bindex = 0; bindex < pcount - 1; bindex++) {
        partition_t* remote_par = &pset->partitions[(pid + 1 + bindex)%pcount];
        if (remote_par->processor.type == PROCESSOR_GPU &&
            outbox[bindex].count) {
          free(outbox[bindex].rmt_nbrs);
        }
      }
      free(host_outboxes[pid]);
    }
  }
  free(host_outboxes);
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
                                uint32_t bcount) {
  CALL_CU_SAFE(cudaFree(btable_d));
  // finalize the tables on the gpu
  for (uint32_t bindex = 0; bindex < bcount; bindex++) {
    if (btable_h[bindex].count) {
      CALL_CU_SAFE(cudaFree(btable_h[bindex].rmt_nbrs));
      CALL_CU_SAFE(cudaFree(btable_h[bindex].values));
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
      CALL_CU_SAFE(cudaStreamDestroy(partition->streams[0]));
      CALL_CU_SAFE(cudaStreamDestroy(partition->streams[1]));
      finalize_gpu_disable_peer_access(pid, pset);
      finalize_table_gpu(partition->outbox_d, partition->outbox, pcount - 1);
    } else {
      assert(partition->processor.type == PROCESSOR_CPU);
      for (uint32_t bindex = 0; bindex < pcount - 1; bindex++) {
        if (partition->outbox[bindex].count) {
          free(partition->outbox[bindex].rmt_nbrs);
          mem_free(partition->outbox[bindex].values);
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
      finalize_table_gpu(partition->inbox_d, partition->inbox, pcount - 1);
    } else {
      assert(partition->processor.type == PROCESSOR_CPU);
      for (int bindex = 0; bindex < pcount - 1; bindex++) {
        partition_t* remote_par = &pset->partitions[(pid + 1 + bindex)%pcount];
        // free only the inboxes that are the destination of an outbox of a gpu-
        // partition. Others that are destinations to a cpu-partition will be 
        // freed as an outbox in the source partition.
        if (remote_par->processor.type == PROCESSOR_GPU &&
            partition->inbox[bindex].count) {
          free(partition->inbox[bindex].rmt_nbrs);
          mem_free(partition->inbox[bindex].values);
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
      grooves_box_table_t* src_box = 
        &pset->partitions[src_pid].outbox[GROOVES_BOX_INDEX(dst_pid, src_pid, 
                                                           pcount)];
      // if the two partitions share nothing, then we have nothing to do
      if (!src_box->count) continue;

      if (pset->partitions[dst_pid].processor.type == PROCESSOR_GPU) {
        stream = &pset->partitions[dst_pid].streams[0];
      }
      grooves_box_table_t* dst_box = 
        &pset->partitions[dst_pid].inbox[GROOVES_BOX_INDEX(src_pid, dst_pid, 
                                                           pcount)];      
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