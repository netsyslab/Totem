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
void init_get_rmt_nbrs_list(partition_t* par, vid_t vcount, uint32_t pcount,
                            vid_t** nbrs, int* count_per_par) {
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
  for (vid_t vid = 0; vid < subg->vertex_count; vid++) {
    for (eid_t i = subg->vertices[vid]; i < subg->vertices[vid + 1]; i++) {
      vid_t nbr = subg->edges[i];
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
void init_map_rmt_nbrs(partition_t* par, uint32_t pcount, vid_t* nbrs,
                       int* count_per_par, hash_table_t** ht) {
  // Sort the ids. This significantly improves the performance of
  // memory scatter operations and prefetching by improving data locality
  // (hence cache hit rate)
  qsort(nbrs, par->rmt_vertex_count, sizeof(vid_t), compare_ids);

  // Build the hash table map
  CALL_SAFE(hash_table_initialize_cpu(par->rmt_vertex_count, ht));
  vid_t cur_rmt_v = 0;
  while (cur_rmt_v < par->rmt_vertex_count) {
    int count = count_per_par[GET_PARTITION_ID(nbrs[cur_rmt_v])];
    OMP(omp parallel for)
    for (int i = 0; i < count; i++) {
      CALL_SAFE(hash_table_put_cpu(*ht, nbrs[cur_rmt_v + i], i));
    }
    cur_rmt_v += count;
  }
}

PRIVATE
void init_map_rmt_nbrs(partition_t* par, uint32_t pcount, vid_t* nbrs,
                       int* count_per_par, hash_table_t* ht, vid_t** rmt_nbrs) {
  // allocate the state, it is per remote partition
  memset(rmt_nbrs, 0, MAX_PARTITION_COUNT * sizeof(vid_t*));
  for (int i = 0; i < MAX_PARTITION_COUNT; i++) {
    if (count_per_par[i]) {
      rmt_nbrs[i] = (vid_t*)calloc(count_per_par[i], sizeof(vid_t));
      assert(rmt_nbrs[i]);
    }
  }
  // create the final mapping
  OMP(omp parallel for)
  for (vid_t v = 0; v < par->rmt_vertex_count; v++) {
    int index; HT_LOOKUP(ht, nbrs[v], index);
    vid_t* pnbrs = rmt_nbrs[GET_PARTITION_ID(nbrs[v])];
    pnbrs[index] = GET_VERTEX_ID(nbrs[v]);
  }
}

PRIVATE
void init_update_subgraph(partition_t* par, hash_table_t* ht) {
  graph_t* subg = &(par->subgraph);
  OMP(omp parallel for)
  for (vid_t vid = 0; vid < subg->vertex_count; vid++) {
    for (eid_t i = subg->vertices[vid]; i < subg->vertices[vid + 1]; i++) {
      vid_t nbr = subg->edges[i];
      int nbr_pid = GET_PARTITION_ID(nbr);
      if (nbr_pid != par->id) {
        int new_nbr_id;
        HT_LOOKUP(ht, nbr, new_nbr_id);
        assert(new_nbr_id != -1);
        subg->edges[i] = SET_PARTITION_ID((vid_t)new_nbr_id, nbr_pid);
      }
    }
  }
}

PRIVATE
void init_get_rmt_nbrs(partition_t* par, vid_t vcount, uint32_t pcount,
                       vid_t** rmt_nbrs, int* count_per_par) {
  // First, identify the remote neighbors to this partition. "nbrs" is a flat
  // array that stores the list of all remote neighbors, irrespective of which
  // remote partition they belong to. "count_per_par" array is the number of
  // remote neighbors per remote partition
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

PRIVATE void init_outbox_table(partition_t* partition, uint32_t pcount,
                               vid_t** rmt_nbrs, int* count_per_par,
                               size_t push_msg_size, size_t pull_msg_size) {
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
        if (push_msg_size > 0) {           
          CALL_SAFE(totem_malloc(bits_to_bytes(outbox[rmt_pid].count * 
                                               push_msg_size),
                                 TOTEM_MEM_HOST_PINNED,
                                 &(outbox[rmt_pid].push_values)));
        }
        if (pull_msg_size > 0) {
          CALL_SAFE(totem_malloc(bits_to_bytes(outbox[rmt_pid].count * 
                                               pull_msg_size),
                                 TOTEM_MEM_HOST_PINNED, 
                                 &(outbox[rmt_pid].pull_values)));
        }
      }
    }
  }
}

PRIVATE void init_outbox(partition_set_t* pset) {
  uint32_t pcount = pset->partition_count;
  for (int pid = 0; pid < pcount; pid++) {
    partition_t* partition = &pset->partitions[pid];

    if (!partition->subgraph.vertex_count ||
        !partition->subgraph.edge_count) continue;

    // identify the remote nbrs and their count per remote partition
    vid_t* rmt_nbrs[MAX_PARTITION_COUNT];
    int count_per_par[MAX_PARTITION_COUNT];
    init_get_rmt_nbrs(partition, pset->graph->vertex_count, pcount, rmt_nbrs,
                      count_per_par);
    // build the outbox
    if (partition->rmt_vertex_count) {
      // build the outbox tables for this partition
      init_outbox_table(partition, pcount, rmt_nbrs, count_per_par,
                        pset->push_msg_size, pset->pull_msg_size);
    }
  }
}

PRIVATE void init_inbox(partition_set_t* pset) {
  uint32_t pcount = pset->partition_count;
  for (int pid = 0; pid < pcount; pid++) {
    partition_t* partition = &pset->partitions[pid];

    if (!partition->subgraph.vertex_count) continue;

    for (int src_pid = (pid + 1) % pcount; src_pid != pid;
         src_pid = (src_pid + 1) % pcount) {
      partition_t* remote_par = &pset->partitions[src_pid];
      // An inbox in a partition is an outbox in the source partition.
      // Therefore, we just need to copy the state of the already built
      // source partition's outbox into the destination partition's inbox.
      partition->inbox[src_pid] = remote_par->outbox[pid];
      if (remote_par->processor.type == PROCESSOR_GPU) {
        // if the remote processor is GPU, then a values array for this inbox
        // needs to be allocated on the host
        if (pset->push_msg_size > 0) {
          CALL_SAFE(totem_malloc(bits_to_bytes(partition->inbox[src_pid].count *
                                               pset->push_msg_size), 
                                 TOTEM_MEM_HOST_PINNED, 
                                 &(partition->inbox[src_pid].push_values)));
          CALL_SAFE(totem_malloc(bits_to_bytes(partition->inbox[src_pid].count *
                                               pset->push_msg_size), 
                                 TOTEM_MEM_HOST_PINNED, 
                                 &(partition->inbox[src_pid].push_values_s)));
        }
        if (pset->pull_msg_size > 0) {
          CALL_SAFE(totem_malloc(bits_to_bytes(partition->inbox[src_pid].count *
                                               pset->pull_msg_size), 
                                 TOTEM_MEM_HOST_PINNED, 
                                 &(partition->inbox[src_pid].pull_values)));
          CALL_SAFE(totem_malloc(bits_to_bytes(partition->inbox[src_pid].count *
                                               pset->pull_msg_size), 
                                 TOTEM_MEM_HOST_PINNED, 
                                 &(partition->inbox[src_pid].pull_values_s)));
        }
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

PRIVATE void init_table_gpu(partition_t* par, partition_set_t* pset,
                            bool inbox) {
  // set device context, create the tables for this gpu
  CALL_CU_SAFE(cudaSetDevice(par->processor.id));
  grooves_box_table_t* box = par->outbox;
  if (inbox) box = par->inbox;
  // initialize the tables on the gpu
  for (uint32_t rmt_pid = 0; rmt_pid < pset->partition_count; rmt_pid++) {
    if (rmt_pid == par->id) continue;
    vid_t count = box[rmt_pid].count;
    if (count) {
      vid_t* rmt_nbrs = box[rmt_pid].rmt_nbrs;
      CALL_CU_SAFE(cudaMalloc((void**)&(box[rmt_pid].rmt_nbrs),
                              count * sizeof(vid_t)));
      CALL_CU_SAFE(cudaMemcpy(box[rmt_pid].rmt_nbrs, rmt_nbrs, 
                              count * sizeof(vid_t), cudaMemcpyDefault));
      if((pset->partitions[rmt_pid].processor.type == PROCESSOR_GPU) &&
         inbox) {
        free(rmt_nbrs);
      }
      if (pset->push_msg_size > 0) {
        CALL_CU_SAFE(cudaMalloc((void**)&(box[rmt_pid].push_values),
                                bits_to_bytes(count * pset->push_msg_size)));
        if (inbox) {
          CALL_CU_SAFE(cudaMalloc((void**)&(box[rmt_pid].push_values_s),
                                  bits_to_bytes(count * pset->push_msg_size)));
        }
      }
      if (pset->pull_msg_size > 0) {
        CALL_CU_SAFE(cudaMalloc((void**)&(box[rmt_pid].pull_values),
                                bits_to_bytes(count * pset->pull_msg_size)));
        if (inbox) {
          CALL_CU_SAFE(cudaMalloc((void**)&(box[rmt_pid].pull_values_s),
                                  bits_to_bytes(count * pset->pull_msg_size)));
        }
      }
    }
  }
}

PRIVATE void init_gpu_state(partition_set_t* pset) {
  for (int pid = 0; pid < pset->partition_count; pid++) {
    partition_t* partition = &pset->partitions[pid];
    if (partition->processor.type == PROCESSOR_GPU) {
      init_table_gpu(partition, pset, false);
    }      
  }
  for (int pid = 0; pid < pset->partition_count; pid++) {
    partition_t* partition = &pset->partitions[pid];
    if (partition->processor.type == PROCESSOR_GPU) {
      init_table_gpu(partition, pset, true);
      init_gpu_enable_peer_access(pid, pset);
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

PRIVATE void finalize_table_gpu(partition_set_t* pset, 
                                grooves_box_table_t* btable, bool inbox) {
  // finalize the tables on the gpu
  for (uint32_t pid = 0; pid < pset->partition_count; pid++) {
    if (btable[pid].count) {
      CALL_CU_SAFE(cudaFree(btable[pid].rmt_nbrs));
      if (pset->push_msg_size > 0) {
        CALL_CU_SAFE(cudaFree(btable[pid].push_values));
        if (inbox) {
          CALL_CU_SAFE(cudaFree(btable[pid].push_values_s));
        }
      }
      if (pset->pull_msg_size > 0) {
        CALL_CU_SAFE(cudaFree(btable[pid].pull_values));
        if (inbox) {
          CALL_CU_SAFE(cudaFree(btable[pid].pull_values_s));
        }
      }
    }
  }
}

PRIVATE
void finalize_gpu_disable_peer_access(uint32_t pid, partition_set_t* pset) {
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
    if (partition->processor.type == PROCESSOR_GPU) {
      CALL_CU_SAFE(cudaSetDevice(partition->processor.id));
      finalize_gpu_disable_peer_access(pid, pset);
      finalize_table_gpu(pset, partition->outbox, false);
    } else {
      assert(partition->processor.type == PROCESSOR_CPU);
      for (uint32_t rmt_pid = 0; rmt_pid < pcount; rmt_pid++) {
        if (rmt_pid == pid) continue;
        if (partition->outbox[rmt_pid].count) {
          free(partition->outbox[rmt_pid].rmt_nbrs);
          if (pset->push_msg_size > 0) {
            totem_free(partition->outbox[rmt_pid].push_values,
                       TOTEM_MEM_HOST_PINNED);
          }
          if (pset->pull_msg_size > 0) {
            totem_free(partition->outbox[rmt_pid].pull_values, 
                       TOTEM_MEM_HOST_PINNED);
          }
        }
      }
    }
  }
}

PRIVATE void finalize_inbox(partition_set_t* pset) {
  uint32_t pcount = pset->partition_count;
  for (int pid = 0; pid < pcount; pid++) {
    partition_t* partition = &pset->partitions[pid];
    if (partition->processor.type == PROCESSOR_GPU) {
      CALL_CU_SAFE(cudaSetDevice(partition->processor.id));
      finalize_table_gpu(pset, partition->inbox, true);
    } else {
      assert(partition->processor.type == PROCESSOR_CPU);
      for (int rmt_pid = 0; rmt_pid < pcount; rmt_pid++) {
        if (rmt_pid == pid) continue;
        partition_t* remote_par = &pset->partitions[rmt_pid];
        // free only the inboxes that are the destination of an outbox of a gpu-
        // partition. Others that are destinations to a cpu-partition will be
        // freed as an outbox in the source partition.
        if (remote_par->processor.type == PROCESSOR_GPU &&
            partition->inbox[rmt_pid].count) {
          free(partition->inbox[rmt_pid].rmt_nbrs);
          if (pset->push_msg_size > 0) {
            totem_free(partition->inbox[rmt_pid].push_values, 
                       TOTEM_MEM_HOST_PINNED);
            totem_free(partition->inbox[rmt_pid].push_values_s, 
                       TOTEM_MEM_HOST_PINNED);
          }
          if (pset->pull_msg_size > 0) {
            totem_free(partition->inbox[rmt_pid].pull_values, 
                       TOTEM_MEM_HOST_PINNED);
            totem_free(partition->inbox[rmt_pid].pull_values_s, 
                       TOTEM_MEM_HOST_PINNED);
          }
        }
      }
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

PRIVATE
void launch_communications_setup(partition_set_t* pset, 
                                  grooves_direction_t direction, int local_pid,
                                  int remote_pid, void** src, void** dst, 
                                  vid_t* count, size_t* msg_size, 
                                  cudaStream_t** stream) {
  if (direction == GROOVES_PUSH) {
    *msg_size = pset->push_msg_size;
    *src = pset->partitions[local_pid].outbox[remote_pid].push_values;
    *dst = pset->partitions[remote_pid].inbox[local_pid].push_values_s;
    *count = pset->partitions[local_pid].outbox[remote_pid].count;
    if (pset->partitions[local_pid].processor.type == PROCESSOR_GPU) {
      CALL_CU_SAFE(cudaSetDevice(pset->partitions[local_pid].processor.id));
      *stream = &pset->partitions[local_pid].streams[1];
    } else {
      CALL_CU_SAFE(cudaSetDevice(pset->partitions[remote_pid].processor.id));
      *stream = &pset->partitions[remote_pid].streams[0];
    }
  } else if (direction == GROOVES_PULL) {
    *msg_size = pset->pull_msg_size;
    *src = pset->partitions[local_pid].inbox[remote_pid].pull_values_s;
    *dst = pset->partitions[remote_pid].outbox[local_pid].pull_values;
    *count = pset->partitions[local_pid].inbox[remote_pid].count;

    if (pset->partitions[remote_pid].processor.type == PROCESSOR_GPU) {
      CALL_CU_SAFE(cudaSetDevice(pset->partitions[remote_pid].processor.id));
      *stream = &pset->partitions[remote_pid].streams[1];
    } else {
      CALL_CU_SAFE(cudaSetDevice(pset->partitions[local_pid].processor.id));
      *stream = &pset->partitions[local_pid].streams[0];
    }
  } else {
    printf("Direction not supported: %s", direction); 
    fflush(stdout);
    exit(EXIT_FAILURE);
  }
}

error_t grooves_launch_communications(partition_set_t* pset, int pid, 
                                      grooves_direction_t direction) {
  uint32_t pcount = pset->partition_count;
  for (int remote_pid = (pid + 1) % pcount; remote_pid != pid;
       remote_pid = (remote_pid + 1) % pcount) {
    // if both partitions are on the host, then, by design the source
    // partition's outbox is shared with the destination partition's inbox,
    // hence no need to copy data
    if ((pset->partitions[pid].processor.type == PROCESSOR_CPU) &&
        (pset->partitions[remote_pid].processor.type == PROCESSOR_CPU)) {
      continue;
    }

    size_t msg_size = 0;
    void* src = NULL;
    void* dst = NULL;
    vid_t count = 0;
    cudaStream_t* stream = NULL;
    launch_communications_setup(pset, direction, pid, remote_pid,
                                &src, &dst, &count, &msg_size, &stream);
    
    if (count == 0) continue;
    CALL_CU_SAFE(cudaMemcpyAsync(dst, src, bits_to_bytes(count * msg_size),
                                 cudaMemcpyDefault, *stream));
  }
  return SUCCESS;
}

error_t grooves_synchronize(partition_set_t* pset, 
                            grooves_direction_t direction) {
  for (int pid = 0; pid < pset->partition_count; pid++) {
    if (pset->partitions[pid].processor.type == PROCESSOR_GPU) {
      CALL_CU_SAFE(cudaSetDevice(pset->partitions[pid].processor.id));
      CALL_CU_SAFE(cudaStreamSynchronize(pset->partitions[pid].streams[0]));
      CALL_CU_SAFE(cudaStreamSynchronize(pset->partitions[pid].streams[1]));
    }
  }
  if (pset->partition_count <= 1) return SUCCESS;
  for (int pid = 0; pid < pset->partition_count; pid++) {
    partition_t* par = &pset->partitions[pid];
    for (int rmt_pid = 0; rmt_pid < pset->partition_count; rmt_pid++) {
      if (rmt_pid == pid) continue;
      // For push-based communication
      void* tmp = par->inbox[rmt_pid].push_values;
      par->inbox[rmt_pid].push_values = par->inbox[rmt_pid].push_values_s;
      par->inbox[rmt_pid].push_values_s = tmp;

      // For pull-based communication
      tmp = par->inbox[rmt_pid].pull_values;
      par->inbox[rmt_pid].pull_values = par->inbox[rmt_pid].pull_values_s;
      par->inbox[rmt_pid].pull_values_s = tmp;
    }
  }
  return SUCCESS;
}
