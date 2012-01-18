/**
 * Implements the Grooves interface.
 *
 *  Created on: 2011-01-05
 *  Author: Abdullah Gharaibeh
 */

// totem includes
#include "totem_comkernel.cuh"
#include "totem_grooves.h"
#include "totem_partition.h"

PRIVATE void init_get_remote_nbrs(partition_t* partition, int pid, 
                                  uint32_t vertex_count, uint32_t pcount, 
                                  id_t** nbrs, uint32_t** count_per_par, 
                                  uint32_t* count_total) {
  graph_t* subgraph = &(partition->subgraph);

  // This is a temporary hash table to identify the remote neighbors.
  // It is initialized with conservative space such that it can accommodate
  // the extreme case where all vertices in other partitions are remote to
  // this partition
  hash_table_t* ht;
  CALL_SAFE(hash_table_initialize_cpu(vertex_count - subgraph->vertex_count, 
                                      &ht));
  *count_per_par = (uint32_t*)calloc(pcount - 1, sizeof(uint32_t));
  uint32_t remote_nbrs_count = 0;
  for (id_t vid = 0; vid < subgraph->vertex_count; vid++) {
    for (id_t i = subgraph->vertices[vid]; 
         i < subgraph->vertices[vid + 1]; i++) {
      id_t nbr = subgraph->edges[i];
      int nbr_pid = GET_PARTITION_ID(nbr);
      if (nbr_pid != pid) {
        int index;
        if (hash_table_get_cpu(ht, nbr, &index) == FAILURE) {
          CALL_SAFE(hash_table_put_cpu(ht, nbr, 1));
          remote_nbrs_count++;
          int bindex = GROOVES_BOX_INDEX(nbr_pid, pid, pcount);
          (*count_per_par)[bindex]++;
        }
      }
    }
  }
  if (remote_nbrs_count) {
    CALL_SAFE(hash_table_get_keys_cpu(ht, nbrs, count_total));
    assert(*count_total == remote_nbrs_count);
  }
  hash_table_finalize_cpu(ht);
}

PRIVATE void init_allocate_table(grooves_box_table_t* btable, uint32_t pid, 
                                 uint32_t pcount, uint32_t* count_per_par) {
  // initialize the outbox hash tables
  for (int remote_pid = pid + 1; remote_pid != pid; 
       remote_pid = (remote_pid + 1) % pcount) {
    int bindex = GROOVES_BOX_INDEX(remote_pid, pid, pcount);
    if (count_per_par[bindex]) {
      CALL_SAFE(hash_table_initialize_cpu(count_per_par[bindex], 
                                          &(btable[bindex].ht)));
    }
  }
}

PRIVATE void init_table_gpu(grooves_box_table_t* btable, uint32_t bcount, 
                            grooves_box_table_t** btable_d) {
  // initialize the tables on the gpu
  for (uint32_t bindex = 0; bindex < bcount; bindex++) {
    hash_table_t hash_table_d;
    if (btable[bindex].ht.size) {
      CALL_SAFE(hash_table_initialize_gpu(&(btable[bindex].ht), &hash_table_d));
      hash_table_finalize_cpu(&(btable[bindex].ht));
      btable[bindex].ht = hash_table_d;
    }
  }

  // transfer the table array
  CALL_CU_SAFE(cudaMalloc((void**)(btable_d), bcount * 
                          sizeof(grooves_box_table_t)));
  CALL_CU_SAFE(cudaMemcpy(*btable_d, btable, 
                          bcount * sizeof(grooves_box_table_t), 
                          cudaMemcpyHostToDevice));
}

PRIVATE void init_outbox_table(partition_t* partition, uint32_t pid, 
                               uint32_t pcount, uint32_t* remote_nbrs,
                               uint32_t count_total) {
  grooves_box_table_t* outbox = partition->outbox;
  for (uint32_t i = 0; i < count_total; i++) {
    uint32_t nbr = remote_nbrs[i];
    uint32_t nbr_pid = GET_PARTITION_ID(nbr);
    int bindex = GROOVES_BOX_INDEX(nbr_pid, pid, pcount);
    int vindex;
    assert(hash_table_get_cpu(&(outbox[bindex].ht), nbr, &vindex) == FAILURE);
    CALL_SAFE(hash_table_put_cpu(&(outbox[bindex].ht), nbr, 
                                 outbox[bindex].count));
    outbox[bindex].count++;
  }

  if (partition->processor.type == PROCESSOR_GPU) {
    grooves_box_table_t* outbox_d = NULL;
    init_table_gpu(partition->outbox, pcount - 1, &outbox_d);
    free(partition->outbox);
    partition->outbox = outbox_d;
  }
}

PRIVATE void init_outbox(partition_set_t* pset) {
  uint32_t pcount = pset->partition_count;
  for (int pid = 0; pid < pcount; pid++) {
    partition_t* partition = &pset->partitions[pid];
    if (!partition->subgraph.vertex_count || 
        !partition->subgraph.edge_count) continue;

    // each remote partition has a slot in the outbox array
    partition->outbox = 
      (grooves_box_table_t*)calloc(pcount - 1, sizeof(grooves_box_table_t));

    // identify the remote nbrs and their count per remote partition
    id_t*     remote_nbrs   = NULL;
    uint32_t* count_per_par = NULL;
    uint32_t  count_total   = 0;
    init_get_remote_nbrs(partition, pid, pset->graph->vertex_count, pcount,
                         &remote_nbrs, &count_per_par, &count_total);

    // build the outbox
    if (count_total) {
      assert(remote_nbrs && count_per_par);
      // initialize the outbox hash tables
      init_allocate_table(partition->outbox, pid, pcount, count_per_par);
      // build the outbox hash tables
      init_outbox_table(partition, pid, pcount, remote_nbrs, count_total);
      free(remote_nbrs);
      free(count_per_par);
    }
  }
}

error_t grooves_initialize(partition_set_t* pset) {  
  init_outbox(pset);
  // TODO(abdullah): inbox the inbox stubs
  return SUCCESS;
}

PRIVATE void finalize_table_gpu(grooves_box_table_t* btable_d, 
                                uint32_t bcount) {
  // transfer the table array
  grooves_box_table_t* btable_h = 
    (grooves_box_table_t*)calloc(bcount, sizeof(grooves_box_table_t));
  CALL_CU_SAFE(cudaMemcpy(btable_h, btable_d, 
                          bcount * sizeof(grooves_box_table_t), 
                          cudaMemcpyDeviceToHost));
  cudaFree(btable_d);

  // finalize the tables on the gpu
  for (uint32_t bindex = 0; bindex < bcount; bindex++) {
    hash_table_finalize_gpu(&(btable_h[bindex].ht));
  }
  free(btable_h);
}

PRIVATE void finalize_table_cpu(grooves_box_table_t* btable, uint32_t bcount) {
  for (uint32_t pid = 0; pid < bcount; pid++) {
    if (btable[pid].ht.size) hash_table_finalize_cpu(&(btable[pid].ht));
  }
  free(btable);
}

PRIVATE void finalize_outbox(partition_set_t* pset) {
  uint32_t pcount = pset->partition_count;
  for (int pid = 0; pid < pcount; pid++) {
    partition_t* partition = &pset->partitions[pid];
    if (!partition->subgraph.vertex_count || 
        !partition->subgraph.edge_count) continue;
    assert(partition->outbox);
    if (partition->processor.type == PROCESSOR_GPU) {
      finalize_table_gpu(partition->outbox, pcount - 1);
    } else {
      assert(partition->processor.type == PROCESSOR_CPU);
      finalize_table_cpu(partition->outbox, pcount - 1);
    }
  }
}

error_t grooves_finalize(partition_set_t* pset) {
  finalize_outbox(pset);
  // TODO(abdullah): finalize the inbox state
  return SUCCESS;
}
