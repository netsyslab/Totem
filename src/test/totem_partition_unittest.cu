/* TODO(lauro,abdullah,elizeu): Add license.
 *
 * Contains unit tests for partition.
 *
 *  Created on: 2011-12-30
 *      Author: Elizeu Santos-Neto
 */

// totem includes
#include "totem_common_unittest.h"
#include "totem_comkernel.cuh"
#include "totem_grooves.h"
#include "totem_partition.h"

__global__ void VerifyPartitionGPUKernel(partition_t partition, uint32_t pid, 
                                         uint32_t pcount) {
  const graph_t* subgraph = &partition.subgraph;
  const int vid = THREAD_GLOBAL_INDEX;
  if (vid >= subgraph->vertex_count) return;
  for (id_t i = subgraph->vertices[vid]; 
       i < subgraph->vertices[vid + 1]; i++) {
    uint32_t nbr = subgraph->edges[i];
    uint32_t nbr_pid = GET_PARTITION_ID(nbr);
    KERNEL_EXPECT_TRUE(nbr_pid < pcount);
    if (nbr_pid != pid) {
      grooves_box_table_t* outbox =
        &partition.outbox_d[GROOVES_BOX_INDEX(nbr_pid, pid, pcount)];
      KERNEL_EXPECT_TRUE(outbox->count > 0);
    }
  }
}

__global__ void VerifyPartitionInboxGPUKernel(partition_t partition, 
                                              uint32_t pid, uint32_t pcount) {
  const int index = THREAD_GLOBAL_INDEX;
  for (int r = 0; r < pcount - 1; r++) {
    grooves_box_table_t* inbox = &partition.inbox_d[r];
    if (index >= inbox->count) continue;
    KERNEL_EXPECT_TRUE(inbox->rmt_nbrs[index] < 
                       partition.subgraph.vertex_count);
  }
}

__global__ void CheckInboxValuesGPUKernel(uint32_t pid, int* values, 
                                          uint32_t count) {
  const int index = THREAD_GLOBAL_INDEX;
  if (index >= count) return;
  KERNEL_EXPECT_TRUE(values[index] == pid);
}

class GraphPartitionTest : public ::testing::Test {
 protected:
  graph_t* graph_;
  id_t* partitions_;
  uint32_t partition_count_;
  processor_t* partition_processor_;
  partition_set_t* partition_set_;

  virtual void SetUp() {
    // Ensure the minimum CUDA architecture is supported
    CUDA_CHECK_VERSION();
    int gpu_count = 0;
    CALL_CU_SAFE(cudaGetDeviceCount(&gpu_count));
    graph_ = NULL;
    partitions_ = NULL;
    partition_set_ = NULL;
    partition_count_ = gpu_count + 1;
    partition_processor_ = 
      (processor_t*)calloc(partition_count_, sizeof(processor_t));
    partition_processor_[0].type = PROCESSOR_CPU;
    for (int gpu = 0; gpu < gpu_count; gpu++) {
      partition_processor_[gpu + 1].type = PROCESSOR_GPU;
      partition_processor_[gpu + 1].id = gpu;
    }
  }

  virtual void TearDown() {
    free(partition_processor_);
    if (graph_ != NULL) {
      graph_finalize(graph_);
    }
    if (partitions_ != NULL) {
      free(partitions_);
    }
    if (partition_set_ != NULL) {
      EXPECT_EQ(SUCCESS, partition_set_finalize(partition_set_));
    }
  }

  void VerifyPartitionGPU(uint32_t pid) {
    ASSERT_EQ(cudaSuccess, 
              cudaSetDevice(partition_set_->partitions[pid].processor.id));
    dim3 blocks, threads_per_block;
    KERNEL_CONFIGURE(partition_set_->partitions[pid].subgraph.vertex_count, 
                     blocks, threads_per_block);
    VerifyPartitionGPUKernel<<<blocks, 
      threads_per_block>>>(partition_set_->partitions[pid], pid, 
                           partition_set_->partition_count);
    ASSERT_EQ(cudaSuccess, cudaGetLastError());
    ASSERT_EQ(cudaSuccess, cudaThreadSynchronize());

    VerifyPartitionInboxGPUKernel<<<blocks, 
      threads_per_block>>>(partition_set_->partitions[pid], pid, 
                           partition_set_->partition_count);
    ASSERT_EQ(cudaSuccess, cudaGetLastError());
    ASSERT_EQ(cudaSuccess, cudaThreadSynchronize());
  }

  void VerifyPartitionCPU(uint32_t pid) {
    partition_t* partition = &partition_set_->partitions[pid];
    graph_t* subgraph = &partition->subgraph;
    uint32_t pcount = partition_set_->partition_count;
    for (id_t vid = 0; vid < subgraph->vertex_count; vid++) {
      for (id_t i = subgraph->vertices[vid]; 
           i < subgraph->vertices[vid + 1]; i++) {
        uint32_t nbr_pid = GET_PARTITION_ID(subgraph->edges[i]);
        EXPECT_TRUE((nbr_pid < pcount));
        partition_t* nbr_partition = &partition_set_->partitions[nbr_pid];
        id_t nbr_id = GET_VERTEX_ID(subgraph->edges[i]);
        EXPECT_TRUE((nbr_id < nbr_partition->subgraph.vertex_count));
        if (nbr_pid != pid) {
          grooves_box_table_t* outbox = 
            &partition->outbox[GROOVES_BOX_INDEX(nbr_pid, pid, pcount)];
          EXPECT_GT(outbox->count, (uint32_t)0);
        }
      }
    }
    // verify inbox tables, all the vertices in the table must belong to this 
    // partition
    for (int r = 0; r < pcount - 1; r++) {
      grooves_box_table_t* inbox = &partition->inbox[r];
      for (int index = 0; index < inbox->count; index++) {
        KERNEL_EXPECT_TRUE(inbox->rmt_nbrs[index] < 
                           partition->subgraph.vertex_count);
      }
    }
  }

  void TestState() {
    uint32_t pcount = partition_set_->partition_count;
    for (uint32_t pid = 0; pid < pcount; pid++) {
      partition_t* partition = &partition_set_->partitions[pid];
      EXPECT_EQ(partition_processor_[pid].type, partition->processor.type);
      EXPECT_EQ(partition_processor_[pid].id, partition->processor.id);
      if (partition->processor.type == PROCESSOR_CPU) VerifyPartitionCPU(pid);
      if (partition->processor.type == PROCESSOR_GPU) VerifyPartitionGPU(pid);
    }
  }

  void InitOutboxValues() {
    for (uint32_t pid = 0; pid < partition_set_->partition_count; pid++) {
      partition_t* partition = &partition_set_->partitions[pid];
      EXPECT_EQ(pid, partition->id);
      uint32_t pcount = partition_set_->partition_count;
      for (uint32_t remote_pid = (pid + 1) % pcount; remote_pid != pid; 
           remote_pid = (remote_pid + 1) % pcount) {
        grooves_box_table_t* remote_outbox = 
          &partition->outbox[GROOVES_BOX_INDEX(remote_pid, pid, pcount)];
        if (remote_outbox->count == 0) continue;
        if (partition->processor.type == PROCESSOR_GPU) {
          dim3 blocks, threads_per_block;
          KERNEL_CONFIGURE(remote_outbox->count, blocks, threads_per_block);
          memset_device<<<blocks, threads_per_block>>>
            ((int*)remote_outbox->values, (int)remote_pid, 
             remote_outbox->count);
          ASSERT_EQ(cudaSuccess, cudaGetLastError());
          ASSERT_EQ(cudaSuccess, cudaThreadSynchronize());
        } else {
          ASSERT_EQ(PROCESSOR_CPU, partition->processor.type);
          int* values = (int*)remote_outbox->values;
          for (int i = 0; i < remote_outbox->count; i++) {
            values[i] = remote_pid;
          }
        }
      }
    }
  }
  
  void CheckInboxValues() {
    for (uint32_t pid = 0; pid < partition_set_->partition_count; pid++) {
      partition_t* partition = &partition_set_->partitions[pid];
      grooves_box_table_t* inbox = partition->inbox;
      uint32_t bcount = partition_set_->partition_count - 1;
      for (uint32_t bindex = 0; bindex < bcount; bindex++) {
        if (inbox[bindex].count == 0) continue;
        if (partition->processor.type == PROCESSOR_GPU) {
          dim3 blocks, threads_per_block;
          KERNEL_CONFIGURE(inbox[bindex].count, blocks, threads_per_block);
          CheckInboxValuesGPUKernel<<<blocks, threads_per_block>>>
            (pid, (int*)inbox[bindex].values, inbox[bindex].count);
          ASSERT_EQ(cudaSuccess, cudaGetLastError());
          ASSERT_EQ(cudaSuccess, cudaThreadSynchronize());
        } else {
          ASSERT_EQ(PROCESSOR_CPU, partition->processor.type);
          for (uint32_t bindex = 0; bindex < bcount; bindex++) {
            int* values = (int*)inbox[pid].values;
            for (int i = 0; i < inbox[bindex].count; i++) {
              EXPECT_EQ(pid, values[i]);
            }
          }
        }
      }
    }
  }
  
  void TestCommunication() {
    InitOutboxValues();
    EXPECT_EQ(SUCCESS, grooves_launch_communications(partition_set_));
    EXPECT_EQ(SUCCESS, grooves_synchronize(partition_set_));
    CheckInboxValues();
  }

  void TestGraph() {
    EXPECT_EQ(SUCCESS, partition_random(graph_, partition_count_, NULL, 13, 
                                        &partitions_));
    EXPECT_EQ(SUCCESS, partition_set_initialize(graph_, partitions_, 
                                                partition_processor_,
                                                partition_count_, 
                                                sizeof(int),
                                                &partition_set_));
    TestState();
    TestCommunication();
  }
};

TEST_F(GraphPartitionTest , RandomPartitionInvalidPartitionNumber) {
  EXPECT_EQ(SUCCESS, graph_initialize(DATA_FOLDER("single_node.totem"),
                                      false, &graph_));
  EXPECT_EQ(FAILURE, partition_random(graph_, -1, NULL, 13, &partitions_));
}

TEST_F(GraphPartitionTest , RandomPartitionFractionInvalidFraction) {
  EXPECT_EQ(SUCCESS, graph_initialize(DATA_FOLDER("single_node.totem"),
                                      false, &graph_));
  float* partition_fraction = (float *) calloc(2, sizeof(float));
  partition_fraction[0] = 2.0;
  partition_fraction[1] = -1.0; // Invalid fraction
  EXPECT_EQ(FAILURE, partition_random(graph_, 2, partition_fraction, 13, 
                                      &partitions_));
  partition_fraction[0] = 0.8;
  partition_fraction[1] = 0.1; // Invalid fraction sum
  EXPECT_EQ(FAILURE, partition_random(graph_, 2, partition_fraction, 13,
                                      &partitions_));
  free(partition_fraction);
}

TEST_F(GraphPartitionTest , RandomPartitionSingleNodeGraph) {
  EXPECT_EQ(SUCCESS, graph_initialize(DATA_FOLDER("single_node.totem"),
                                      false, &graph_));
  EXPECT_EQ(SUCCESS, partition_random(graph_, 10, NULL, 13, &partitions_));
  EXPECT_TRUE(partitions_[0] < 10);
}

TEST_F(GraphPartitionTest , RandomPartitionFractionSingleNodeGraph) {
  EXPECT_EQ(SUCCESS, graph_initialize(DATA_FOLDER("single_node.totem"),
                                      false, &graph_));
  float* partition_fraction = (float *) calloc(10, sizeof(float));
  for (int i = 0; i < 10; i++) {
    partition_fraction[i] = (1.0 / 10);
  }
  EXPECT_EQ(SUCCESS, partition_random(graph_, 10, partition_fraction, 13,
                                      &partitions_));
  EXPECT_TRUE(partitions_[0] < 10);
  EXPECT_EQ(9, partitions_[0]);
  free(partition_fraction);
}

TEST_F(GraphPartitionTest , RandomPartitionChainGraph) {
  EXPECT_EQ(SUCCESS, graph_initialize(DATA_FOLDER("chain_1000_nodes.totem"),
                                      false, &graph_));
  EXPECT_EQ(SUCCESS, partition_random(graph_, 10, NULL, 13, &partitions_));
  for (id_t i = 0; i < graph_->vertex_count; i++) {
    EXPECT_TRUE(partitions_[i] < 10);
  }
}

TEST_F(GraphPartitionTest , RandomPartitionFractionChainGraph) {
  EXPECT_EQ(SUCCESS, graph_initialize(DATA_FOLDER("chain_1000_nodes.totem"),
                                      false, &graph_));
  float* partition_fraction = (float *) calloc(10, sizeof(float));
  for (int i = 0; i < 10; i++) {
    partition_fraction[i] = (1.0 / 10);
  }
  EXPECT_EQ(SUCCESS, partition_random(graph_, 10, partition_fraction, 13,
                                      &partitions_));
  for (id_t i = 0; i < graph_->vertex_count; i++) {
    EXPECT_TRUE(partitions_[i] < 10);
  }
  free(partition_fraction);
}

TEST_F(GraphPartitionTest , GetPartitionsSingleNodeGraph) {
  EXPECT_EQ(SUCCESS, graph_initialize(DATA_FOLDER("single_node.totem"),
                                      false, &graph_));
  partition_count_ = 1;
  EXPECT_EQ(SUCCESS, partition_random(graph_, partition_count_, NULL,
                                      13, &partitions_));
  EXPECT_EQ(SUCCESS, partition_set_initialize(graph_, partitions_, 
                                              partition_processor_,
                                              partition_count_, 
                                              sizeof(int),
                                              &partition_set_));
  EXPECT_EQ(partition_set_->partition_count, 1);
  partition_t* partition = &partition_set_->partitions[0];
  EXPECT_EQ(partition->subgraph.vertex_count, (uint32_t)1);
  EXPECT_EQ(partition->subgraph.edge_count, (uint32_t)0);
}

TEST_F(GraphPartitionTest, GetPartitionsChainGraph) {
  graph_initialize(DATA_FOLDER("chain_1000_nodes.totem"), false, &graph_);
  TestGraph();
}

TEST_F(GraphPartitionTest, GetPartitionsStarGraph) {
  graph_initialize(DATA_FOLDER("star_1000_nodes.totem"), false, &graph_);
  TestGraph();
}

TEST_F(GraphPartitionTest, GetPartitionsCompleteGraph) {
  graph_initialize(DATA_FOLDER("complete_graph_300_nodes.totem"), 
                   false, &graph_);
  TestGraph();
}

TEST_F(GraphPartitionTest, GetPartitionsImbalancedChainGraph) {
  EXPECT_EQ(SUCCESS, graph_initialize(DATA_FOLDER("chain_1000_nodes.totem"),
                                      false, &graph_));
  // set the processor of all partitions to CPU
  for (uint32_t pid = 0; pid < partition_count_; pid++) {
    partition_processor_[pid].type = PROCESSOR_CPU; 
  }
  // Divide the graph in two partitions, one node in one partition and the 
  // other 999 in the second partition.
  partitions_ = (id_t*)calloc(1000, sizeof(id_t));
  partitions_[0] = 1;
  EXPECT_EQ(SUCCESS, partition_set_initialize(graph_, partitions_, 
                                              partition_processor_,
                                              partition_count_, 
                                              sizeof(int),
                                              &partition_set_));
  for (int pid = 0; pid < partition_set_->partition_count; pid++) {
    partition_t* partition = &partition_set_->partitions[pid];
    EXPECT_EQ(pid, partition->id);
    for (id_t vid = 0; vid < partition->subgraph.vertex_count; vid++) {
      // Only the vertex-0 and vertex-999 in the original graph have a single
      // neighbor. Vertex-0 is in partition-1, and vertex-999 is renamed to 998
      // in partition-0.
      id_t expected = (pid == 1 || vid == 998 ? 1 : 2);
      EXPECT_EQ(expected, 
                partition->subgraph.vertices[vid + 1] - 
                partition->subgraph.vertices[vid]);
    }
  }
}
