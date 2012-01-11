/* TODO(lauro,abdullah,elizeu): Add license.
 *
 * Contains unit tests for totem helper functions.
 *
 *  Created on: 2011-12-30
 *      Author: Elizeu Santos-Neto
 */

// totem includes
#include "totem_common_unittest.h"
#include "totem_comkernel.cuh"

/**
 * A simple macro to do basic true/false condition testing for kernels
 * TODO(abdullah): change the way state is tested to use standard report from 
 * the GTest framework as follows:
 * 1. to use the macro to test (so the code will be simple).
 * 2. still have the variable to store the line number where it fails or -1
 * otherwise.
 * 3. in the test fixture you would copy back the variable with the line number
 * and expects -1.
 */
#define KERNEL_EXPECT_TRUE(stmt)                \
  do {                                          \
    if (!(stmt)) {                              \
      printf("%d\n", __LINE__);                 \
      return;                                   \
    }                                           \
  } while(0)

__global__ void VerifyPartitionGPUKernel(partition_t partition, uint32_t pid, 
                                         uint32_t pcount) {
  const graph_t* subgraph = &partition.subgraph;
  const int vid = THREAD_GLOBAL_INDEX;
  if (vid >= subgraph->vertex_count) return;
  for (id_t i = subgraph->vertices[vid]; i < subgraph->vertices[vid + 1]; i++) {
    uint32_t nbr_pid = GET_PARTITION_ID(subgraph->edges[i]);
    KERNEL_EXPECT_TRUE(nbr_pid < pcount);
    uint32_t nbr_id = GET_VERTEX_ID(subgraph->edges[i]);
    if (nbr_pid == pid) KERNEL_EXPECT_TRUE(nbr_id < subgraph->vertex_count);
  }
}

void VerifyPartitionGPU(partition_set_t* partition_set_, uint32_t pid) {
  dim3 blocks, threads_per_block;
  KERNEL_CONFIGURE(partition_set_->partitions[pid].subgraph.vertex_count,
                   blocks, threads_per_block);
  VerifyPartitionGPUKernel<<<blocks, 
    threads_per_block>>>(partition_set_->partitions[pid], pid, 
                         partition_set_->partition_count);
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
    graph_ = NULL;
    partitions_ = NULL;
    partition_set_ = NULL;
    partition_count_ = 2;
    partition_processor_ = 
      (processor_t*)calloc(partition_count_, sizeof(processor_t));
    partition_processor_[0].type = PROCESSOR_CPU;
    partition_processor_[1].type = PROCESSOR_GPU;
    partition_processor_[1].id = 0;    
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
      }
    }
  }

  void TestGraph() {
    EXPECT_EQ(SUCCESS, partition_random(graph_, partition_count_, 13, 
                                        &partitions_));
    EXPECT_EQ(SUCCESS, partition_set_initialize(graph_, partitions_, 
                                                partition_processor_,
                                                partition_count_, 
                                                &partition_set_));
    uint32_t pcount = partition_set_->partition_count;
    for (uint32_t pid = 0; pid < pcount; pid++) {
      partition_t* partition = &partition_set_->partitions[pid];
      EXPECT_EQ(partition_processor_[pid].type, partition->processor.type);
      EXPECT_EQ(partition_processor_[pid].id, partition->processor.id);
      // TODO(abdullah): test the gpu-based partitions
      if (partition->processor.type == PROCESSOR_CPU) VerifyPartitionCPU(pid);
      if (partition->processor.type == PROCESSOR_GPU) 
        VerifyPartitionGPU(partition_set_, pid);
    }
  }
};

TEST_F(GraphPartitionTest , RandomPartitionInvalidPartitionNumber) {
  EXPECT_EQ(SUCCESS, graph_initialize(DATA_FOLDER("single_node.totem"),
                                      false, &graph_));
  EXPECT_EQ(FAILURE, partition_random(graph_, -1, 13, &partitions_));
}

TEST_F(GraphPartitionTest , RandomPartitionSingleNodeGraph) {
  EXPECT_EQ(SUCCESS, graph_initialize(DATA_FOLDER("single_node.totem"),
                                      false, &graph_));
  EXPECT_EQ(SUCCESS, partition_random(graph_, 10, 13, &partitions_));
  EXPECT_TRUE(partitions_[0] < 10);
}

TEST_F(GraphPartitionTest , RandomPartitionChainGraph) {
  EXPECT_EQ(SUCCESS, graph_initialize(DATA_FOLDER("chain_1000_nodes.totem"),
                                      false, &graph_));
  EXPECT_EQ(SUCCESS, partition_random(graph_, 10, 13, &partitions_));
  for (id_t i = 0; i < graph_->vertex_count; i++) {
    EXPECT_TRUE(partitions_[i] < 10);
  }
}

TEST_F(GraphPartitionTest , GetPartitionsSingleNodeGraph) {
  EXPECT_EQ(SUCCESS, graph_initialize(DATA_FOLDER("single_node.totem"),
                                      false, &graph_));
  partition_count_ = 1;
  EXPECT_EQ(SUCCESS, partition_random(graph_, partition_count_, 
                                      13, &partitions_));
    EXPECT_EQ(SUCCESS, partition_set_initialize(graph_, partitions_, 
                                                partition_processor_,
                                                partition_count_, 
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
                                              &partition_set_));
  for (int pid = 0; pid < partition_set_->partition_count; pid++) {
    partition_t* partition = &partition_set_->partitions[pid];
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
