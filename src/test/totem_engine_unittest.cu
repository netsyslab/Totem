/**
 * Contains unit tests for the execution engine.
 *
 *  Created on: 2012-02-09
 *      Author: Abdullah
 */

// totem includes
#include "totem_common_unittest.h"
#include "totem_engine.cuh"

#if GTEST_HAS_PARAM_TEST

using ::testing::TestWithParam;
using ::testing::Values;

int* degree_g;
int* degree_h;

__global__ void degree_kernel(partition_t par, int pcount) {
  id_t v = THREAD_GLOBAL_INDEX;
  if (v >= par.subgraph.vertex_count) return;
  for (id_t i = par.subgraph.vertices[v]; 
       i < par.subgraph.vertices[v + 1]; i++) {
    int* dst;
    id_t nbr = par.subgraph.edges[i];
    ENGINE_FETCH_DST(par.id, nbr, par.outbox_d, (int*)par.algo_state, 
                     pcount, dst, int);
    atomicAdd(dst, 1);
  }
}

void degree_gpu(partition_t* par) {
  int pcount = engine_partition_count();
  dim3 blocks, threads;
  KERNEL_CONFIGURE(par->subgraph.vertex_count, blocks, threads);
  degree_kernel<<<blocks, threads, 1, par->streams[1]>>>(*par, pcount);
  CALL_CU_SAFE(cudaGetLastError());
}

void degree_cpu(partition_t* par) {
  int pcount = engine_partition_count();
  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (id_t v = 0; v < par->subgraph.vertex_count; v++) {
    for (id_t i = par->subgraph.vertices[v]; 
         i < par->subgraph.vertices[v + 1]; i++) {
      int* dst;
      id_t nbr = par->subgraph.edges[i];
      ENGINE_FETCH_DST(par->id, nbr, par->outbox, (int*)par->algo_state, 
                       pcount, dst, int);
      __sync_fetch_and_add(dst, 1);
    }
  }
}

void degree(partition_t* par) {
  if (engine_superstep() == 1) {
    if (par->processor.type == PROCESSOR_GPU) { 
      degree_gpu(par);      
    } else {
      assert(par->processor.type == PROCESSOR_CPU);
      degree_cpu(par);
    }
  } else {
    engine_report_finished(par->id);
  }
}

void degree_scatter(partition_t* par) {
  int* pstate = (int*)par->algo_state;
  engine_scatter_inbox_add(par->id, pstate);
}

void degree_init(partition_t* par) {
  uint64_t vcount = par->subgraph.vertex_count;
  if (par->processor.type == PROCESSOR_GPU) {
    CALL_CU_SAFE(cudaMalloc(&(par->algo_state), vcount * sizeof(int)));
    ASSERT_TRUE(par->algo_state);
    CALL_CU_SAFE(cudaMemset(par->algo_state, 0, vcount * sizeof(int)));    
  } else {
    ASSERT_TRUE(par->processor.type == PROCESSOR_CPU);    
    par->algo_state = calloc(vcount, sizeof(int));
    ASSERT_TRUE(par->algo_state);
  }
  engine_set_outbox(par->id, 0);
}

void degree_finalize(partition_t* par) {
  int* pstate = (int*)par->algo_state;
  ASSERT_TRUE(pstate);
  if (par->processor.type == PROCESSOR_GPU) {
    CALL_CU_SAFE(cudaFree(pstate));
  } else {
    ASSERT_EQ(PROCESSOR_CPU, par->processor.type);
    free(pstate);
  }
  par->algo_state = NULL;
}

void degree_aggr(partition_t* par) {
  int* src = NULL;
  if (par->processor.type == PROCESSOR_GPU) {
    CALL_CU_SAFE(cudaMemcpy(degree_h, par->algo_state,
                            par->subgraph.vertex_count * sizeof(int),
                            cudaMemcpyDefault));
    src = degree_h;
  } else {
    ASSERT_EQ(PROCESSOR_CPU, par->processor.type);
    src = (int*)par->algo_state;
  }
  // aggregate the results
  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (id_t v = 0; v < par->subgraph.vertex_count; v++) {
    degree_g[par->map[v]] = src[v];
  }
}

class EngineTest : public TestWithParam<platform_t> {
 protected:
  graph_t* graph_;
  engine_config_t config_;
  virtual void SetUp() {
    // Ensure the minimum CUDA architecture is supported
    CUDA_CHECK_VERSION();
    graph_ = NULL;
    engine_config_t config  = {
      NULL,
      PAR_RANDOM,
      sizeof(int),
      GetParam(),
      NULL,
      degree,
      degree_scatter,
      degree_init,
      degree_finalize,
      degree_aggr
    };
    config_ = config;
  }

  virtual void TearDown() {
    if (graph_ != NULL) {
      graph_finalize(graph_);
    }
  }

  void TestGraph(const char* graph_str) {
    graph_initialize(graph_str, false, &graph_);
    EXPECT_FALSE(graph_->directed);
    config_.graph = graph_;
    engine_init(&config_);
    degree_g = (int*)calloc(graph_->vertex_count, sizeof(int));
    if (engine_largest_gpu_partition()) {
      degree_h = (int*)mem_alloc(engine_largest_gpu_partition() * sizeof(int));
    }
    engine_execute();
    if (engine_largest_gpu_partition()) mem_free(degree_h);
    for (id_t v = 0; v < graph_->vertex_count; v++) {
      int nbr_count = graph_->vertices[v + 1] - graph_->vertices[v];
      EXPECT_EQ(nbr_count, degree_g[v]);
    }
    free(degree_g);
  }
};

TEST_P(EngineTest, ChainGraph) {
  TestGraph(DATA_FOLDER("chain_1000_nodes.totem"));
}

TEST_P(EngineTest, StarGraph) {
  TestGraph(DATA_FOLDER("star_1000_nodes.totem"));
}

TEST_P(EngineTest, CompleteGraph) {
  TestGraph(DATA_FOLDER("complete_graph_300_nodes.totem"));
}


// From Google documentation:
// In order to run value-parameterized tests, we need to instantiate them,
// or bind them to a list of values which will be used as test parameters.
//
// Values() receives a list of parameters and the framework will execute the
// whole set of tests BFSTest for each element of Values()
INSTANTIATE_TEST_CASE_P(EngineTestAllPlatforms, EngineTest, 
                        Values(PLATFORM_CPU,       // on the CPU only
                               PLATFORM_GPU,       // on one GPU only
                               PLATFORM_MULTI_GPU, // all available GPUs
                               PLATFORM_HYBRID,    // on CPU and one GPU
                               PLATFORM_ALL));     // on CPU and all GPUs
                               

#else

// From Google documentation:
// Google Test may not support value-parameterized tests with some
// compilers. This dummy test keeps gtest_main linked in.
TEST_P(DummyTest, ValueParameterizedTestsAreNotSupportedOnThisPlatform) {}

#endif  // GTEST_HAS_PARAM_TEST
