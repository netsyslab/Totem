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

__global__ void degree_kernel(partition_t par) {
  vid_t v = THREAD_GLOBAL_INDEX;
  if (v >= par.subgraph.vertex_count) return;
  for (eid_t i = par.subgraph.vertices[v];
       i < par.subgraph.vertices[v + 1]; i++) {
    vid_t nbr = par.subgraph.edges[i];
    int* dst = engine_get_dst_ptr(par.id, nbr, par.outbox, 
                                  (int*)par.algo_state);
    atomicAdd(dst, 1);
  }
}

void degree_gpu(partition_t* par) {
  dim3 blocks, threads;
  KERNEL_CONFIGURE(par->subgraph.vertex_count, blocks, threads);
  degree_kernel<<<blocks, threads, 1, par->streams[1]>>>(*par);
  CALL_CU_SAFE(cudaGetLastError());
}

void degree_cpu(partition_t* par) {
  OMP(omp parallel for)
  for (vid_t v = 0; v < par->subgraph.vertex_count; v++) {
    for (eid_t i = par->subgraph.vertices[v];
         i < par->subgraph.vertices[v + 1]; i++) {
      vid_t nbr = par->subgraph.edges[i];
      int* dst = engine_get_dst_ptr(par->id, nbr, par->outbox, 
                                    (int*)par->algo_state);
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
    engine_report_not_finished();
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
  OMP(omp parallel for)
  for (vid_t v = 0; v < par->subgraph.vertex_count; v++) {
    degree_g[par->map[v]] = src[v];
  }
}

class EngineTest : public TestWithParam<totem_attr_t*> {
 protected:
  graph_t* _graph;
  engine_config_t _config;
  totem_attr_t _attr;
  virtual void SetUp() {
    // Ensure the minimum CUDA architecture is supported
    CUDA_CHECK_VERSION();
    _graph = NULL;
    engine_config_t config  = {
      NULL, degree, degree_scatter, NULL, degree_init, degree_finalize, 
      degree_aggr, GROOVES_PUSH};
    _config = config;
    _attr = *GetParam();
    _attr.push_msg_size = MSG_SIZE_WORD;
  }

  virtual void TearDown() {
    if (_graph != NULL) {
      graph_finalize(_graph);
    }
  }

  void TestGraph(const char* graph_str) {
    graph_initialize(graph_str, false, &_graph);
    EXPECT_FALSE(_graph->directed);
    EXPECT_EQ(SUCCESS, engine_init(_graph, &_attr));
    EXPECT_EQ(SUCCESS, engine_config(&_config));
    degree_g = (int*)calloc(_graph->vertex_count, sizeof(int));
    if (engine_largest_gpu_partition()) {
      totem_malloc(engine_largest_gpu_partition() * sizeof(int), 
                   TOTEM_MEM_HOST_PINNED, (void**)&degree_h);
    }
    EXPECT_EQ(SUCCESS, engine_execute());
    engine_finalize();
    if (engine_largest_gpu_partition()) { 
      totem_free(degree_h, TOTEM_MEM_HOST_PINNED);
    }
    for (vid_t v = 0; v < _graph->vertex_count; v++) {
      int nbr_count = _graph->vertices[v + 1] - _graph->vertices[v];
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

TEST_P(EngineTest, InvalidGPUCount) {
  graph_initialize(DATA_FOLDER("chain_1000_nodes.totem"), false, &_graph);
  totem_attr_t attr = TOTEM_DEFAULT_ATTR;
  attr.gpu_count = get_gpu_count() + 1;
  EXPECT_EQ(FAILURE, engine_init(_graph, &attr));
  attr.platform = PLATFORM_GPU;
  EXPECT_EQ(FAILURE, engine_init(_graph, &attr));
}

// From Google documentation:
// In order to run value-parameterized tests, we need to instantiate them,
// or bind them to a list of values which will be used as test parameters.
//
// Values() receives a list of parameters and the framework will execute the
// whole set of tests BFSTest for each element of Values()
INSTANTIATE_TEST_CASE_P(EngineTestAllPlatforms, EngineTest,
                        Values(&totem_attrs[0],
                               &totem_attrs[1],
                               &totem_attrs[2],
                               &totem_attrs[3],
                               &totem_attrs[4],
                               &totem_attrs[5],
                               &totem_attrs[6]));

#else

// From Google documentation:
// Google Test may not support value-parameterized tests with some
// compilers. This dummy test keeps gtest_main linked in.
TEST_P(DummyTest, ValueParameterizedTestsAreNotSupportedOnThisPlatform) {}

#endif  // GTEST_HAS_PARAM_TEST
