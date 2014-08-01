/**
 * Main entry of the benchmark
 *
 *  Created on: 2011-02-28
 *  Author: Abdullah Gharaibeh
 */

// totem includes
#include "totem_benchmark.h"
#include "totem_mem.h"

// Defines attributes of the algorithms available for benchmarking
PRIVATE void benchmark_bfs(graph_t*, void*, totem_attr_t*);
PRIVATE void benchmark_pagerank(graph_t*, void*, totem_attr_t*);
PRIVATE void benchmark_dijkstra(graph_t*, void*, totem_attr_t*);
PRIVATE void benchmark_betweenness(graph_t*, void*, totem_attr_t*);
PRIVATE void benchmark_graph500(graph_t* graph, void* tree, totem_attr_t* attr);
PRIVATE void benchmark_clustering_coefficient(graph_t* graph, void*, 
                                              totem_attr_t* attr);
const benchmark_attr_t BENCHMARKS[] = {
  {
    benchmark_bfs,
    "BFS",
    sizeof(cost_t),
    true,
    1,
    MSG_SIZE_ZERO,
    NULL,
    NULL
  },
  {
    benchmark_pagerank,
    "PAGERANK",
    sizeof(rank_t),
    true,
    MSG_SIZE_ZERO,
    sizeof(rank_t) * BITS_PER_BYTE,
    NULL,
    NULL
  },
  {
    benchmark_dijkstra,
    "DIJKSTRA",
    sizeof(weight_t),
    false,
    sizeof(weight_t) * BITS_PER_BYTE,
    MSG_SIZE_ZERO,
    NULL,
    NULL
  },
  {
    benchmark_betweenness,
    "BETWEENNESS",
    sizeof(weight_t),
    true,
    sizeof(uint32_t) * BITS_PER_BYTE,
    sizeof(score_t) * BITS_PER_BYTE,
    NULL,
    NULL
  },
  {
    benchmark_graph500,
    "GRAPH500",
    sizeof(bfs_tree_t),
    true,
    (sizeof(vid_t) * BITS_PER_BYTE) + 1,
    MSG_SIZE_ZERO,
    graph500_alloc,
    graph500_free
  }, 
  {
    benchmark_clustering_coefficient,
    "CLUSTERING_COEFFICIENT",
    sizeof(weight_t),
    false,
    MSG_SIZE_ZERO,
    MSG_SIZE_ZERO,
    NULL,
    NULL 
  }
};


// A reference to the options used to configure the benchmark
PRIVATE benchmark_options_t* options = NULL;
PRIVATE const int SEED = 1985;

/**
 * Returns the number of traversed edges used in computing the processing rate
 */
PRIVATE uint64_t get_traversed_edges(graph_t* graph, void* benchmark_output) {
  uint64_t trv_edges = 0;
  switch(options->benchmark) {
    case BENCHMARK_BFS: {
      OMP(omp parallel for reduction(+ : trv_edges))
      for (vid_t vid = 0; vid < graph->vertex_count; vid++) {
        cost_t* cost = (cost_t*)benchmark_output;
        if (cost[vid] != INF_COST) {
          trv_edges += (graph->vertices[vid + 1] - graph->vertices[vid]);
        }
      }
      break;
    }
    case BENCHMARK_DIJKSTRA: {
      OMP(omp parallel for reduction(+ : trv_edges))
      for (vid_t vid = 0; vid < graph->vertex_count; vid++) {
        weight_t* distance = (weight_t*)benchmark_output;
        if (distance[vid] != WEIGHT_MAX) {
          trv_edges += (graph->vertices[vid + 1] - graph->vertices[vid]);
        }
      }
      break;
    }
    case BENCHMARK_PAGERANK: {
      trv_edges = (uint64_t)graph->edge_count * PAGE_RANK_ROUNDS;
      break;
    }
    case BENCHMARK_BETWEENNESS: {
      // The two is for the two phases: forward and backward
      trv_edges = (uint64_t)graph->edge_count * 2;
      break;
    }
    case BENCHMARK_CLUSTERING_COEFFICIENT: {
      // Each triangle is counted three times
      trv_edges = (uint64_t)graph->edge_count;
      break;
    }
    default: {
      trv_edges = graph->edge_count;
      break;
    }
  }
  return trv_edges;
}

/**
 * Randomly picks a random source vertex, and ensures that it is connected to at
 * least one other vertex.
 */
PRIVATE vid_t get_random_src(graph_t* graph) {
  vid_t src = rand() % graph->vertex_count;
  while ((graph->vertices[src + 1] - graph->vertices[src] == 0)) {
    src = rand() % graph->vertex_count;
  }
  return src;
}

/**
 * Runs BFS benchmark according to Graph500 specification
 */
PRIVATE void benchmark_bfs(graph_t* graph, void* cost, totem_attr_t* attr) {
  CALL_SAFE(bfs_hybrid(get_random_src(graph), (cost_t*)cost));
}

/**
 * Runs PageRank benchmark
 */
PRIVATE void benchmark_pagerank(graph_t* graph, void* rank, 
                                totem_attr_t* attr) {
  CALL_SAFE(page_rank_incoming_hybrid(NULL, (rank_t*)rank));
}

/**
 * Runs Dijkstra benchmark
 */
PRIVATE void benchmark_dijkstra(graph_t* graph, void* distance,
                                totem_attr_t* attr) {
  if (options->platform == PLATFORM_CPU) {
    CALL_SAFE(dijkstra_cpu(graph, get_random_src(graph), (weight_t*)distance));
  } else if (options->platform == PLATFORM_GPU && options->gpu_count == 1) {
    CALL_SAFE(dijkstra_vwarp_gpu(graph, get_random_src(graph), 
                                 (weight_t*)distance));
  } else {
    assert(false);
  }
}

/**
 * Runs Betweenness Centrality benchmark
 */
PRIVATE void benchmark_betweenness(graph_t* graph, void* betweenness_score, 
                                   totem_attr_t* attr) {
  CALL_SAFE(betweenness_hybrid(CENTRALITY_SINGLE,
                               (score_t*)betweenness_score));
}

PRIVATE
void benchmark_graph500(graph_t* graph, void* tree, totem_attr_t* attr) {
  CALL_SAFE(graph500_hybrid(get_random_src(graph), (bfs_tree_t*)tree));
}

/**
 * Runs Clustering Coefficient benchmark
 */

PRIVATE
void benchmark_clustering_coefficient(graph_t* graph, void* coefficients, 
                                      totem_attr_t* attr) {
  if (options->platform == PLATFORM_CPU) {
    CALL_SAFE(clustering_coefficient_sorted_neighbours_cpu(
              graph, (weight_t**)&coefficients));
  } else if (options->platform == PLATFORM_GPU && options->gpu_count == 1) {
      CALL_SAFE(clustering_coefficient_sorted_neighbours_gpu(
              graph, (weight_t**)&coefficients));
  } else {
      assert(false);
  }
}

/**
 * The main execution loop of the benchmark
 */
PRIVATE void benchmark_run() {
  assert(options);
  
  graph_t* graph = NULL;
  CALL_SAFE(graph_initialize(options->graph_file, 
                             (options->benchmark == BENCHMARK_DIJKSTRA),
                             &graph));
  print_config(graph, options, BENCHMARKS[options->benchmark].name);

  void* benchmark_state = NULL;
  totem_malloc(graph->vertex_count * BENCHMARKS[options->benchmark].output_size,
               TOTEM_MEM_HOST, (void**)&benchmark_state);
  assert(benchmark_state || (BENCHMARKS[options->benchmark].output_size == 0));

  bool totem_based = BENCHMARKS[options->benchmark].has_totem;
  totem_attr_t attr = TOTEM_DEFAULT_ATTR;
  if (totem_based) {
    attr.par_algo = options->par_algo;
    attr.cpu_par_share = (float)options->alpha / 100.0;
    attr.platform = options->platform;
    attr.gpu_count = options->gpu_count;
    attr.gpu_graph_mem = options->gpu_graph_mem;
    attr.gpu_par_randomized = options->gpu_par_randomized;
    attr.sorted = options->sorted;
    attr.push_msg_size = BENCHMARKS[options->benchmark].push_msg_size;
    attr.pull_msg_size = BENCHMARKS[options->benchmark].pull_msg_size;
    attr.alloc_func = BENCHMARKS[options->benchmark].alloc_func;
    attr.free_func = BENCHMARKS[options->benchmark].free_func;
    CALL_SAFE(totem_init(graph, &attr));
  }

  // Configure OpenMP 
  omp_set_num_threads(options->thread_count);
  omp_set_schedule(options->omp_sched, 0);
  print_header(graph, totem_based);

  srand(SEED);
  for (int s = 0; s < options->repeat; s++) {
    totem_timing_reset();
    stopwatch_t stopwatch;
    stopwatch_start(&stopwatch);
    BENCHMARKS[options->benchmark].func(graph, benchmark_state, &attr);
    print_timing(graph, stopwatch_elapsed(&stopwatch), 
                 get_traversed_edges(graph, benchmark_state), totem_based);
  }

  if (totem_based) {
    totem_finalize();
  }
  totem_free(benchmark_state, TOTEM_MEM_HOST);
  CALL_SAFE(graph_finalize(graph));
}

void benchmark_check_configuration() {
  if (!BENCHMARKS[options->benchmark].has_totem) {
    if (options->platform == PLATFORM_HYBRID) {
      fprintf(stderr, "Error: No hybrid implementation for benchmark %s\n",
              BENCHMARKS[options->benchmark].name);
      exit(-1);
    } else if (options->platform == PLATFORM_GPU && options->gpu_count > 1) {
      fprintf(stderr, "Error: No multi-GPU implementation for benchmark %s\n",
              BENCHMARKS[options->benchmark].name);
      exit(-1);
    }
  }
}

/**
 * The main entry of the program
 */
int main(int argc, char** argv) {
  CALL_SAFE(check_cuda_version());
  options = benchmark_cmdline_parse(argc, argv);
  benchmark_check_configuration();
  benchmark_run();
  return 0;
}
