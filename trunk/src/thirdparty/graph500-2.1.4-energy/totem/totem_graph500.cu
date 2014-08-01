// Totem includes.
#include "totem.h"
#include "totem_alg.h"
#include "totem_util.h"
#include "totem_benchmark.h"

// Graph500 includes.
#include "../graph500.h"
#include "../xalloc.h"
#include "../generator/graph_generator.h"

// Counts the number of vertices in the edge list.
static int64_t find_nv (const struct packed_edge * IJ, const int64_t nedge) {
  int64_t maxvtx = -1;

  // Offers a minimal speed-up by doing a parallel maximum.
  OMP(omp parallel for shared (maxvtx))
  for (int64_t k = 0; k < nedge; k++) {
    int64_t localmax = -1;

    if (get_v0_from_edge(&IJ[k]) > maxvtx)
      localmax = get_v0_from_edge(&IJ[k]);
    if (get_v1_from_edge(&IJ[k]) > (localmax > maxvtx ? localmax : maxvtx))
      localmax = get_v1_from_edge(&IJ[k]);

    if (localmax > maxvtx) {   // Avoid critical section if possible.
      OMP(omp critical(updatemax))
      {
        if (localmax > maxvtx) // Final update inside critical section.
          maxvtx = localmax;
      }
    }
  }

  return 1 + maxvtx;
}

static void allocate_graph(vid_t vertex_count, eid_t edge_count,
                           graph_t** graph_ret) {
  // Allocate the main structure.
  graph_t* graph = (graph_t*)calloc(1, sizeof(graph_t));
  assert(graph);

  // Allocate the buffers. An extra slot is allocated in the vertices array to
  // make it easy to calculate the number of neighbors of the last vertex.
  graph->vertices = (eid_t*)malloc((vertex_count + 1) * sizeof(eid_t));
  graph->edges    = (vid_t*)malloc(edge_count * sizeof(vid_t));

  // Ensure buffer allocation
  assert(graph->vertices && graph->edges);

  // Set the member variables
  graph->vertex_count = vertex_count;
  graph->edge_count = edge_count;
  *graph_ret = graph;
}

// Sorts the neighbours in ascending order.
void sort_nbrs(graph_t* graph) {
OMP(omp parallel for schedule(guided))
  for (vid_t v = 0; v < graph->vertex_count; v++) {
    vid_t* nbrs = &graph->edges[graph->vertices[v]];
    qsort(nbrs, graph->vertices[v+1] - graph->vertices[v], sizeof(vid_t),
          compare_ids_asc);
  }
}

// Global reference to the graph structure that will be used in the callbacks
// provided by Totem to the Graph500 benchmark.
static graph_t* graph = NULL;

// Creates a Totem graph from a graph500 graph.
static void create_graph(struct packed_edge* IJ, vid_t vertex_count,
                         eid_t edge_count) {
  // The graph is undirected, hence the number of edges allocated is multiplied
  // by 2 as edges in Totem's graph representation are considered directed.
  allocate_graph(vertex_count, edge_count * 2, &graph);

  eid_t* degree = (eid_t*)calloc(vertex_count, sizeof(eid_t));
  assert(degree);

  // Calculate the degree of each vertex.
  OMP(omp parallel)
  {
    OMP(omp for nowait)
    for(eid_t i = 0; i < edge_count; i++) {
      vid_t v0 = get_v0_from_edge(&IJ[i]);
      OMP(omp atomic)
        degree[v0]++;
    }

    OMP(omp for nowait)
    for(eid_t i = 0; i < edge_count; i++) {
      vid_t v1 = get_v1_from_edge(&IJ[i]);
      OMP(omp atomic)
        degree[v1]++;
    }
  }

  // Calculate the running total (prefix sum) of the degree for each vertex.
  graph->vertices[0] = 0;
  for (vid_t i = 1; i <= vertex_count; i++) {
    graph->vertices[i] = graph->vertices[i - 1] + degree[i - 1];
  }

  // Build the Totem edges, one in each direction.
  OMP(omp parallel)
  {
    OMP(omp for nowait)
    for (eid_t i = 0; i < edge_count; i++) {
      vid_t u = get_v0_from_edge(&IJ[i]);
      vid_t v = get_v1_from_edge(&IJ[i]);
      eid_t pos;

      // one direction
      OMP(omp atomic capture)
        { pos = degree[u]; degree[u]--; }
      graph->edges[graph->vertices[u] + pos - 1] = v;
    }

    OMP(omp for nowait)
    for (eid_t i = 0; i < edge_count; i++) {
      vid_t u = get_v0_from_edge(&IJ[i]);
      vid_t v = get_v1_from_edge(&IJ[i]);
      eid_t pos;

      // other direction
      OMP(omp atomic capture)
        { pos = degree[v]; degree[v]--; }
      graph->edges[graph->vertices[v] + pos - 1] = u;
    }
  }

  sort_nbrs(graph);
  free(degree);
}

extern "C" {

/**
 *  Parses totem flags from a string.
 *
 *  @param input_optarg   The input arguments, as a string.
 *  @param program_name   The runtime program name, for the first argument
 *                        to be passed into the next parser.
 */
void totem_set_options(const char* input_optarg, char* program_name) {
  char* totem_args;
  int   new_argc;
  char* new_argv[32]; // Probably don't need more than 32 arguments?
  char* tokened_args;

  new_argv[0] = program_name;              // Copy program name.
  new_argc = 1;                            // argc count starts at 1.

  totem_args = (char*) malloc(sizeof(char) * strlen(input_optarg));
  strcpy(totem_args, input_optarg);        // Since strtok modifies string,
                                           // don't globber the original.

  tokened_args = strtok(totem_args, " ");  // Tokenize over spaces.
  while (tokened_args) {
    new_argv[new_argc++] = tokened_args;   // Add all words to the argv.
    tokened_args = strtok(NULL, " ");
    assert(new_argc < 32);
  }

  new_argv[new_argc++] = "/dev/null";     // Don't need a graph file.
  new_argv[new_argc] = NULL;              // End of args.

  // Parse Totem benchmarking options. No need to retrieve the result here.
  benchmark_cmdline_parse(new_argc, new_argv);

  free(totem_args);
  return;
}

int create_graph_from_edgelist(struct packed_edge* IJ, int64_t nedge) {
  eid_t edge_count   = nedge;
  vid_t vertex_count = find_nv(IJ, nedge);
  create_graph(IJ, vertex_count, edge_count);

  // Use the options specified by created benchmark options.
  benchmark_options_t* b_options = totem_benchmark_get_options();
  assert(b_options);

  totem_attr_t attr = TOTEM_DEFAULT_ATTR;
  attr.par_algo           = b_options->par_algo;
  attr.cpu_par_share      = b_options->alpha / 100.0;
  attr.platform           = b_options->platform;
  attr.gpu_count          = b_options->gpu_count;
  attr.gpu_graph_mem      = b_options->gpu_graph_mem;
  attr.gpu_par_randomized = b_options->gpu_par_randomized;
  attr.sorted             = b_options->sorted;

  // OpenMP attributes.
  omp_set_num_threads(b_options->thread_count);
  omp_set_schedule(b_options->omp_sched, 0);

  // Static graph500 attributes.
  attr.push_msg_size      = (sizeof(vid_t) * BITS_PER_BYTE) + 1;
  attr.pull_msg_size      = MSG_SIZE_ZERO;
  attr.alloc_func         = graph500_alloc;
  attr.free_func          = graph500_free;

  // Free the edge list to free up space for creating Totem's partitined graph.
  free(IJ);

  // Print out the configurations.
  print_config(graph, totem_benchmark_get_options(), "GRAPH500");

  // Partitions the graph and loads the GPU-partitions.
  CALL_SAFE(totem_init(graph, &attr));
  print_header(graph, true);

  return 0;
}

int make_bfs_tree(int64_t* bfs_tree_out, int64_t* max_vtx_out,
                  int64_t srcvtx) {
  totem_timing_reset();
  stopwatch_t stopwatch;
  stopwatch_start(&stopwatch);
  *max_vtx_out = graph->vertex_count - 1;
  CALL_SAFE(graph500_hybrid(srcvtx, bfs_tree_out));
  print_timing(graph, stopwatch_elapsed(&stopwatch), graph->edge_count, true);
  return 0;
}

void destroy_graph() {
  totem_finalize();
  graph_finalize(graph);
  graph = NULL;
}

} // End of extern "C"
