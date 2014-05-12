

#include "totem.h"
#include "totem_alg.h"
#include "totem_util.h"

#include "../graph500.h"
#include "../xalloc.h"
#include "../generator/graph_generator.h"


int64_t int64_casval(int64_t* p, int64_t oldval, int64_t newval) {
  return __sync_val_compare_and_swap (p, oldval, newval);
}

// Counts the number of vertices in the edge list.
static int64_t find_nv (const struct packed_edge * IJ, const int64_t nedge) {
  int64_t maxvtx = -1;  
  int64_t k, gmaxvtx, tmaxvtx = -1;  
  for (k = 0; k < nedge; ++k) {
    if (get_v0_from_edge(&IJ[k]) > tmaxvtx)
      tmaxvtx = get_v0_from_edge(&IJ[k]);
    if (get_v1_from_edge(&IJ[k]) > tmaxvtx)
      tmaxvtx = get_v1_from_edge(&IJ[k]);
  }
  gmaxvtx = maxvtx;
  while (tmaxvtx > gmaxvtx)
    gmaxvtx = int64_casval (&maxvtx, gmaxvtx, tmaxvtx);
  
  return 1+maxvtx;
}

static void allocate_graph(vid_t vertex_count, eid_t edge_count, 
                           graph_t** graph_ret) {
  // Allocate the main structure
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
#pragma omp parallel for
  for (vid_t v = 0; v < graph->vertex_count; v++) {
    vid_t* nbrs = &graph->edges[graph->vertices[v]];
    qsort(nbrs, graph->vertices[v+1] - graph->vertices[v], sizeof(vid_t),
          compare_ids);
  }
}

// Global reference to the graph structure that will be used in the callbacks
// provided by Totem to the Graph500 benchmark.
static graph_t* graph = NULL;

// TODO(scott): A parallel way to create the graph.
static void create_graph(const struct packed_edge * IJ, vid_t vertex_count,
                         eid_t edge_count) {
  // The graph is undirected, hence the number of edges allocated is multiplied
  // by 2 as edges in Totem's graph representation are considered directed.
  allocate_graph(vertex_count, edge_count * 2, &graph);

  eid_t* degree = (eid_t*)calloc(vertex_count, sizeof(eid_t));
  for(eid_t i = 0; i < edge_count; i++) {
    degree[get_v0_from_edge(&IJ[i])]++;
    degree[get_v1_from_edge(&IJ[i])]++;
  }

  graph->vertices[0] = 0;
  for (vid_t i = 1; i <= vertex_count; i++) {
    graph->vertices[i] = graph->vertices[i - 1] + degree[i - 1];
  }
  
  for (eid_t i = 0; i < edge_count; i++) {
    vid_t u = get_v0_from_edge(&IJ[i]);
    vid_t v = get_v1_from_edge(&IJ[i]);

    // one direction
    eid_t pos = degree[u]--;
    graph->edges[graph->vertices[u] + pos - 1] = v;

    // other direction
    pos = degree[v]--;
    graph->edges[graph->vertices[v] + pos - 1] = u;
  }

  sort_nbrs(graph);
  free(degree);
}

extern "C" {
int create_graph_from_edgelist(struct packed_edge *IJ, int64_t nedge) {
  eid_t edge_count = nedge;
  vid_t vertex_count = find_nv(IJ, nedge);
  create_graph(IJ, vertex_count, edge_count);

  // TODO(scott): allow the settings below to be set via command line.
  totem_attr_t attr       = TOTEM_DEFAULT_ATTR;
  attr.par_algo           = PAR_SORTED_ASC; // LOW (DSC) HIGH (ASC)
  attr.cpu_par_share      = .52;
  attr.platform           = PLATFORM_HYBRID;
  attr.gpu_count          = 1;
  attr.gpu_graph_mem      = GPU_GRAPH_MEM_DEVICE;
  attr.gpu_par_randomized = true;
  attr.push_msg_size      = (sizeof(vid_t) * BITS_PER_BYTE) + 1;
  attr.pull_msg_size      = MSG_SIZE_ZERO;
  attr.alloc_func         = graph500_alloc;
  attr.free_func          = graph500_free;

  // Partitions the graph and loads the GPU-partitions
  CALL_SAFE(totem_init(graph, &attr));

  return 0;
}
  
  
int make_bfs_tree(int64_t *bfs_tree_out, int64_t *max_vtx_out,
                  int64_t srcvtx) {
  *max_vtx_out = graph->vertex_count - 1;
  CALL_SAFE(graph500_hybrid(srcvtx, bfs_tree_out));
  return 0;
}
  
void destroy_graph (void) {
  totem_finalize();
  graph_finalize(graph);
  graph = NULL;
}
}
