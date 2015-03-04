/**
 * Defines the graph interface. Mainly the data-structure, and its initialize
 * and finalize methods.
 *
 * The following is the totem graph file format template:
 *
 * # NODES: vertex_count [Y]
 * # EDGES: edge_count
 * # DIRECTED|UNDIRECTED
 * [VERTEX LIST]
 * [EDGE LIST]
 *
 * The first three lines specify the vertex and edge counts, whether the
 * graph is directed or not and whether the graph has a vertex list.
 * Note that the flag [Y] after vertex_count indicates that a vertex list
 * should be expected.
 *
 * The vertices are assumed to have numerical IDs that ranges from 0 to
 * vertex_count. The vertices are sorted in an increasing order.
 *
 * A vertex list is an optional list that defines a value for each vertex.
 * Each line in the vertex list defines the value associated with a vertex
 * as follows: "VERTEX_ID VALUE". The parser expects the vertex ids to be sorted
 * in the vertex list. Although a value is not needed for each vertex, a value
 * for the last vertex (i.e., vertex id = vertex_count - 1) is required as it is
 * used as an end-of-list signal. If a value does not exist for a vertex, it
 * will be assigned a default one.
 *
 * An edge list represents the edges in the graph. Each line describes a single
 * edge, optionally with a weight as follows: "SOURCE DESTINATION [WEIGHT]". If
 * the weight does not exist, it will be assigned a default value.
 *
 *  Created on: 2011-02-28
 *  Author: Abdullah Gharaibeh
 */
#ifndef TOTEM_GRAPH_H
#define TOTEM_GRAPH_H

// totem includes
#include "totem_comdef.h"

// TODO(elizeu): We should define a #-directive to allow multiple definitions
//               of the following two types and related constants. The goal is
//               to allow clients to set on compile time the specific type their
//               applications will use.

// Specifies an id type.
// We have two id types (vid_t and eid_t). The rule to use them is as follows:
// anything that is constrained by the number of vertices should be defined
// using the vid_t type, similarly anything that is constrained by the number
// of edges eid_t should be used as a type. For example, to access the vertices
// array, a vid_t index is used, while accessing the edges array requires an
// index of type eid_t. A typical iteration over the graph looks like this:
//
// for (vid_t vid = 0; vid < graph->vertex_count; vid++) {
//   for (eid_t eid = graph->vertices[vid];
//        eid < graph->vertices[vid + 1]; eid++) {
//     vid_t nbr_id = graph->edges[eid];
//     // do stuff to the neighbour
//   }
// }
//
// Finally, to enable 64 bit edge ids, the code must be compiled: make EID=64
typedef uint32_t vid_t;
typedef uint32_t eid_device_t;
#ifdef FEATURE_64BIT_EDGE_ID
typedef uint64_t eid_t;
#else
typedef uint32_t eid_t;
#endif

/**
 * A vertex-degree data type used in partitioning algorithms that depend
 * on sorting the vertices by edge degree.
 */
typedef struct vdegree_s {
  vid_t id;      // vertex id
  vid_t degree;  // vertex degree
}vdegree_t;

// Specifies the maximum value an id can hold.
const vid_t VERTEX_ID_MAX = UINT32_MAX;

// Specifies the infinite quantity used by several algorithms (e.g., edge cost).
const uint32_t INFINITE = UINT32_MAX;

// Specifies a type for edge weights. This is useful to allow future changes in
// the precision and value range that edge weights can hold.
typedef uint32_t weight_t;

// Specifies the maximum value a weight can hold.
const weight_t WEIGHT_MAX = UINT32_MAX;

// Specifies the default edge weight
const weight_t DEFAULT_EDGE_WEIGHT =  1;

// Specifies the default vertex value in the vertex list
const weight_t DEFAULT_VERTEX_VALUE = 0;

// Type of memory used to place the GPU graph data structure.
typedef enum {
  GPU_GRAPH_MEM_DEVICE = 0,         // Places the graph on device memory.
  GPU_GRAPH_MEM_MAPPED,             // Places the graph on the host as memory
                                    // mapped space.
  GPU_GRAPH_MEM_MAPPED_VERTICES,    // Only the vertices array on the host.
  GPU_GRAPH_MEM_MAPPED_EDGES,       // Only the edges array on the host.
  GPU_GRAPH_MEM_PARTITIONED_EDGES,  // Partitions the edges array such that part
                                    // of it is placed on device memory and part
                                    // of it mapped on host memory.
  GPU_GRAPH_MEM_MAX
} gpu_graph_mem_t;

// A graph type based on adjacency list representation.
// Modified from [Harish07]:
// A graph G(V,E) is represented as adjacency list, with adjacency lists packed
// into a single large array. Each vertex points to the starting position of its
// own adjacency list in this large array of edges. Vertices of graph G(V,E) are
// represented as a vertices array. Another array of adjacency lists stores the
// edges with edges of vertex i + 1 immediately following the edges of vertex i
// for all i in V. Each entry in the vertices array corresponds to the starting
// index of its adjacency list in the edges array. Each entry of the edges array
// refers to a vertex in vertices array.
//
// IMPORTANT: vertices without edges have the same index in the vertices array
// as the next vertex, hence their number of edges as zero would be calculated
// in the same way as every other vertex.
typedef struct graph_s {
  eid_t*    vertices;      // The vertices list.
  eid_device_t* vertices_d;  // The vertices list for GPU-based partitions that
                             // support compressed vertex list.
  vid_t*    edges;         // The edges list.
  weight_t* weights;       // Stores the weights of the edges.
  weight_t* values;        // Stores the values of the vertices.
  vid_t     vertex_count;  // Number of vertices.
  eid_t     edge_count;    // Number of edges.
  bool      valued;        // Indicates if vertices have values.
  bool      weighted;      // Indicates if edges have weights.
  bool      directed;      // Indicates if the graph is directed.
  bool compressed_vertices;  // Indicates if the graph supports compressed
                             // vertex list or not.
  // The type of memory used to allocate the graph data structure of GPU-based
  // partitions.
  gpu_graph_mem_t gpu_graph_mem;
  // Maintains the host pointer of the vertices array in case it is allocated
  // as a memory mapped buffer for GPU-resident graphs. Keeping this pointer is
  // necessary when freeing the buffer. Note that in this case, vertices will
  // maintain the pointer to the buffer in the device address space.
  eid_t*    mapped_vertices;
  // Maintains the host pointer of the edges array in case it is allocated as a
  // memory mapped buffer for GPU-resident graphs. Keeping this pointer is
  // necessary when freeing the buffer. Note that in this case, "edges" will
  // maintain the pointer to the buffer in the device address space.
  eid_t*    mapped_edges;
  // This member is relevant to GPU-based resident graphs. in case the edge list
  // is partitioned between device memory and mapped memory on the host, this
  // array stores the part of the edge list placed on the host as memory mapped,
  // while "edges" is the pointer to the partition placed on device memory.
  vid_t*    edges_ext;
  // In the case the edges list is partitioned between device memory and mapped
  // memory on the host, this member specifies the boundary after which the
  // vertices should access their edge list via edges_ext.
  vid_t    vertex_ext;
  // In the case the edges list is partitioned between device memory and mapped
  // memory on the host, this member specifies the number of edges placed on
  // the device.
  eid_t    edge_count_ext;
} graph_t;

// Defines a data type for a graph's connected components. components are
// identified by numbers [0 - count). The marker array identifies for each
// vertex the id of the component the vertex is part of.
typedef struct component_set_s {
  graph_t* graph;         // The graph which this component set belongs to.
  vid_t    count;         // Number of components.
  vid_t*   vertex_count;  // Vertex count of each component.
  eid_t*   edge_count;    // Edge count of each component.
  vid_t*   marker;        // The component id for each vertex
  vid_t    biggest;       // The id of the biggest component.
} component_set_t;

/**
 * Allocates space for a graph structure and its buffers, and sets the
 * various members of the structure.
 * @param[in] vertex_count number of vertices
 * @param[in] edge_count number of edges
 * @param[in] weighted indicates if the edge weights are to be loaded
 * @param[in] valued indicates if the vertex values are to be loaded
 * @param[out] graph reference to allocated graph type to store the edge list
 * @return generic success or failure
 */
void graph_allocate(vid_t vertex_count, eid_t edge_count, bool directed,
                    bool weighted, bool valued, graph_t** graph_ret);

/**
 * reads a graph from the given file and builds a graph data type.
 * The function allocates graph data type and the buffers within it.
 * @param[in] graph_file path to the graph file.
 * @param[in] weighted a flag to indicate loading edge weights.
 * @param[out] graph a reference to allocated graph_t type.
 * @return generic success or failure
 */
error_t graph_initialize(const char* graph_file, bool weighted,
                         graph_t** graph);

/**
 * Frees allocated buffers within the "graph" reference initialized
 * via graph_initialize or graph_allocate.
 * @param[in] graph a reference to graph type to be de-allocated
 * @return generic success or failure
 */
error_t graph_finalize(graph_t* graph);

/**
 * Initialize a graph structure (graph_d) to be passed as a parameter to GPU
 * kernels. Both graph_d and graph_h structs reside in host memory. The
 * vertices, edges and weights pointers in graph_d will point to buffers in
 * device memory allocated by the routine. Also, the routine will copy-in the
 * data to the aforementioned three buffers from the corresponding buffers in
 * graph_h.
 * @param[in] graph_h source graph which hosts references to main memory buffers
 * @param[out] graph_d allocated graph that hosts references to device buffers
 * @param[in] gpu_graph_mem an optional parameter that allows to specify the
                            type of memory used to place the data structure
 * @return generic success or failure
 */
error_t graph_initialize_device(const graph_t* graph_h, graph_t** graph_d,
                                gpu_graph_mem_t gpu_graph_mem =
                                GPU_GRAPH_MEM_DEVICE,
                                bool compressed_vertices = false);

/**
 * Free allocated device buffers associated with the graph
 * @param[in] graph_d the graph to be finalized
 */
void graph_finalize_device(graph_t* graph_d);

/**
 * Prints out a graph to standard output in totem format
 * @param[in] graph the graph data structure to print out
 */
void graph_print(graph_t* graph);

/**
 * Stores a graph in binary format in the specified file path
 * @param[in] graph the graph data structure to be stored
 * @param[in] graph_file path to the binary graph file.
 * @return generic success or failure
 */
error_t graph_store_binary(graph_t* graph, const char* filename);

/**
 * Creates a subgraph from a graph. the graph is de-allocated via graph_finalize
 * @param[in] graph the graph to extract the subgraph from
 * @param[in] mask identifies the vertices to be included in the subgraph
 * @param[out] subgraph a reference to allocated subgraph
 * @return generic success or failure
 */
error_t get_subgraph(const graph_t* graph, bool* mask, graph_t** subgraph);

/**
 * Creates a subgraph such that all nodes has at least one incoming or outgoing
 * edge. The subgraph is de-allocated via graph_finalize
 * @param[in] graph the graph to extract the subgraph from
 * @param[out] subgraph a reference to allocated subgraph
 * @return generic success or failure
 */
error_t graph_remove_singletons(const graph_t* graph, graph_t** subgraph);

/**
 * Sorts the neighbours of each vertex by id.
 * @param[in] graph the graph to operate on
 * @param[in] edge_sort_dsc Determine the direction of sorting.
 */
void graph_sort_nbrs(graph_t* graph, bool edge_sort_dsc = false);

/**
 * Sorts the neighbours of each vertex by degree.
 * @param[in] graph the graph to operate on
 * @param[in] edge_sort_dsc Determine the direction of sorting.
 */
void graph_sort_nbrs_by_degree(graph_t* graph, bool edge_sort_dsc = false);

/**
 * Given a given flow graph (ie, a directed graph where for every edge (u,v),
 * there is no edge (v,u)), creates a bidirected graph having reverse edges
 * (v,u) with weight 0 for every edge (u,v) in the original graph. Additionally,
 * for each edge (u,v), it stores the index of the reverse edge (v,u) and vice
 * versa, such that for each edge (u,v) in the original graph:
 *
 *   (v,u) with weight 0 is in the new graph,
 *   reverse_indices[(u,v)] == index of (v,u), and
 *   reverse_indices[(v,u)] == index of (u,v)
 * @param[in] graph the original flow graph
 * @param[out] reverse_indices a reference to array of indices of reverse edges
 * @return bidirected graph
 */
graph_t* graph_create_bidirectional(graph_t* graph, eid_t** reverse_indices);

/**
 * Identifies the weakly connected components in the graph
 * @param[in] graph
 * @param[out] comp_set a component set structure which
 *             identifies the components in the graph
 * @return generic success or failure
 */
error_t get_components_cpu(graph_t* graph, component_set_t** comp_set_ret);

/**
 * De-allocates a component_set_t object
 * @param[in] comp_set a reference to component set type to be de-allocated
 * @return generic success or failure
 */
error_t finalize_component_set(component_set_t* comp_set);

#endif  // TOTEM_GRAPH_H
