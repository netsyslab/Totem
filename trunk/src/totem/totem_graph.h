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
/**
 * Specifies an id type. This is useful to allow future changes in the
 * underlying type that hold vertex and edge ids. For instance, to support
 * larger graphs that need vertex ids in the 64-bit space.
 */
typedef uint32_t id_t;

/**
 * Specifies the maximum value an id can hold.
 */
const id_t ID_MAX = UINT32_MAX;

/**
 * Specifies the infinite quantity used by several algorithms (e.g., edge cost).
 */
const id_t INFINITE = UINT32_MAX;

/**
 * Specifies a type for edge weights. This is useful to allow future changes in
 * the precision and value range that edge weights can hold.
 */
typedef float weight_t;

/**
 * Specifies the maximum value a weight can hold .
 */
const weight_t WEIGHT_MAX = FLT_MAX;

/**
 * Specifies the default edge weight
 */
const weight_t DEFAULT_EDGE_WEIGHT =  1;

/**
 * Specifies the default vertex value in the vertex list
 */
const weight_t DEFAULT_VERTEX_VALUE = 0;

/**
 * A graph type based on adjacency list representation.
 * Modified from [Harish07]:
 * A graph G(V,E) is represented as adjacency list, with adjacency lists packed
 * into a single large array. Each vertex points to the starting position of its
 * own adjacency list in this large array of edges. Vertices of graph G(V,E) are
 * represented as a vertices array. Another array of adjacency lists stores the
 * edges with edges of vertex i + 1 immediately following the edges of vertex i
 * for all i in V. Each entry in the vertices array corresponds to the starting
 * index of its adjacency list in the edges array. Each entry of the edges array
 * refers to a vertex in vertices array.
 *
 * IMPORTANT: vertices without edges have the same index in the vertices array
 * as the next vertex, hence their number of edges as zero would be calculated
 * in the same way as every other vertex.
 */
typedef struct graph_s {
  id_t*        vertices;        /**< the vertices list. */
  id_t*        edges;           /**< the edges list. */
  weight_t*    weights;         /**< stores the weights of the edges. */
  weight_t*    values;          /**< stores the values of the vertices. */
  uint64_t     vertex_count;    /**< number of vertices. */
  uint64_t     edge_count;      /**< number of edges. */
  bool         valued;          /**< indicates if vertices have values. */
  bool         weighted;        /**< indicates if edges have weights. */
  bool         directed;        /**< indicates if the graph is directed. */
} graph_t;

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
 * via graph_initialize.
 * @param[in] graph a reference to graph type to be de-allocated
 * @return generic success or failure
 */
error_t graph_finalize(graph_t* graph);

// TODO(lauro): Discuss a common order for the parameters.
// TODO(lauro): Organize algorithms alphabetically.

/**
 * Implements a GPU version of the simple PageRank algorithm described in
 * [Malewicz 2010]. Algorithm details are described in totem_page_rank.cu.
 * @param[in]  graph the graph to run PageRank on
 * @param[out] rank the PageRank output array (must be freed via mem_free)
 * @return generic success or failure
 */
error_t page_rank_gpu(graph_t* graph, float** rank);

/**
 * Implements a CPU version of the simple PageRank algorithm described in
 * [Malewicz 2010]. Algorithm details are described in totem_page_rank.cu.
 * @param[in]  graph the graph to run PageRank on
 * @param[out] rank the PageRank output array (must be freed via mem_free)
 * @return generic success or failure
 */
error_t page_rank_cpu(graph_t* graph, float** rank);

/**
 * Given an undirected, unweighted graph and a source vertex, find the minimum
 * number of edges needed to reach every vertex V from the source vertex.
 * Its implementation follows Breadth First Search variation based on
 * in [Harish07] using GPU.
 * @param[in] source_id id of the source vertex
 * @param[in] graph the graph to perform BFS on
 * @return array where the indexes are the vertices' ids and the values are the
 * number of edges needed to reach the vertex. Note that the function gives the
 * ownership of the array and, thus, the client is responsible for freeing the
 * memory area.
*/
error_t bfs_gpu(id_t source_id, const graph_t* graph, uint32_t** cost);

/**
 * Given an undirected, unweighted graph and a source vertex, find the minimum
 * number of edges needed to reach every vertex V from the source vertex.
 * Its implementation follows Breadth First Search variation based on
 * in [Harish07] using CPU.
 * @param[in] source_id id of the source vertex
 * @param[in] graph the graph to perform BFS on
 * @param[out] cost array where the indexes are the vertices' ids and the values
 * are the number of edges needed to reach the vertex. Note that the function
 * gives the ownership of the array and, thus, the client is responsible for
 * freeing the memory area.
 * @return a flag indicating whether the operation succeeded or not.
*/
error_t bfs_cpu(id_t source_id, const graph_t* graph, uint32_t** cost);

/**
 * Given an [un]directed, unweighted graph, a source vertex, and a destination
 * vertex. Check if the destination is reachable from the source using GPU
 * implementation.
 * @param[in] source_id id of the source vertex
 * @param[in] destination_id id of the destination vertex
 * @param[in] graph the graph to perform BFS on
 * @param[out] connected true if destination is reachable from source;
 *             otherwise, false.
 * @return a flag indicating whether the operation succeeded or not.
*/
error_t stcon_gpu(id_t source_id, id_t destination_id, const graph_t* graph,
                  bool* connected);

/**
 * Given an [un]directed, unweighted graph, a source vertex, and a destination
 * vertex. Check if the destination is reachable from the source using CPU
 * implementation.
 * @param[in] source_id id of the source vertex
 * @param[in] destination_id id of the destination vertex
 * @param[in] graph the graph to perform BFS on
 * @param[out] connected true if destination is reachable from source;
 *             otherwise, false.
 * @return a flag indicating whether the operation succeeded or not.
*/
error_t stcon_cpu(id_t source_id, id_t destination_id, const graph_t* graph,
                  bool* connected);

/**
 * Given a weighted graph \f$G = (V, E, w)\f$ and a source vertex \f$v\inV\f$,
 * Dijkstra's algorithm computes the distance from \f$v\f$ to every other
 * vertex in a directed, weighted graph, where the edges have non-negative
 * weights (i.e., \f$\forall (u,v) \in E, w(u,v) \leq 0\f$).
 *
 * @param[in] graph an instance of the graph structure
 * @param[in] source_id vertex id for the source
 * @param[out] shortest_distances the length of the computed shortest paths
 * @return a flag indicating whether the operation succeeded or not.
 */
error_t dijkstra_gpu(graph_t* graph, id_t source_id,
                     weight_t** shortest_distances);


/**
 * Given a weighted graph \f$G = (V, E, w)\f$ and a source vertex \f$v\inV\f$,
 * Dijkstra's algorithm computes the distance from \f$v\f$ to every other
 * vertex in a directed, weighted graph, where the edges have non-negative
 * weights (i.e., \f$\forall (u,v) \in E, w(u,v) \leq 0\f$).
 *
 * @param[in] graph an instance of the graph structure
 * @param[in] source_id vertex id for the source
 * @param[out] shortest_distances the length of the computed shortest paths
 * @return a flag indicating whether the operation succeeded or not.
 */
error_t dijkstra_cpu(graph_t* graph, id_t source_id,
                     weight_t** shortest_distances);

#endif  // TOTEM_GRAPH_H
