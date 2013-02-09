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
 * Specifies an id type. 
 * We have two id types (vid_t and eid_t). The rule to use them is as follows: 
 * anything that is constrained by the number of vertices should be defined 
 * using the vid_t type, similarly anything that is constrained by the number
 * of edges eid_t should be used as a type. For example, to access the vertices
 * array, a vid_t index is used, while accessing the edges array requires an 
 * index of type eid_t. A typical iteration over the graph looks like this:
 * 
 * for (vid_t vid = 0; vid < graph->vertex_count; vid++) {
 *   for (eid_t eid = graph->vertices[vid]; 
 *        eid < graph->vertices[vid + 1]; eid++) {
 *     vid_t nbr_id = graph->edges[eid];
 *     // do stuff to the neighbour
 *   }
 * }
 *
 * Finally, to enable 64 bit edge ids, the code must be compiled: make EID=64
 */
typedef uint32_t vid_t;
#ifdef FEATURE_64BIT_EDGE_ID
typedef uint64_t eid_t;
#else
typedef uint32_t eid_t;
#endif

/**
 * Specifies the maximum value an id can hold.
 */
const vid_t VERTEX_ID_MAX = UINT32_MAX;

/**
 * Specifies the infinite quantity used by several algorithms (e.g., edge cost).
 */
const vid_t INFINITE = UINT32_MAX;

/**
 * Specifies a type for edge weights. This is useful to allow future changes in
 * the precision and value range that edge weights can hold.
 */
typedef float weight_t;

/**
 * Specifies the maximum value a weight can hold.
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
  eid_t*    vertices;        /**< the vertices list. */
  vid_t*    edges;           /**< the edges list. */
  weight_t* weights;         /**< stores the weights of the edges. */
  weight_t* values;          /**< stores the values of the vertices. */
  vid_t     vertex_count;    /**< number of vertices. */
  eid_t     edge_count;      /**< number of edges. */
  bool      valued;          /**< indicates if vertices have values. */
  bool      weighted;        /**< indicates if edges have weights. */
  bool      directed;        /**< indicates if the graph is directed. */
} graph_t;

/**
 * Defines a data type for a graph's connected components. components are
 * identified by numbers [0 - count). The marker array identifies for each
 * vertex the id of the component the vertex is part of.
 */
typedef struct component_set_s {
  graph_t* graph;        /**< the graph which this component set belongs to */
  vid_t    count;        /**< number of components */
  vid_t*   vertex_count; /**< vertex count of each component (length: count) */
  eid_t*   edge_count;   /**< edge count of each component (length: count) */
  vid_t*   marker;       /**< the component id for each vertex */
                         /**< (length: graph->vertex_count) */
  vid_t    biggest;      /**< the id of the biggest component */
} component_set_t;

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
 * Given an undirected, unweighted graph and a source vertex, find the minimum
 * number of edges needed to reach every vertex V from the source vertex.
 * Its implementation follows Breadth First Search variation based on
 * in [Harish07] using the CPU and GPU, respectively.
 *
 * @param[in]  graph  the graph to perform BFS on
 * @param[in]  src_id id of the source vertex
 * @param[out] cost   the distance (number of of hops) of each vertex from the
 *                    source
 * @return array where the indexes are the vertices' ids and the values are the
 * number of edges needed to reach the vertex. Note that the function gives the
 * ownership of the array and, thus, the client is responsible for freeing the
 * memory area.
*/
error_t bfs_cpu(graph_t* graph, vid_t src_id, uint32_t* cost);
error_t bfs_gpu(graph_t* graph, vid_t src_id, uint32_t* cost);
error_t bfs_vwarp_gpu(graph_t* graph, vid_t src_id, uint32_t* cost);
error_t bfs_hybrid(vid_t src_id, uint32_t* cost);

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
error_t dijkstra_cpu(const graph_t* graph, vid_t src_id, weight_t* distance);
error_t dijkstra_gpu(const graph_t* graph, vid_t src_id, weight_t* distance);
error_t dijkstra_vwarp_gpu(const graph_t* graph, vid_t src_id,
                           weight_t* distance);

/**
 * Given a weighted graph \f$G = (V, E, w)\f$, the All Pairs Shortest Path
 * algorithm computes the distance from every vertex to every other vertex
 * in a weighted graph with no negative cycles.
 *
 * @param[in] graph an instance of the graph structure
 * @param[out] path_ret the length of the computed shortest paths for each
 *                      vertex
 * @return generic success or failure
 */
error_t apsp_cpu(graph_t* graph, weight_t** path_ret);
error_t apsp_gpu(graph_t* graph, weight_t** path_ret);

/**
 * Implements a version of the simple PageRank algorithm described in
 * [Malewicz 2010] for both CPU and CPU. Algorithm details are described in
 * totem_page_rank.cu. Note that the "incoming" postfixed funtions take into
 * consideration the incoming edges, while the first two consider the outgoing
 * edges.
 * @param[in]  graph the graph to run PageRank on
 * @param[in]  rank_i the initial rank for each node in the graph (NULL
 *                    indicates uniform initial rankings as default)
 * @param[out] rank the PageRank output array (must be freed via mem_free)
 * @return generic success or failure
 */
error_t page_rank_cpu(graph_t* graph, float* rank_i, float* rank);
error_t page_rank_gpu(graph_t* graph, float* rank_i, float* rank);
error_t page_rank_vwarp_gpu(graph_t* graph, float* rank_i, float* rank);
error_t page_rank_incoming_cpu(graph_t* graph, float* rank_i, float* rank);
error_t page_rank_incoming_gpu(graph_t* graph, float* rank_i, float* rank);
error_t page_rank_hybrid(float* rank_i, float* rank);
error_t page_rank_incoming_hybrid(float* rank_i, float* rank);


/**
 * Implements the push-relabel algorithm for determining the Maximum flow
 * through a directed graph for both the CPU and the GPU, as described in
 * Hong08]. Note that the source graph must be a flow network, that is, for
 * every edge (u,v) in the graph, there must not exist an edge (v,u).
 *
 * @param[in]  graph the graph on which to run Max Flow
 * @param[in]  source_id the id of the source vertex
 * @param[in]  sink_id the id of the sink vertex
 * @param[out] flow_ret the maximum flow through the network
 * @return generic success or failure
 */
error_t maxflow_cpu(graph_t* graph, vid_t source_id, vid_t sink_id,
                    weight_t* flow_ret);
error_t maxflow_gpu(graph_t* graph, vid_t source_id, vid_t sink_id,
                    weight_t* flow_ret);
error_t maxflow_vwarp_gpu(graph_t* graph, vid_t source_id, vid_t sink_id,
                          weight_t* flow_ret);

/**
 * Given a weighted and undirected graph, the algorithm identifies for each
 * vertex the largest p-core it is part of. A p-core is the maximal subset of
 * vertices such that the sum of edge weights each vertex has is at least "p".
 * The word maximal means that there is no other vertex in the graph that can
 * be added to the subset while preserving the aforementioned property.
 * Note that p-core is a variation of the k-core concept: k-core considers
 * degree, while p-core considers edge weights. If all edges have weight 1, then
 * p-core becomes k-core.
 * Specifically, the algorithm computes the p-core for a range of "p" values
 * between "start" and the maximum p the graph has. In each round, "p" is
 * incremented by "step". The output array "round" stores the latest round
 * (equivalent to the highest p-core) a vertex was part of.
 *
 * @param[in] graph an instance of the graph structure
 * @param[in] start the start value of p
 * @param[in] step the value used to increment p in each new round
 * @param[out] round for each vertex  latest round a vertex was part of
 * @return a flag indicating whether the operation succeeded or not.
 */
error_t pcore_cpu(const graph_t* graph, uint32_t start, uint32_t step,
                  uint32_t** round);

error_t pcore_gpu(const graph_t* graph, uint32_t start, uint32_t step,
                  uint32_t** round);


/**
 * Given an [un]directed, unweighted graph, a source vertex, and a destination
 * vertex. Check if the destination is reachable from the source using the CPU
 * and GPU, respectively.
 * @param[in] source_id id of the source vertex
 * @param[in] destination_id id of the destination vertex
 * @param[in] graph the graph to perform BFS on
 * @param[out] connected true if destination is reachable from source;
 *             otherwise, false.
 * @return a flag indicating whether the operation succeeded or not.
*/
error_t stcon_cpu(const graph_t* graph, vid_t source_id, vid_t destination_id,
                  bool* connected);

error_t stcon_gpu(const graph_t* graph, vid_t source_id, vid_t destination_id,
                  bool* connected);

/**
 * Given a graph, count the number of edges leaving each node.
 * @param[in] graph the graph to use
 * @param[out] node_degree pointer to output list of node degrees, indexed by
 *             vertex id
 * @return a flag indicating whether the operation succeeded or not.
*/
error_t node_degree_cpu(const graph_t* graph, uint32_t** node_degree);
error_t node_degree_gpu(const graph_t* graph, uint32_t** node_degree);

/**
 * Identifies the weakly connected components in the graph
 * @param[in] graph
 * @param[out] comp_set a component set structure which
 *             identifies the components in the graph
 * @return generic success or failure
 */
error_t get_components_cpu(graph_t* graph, component_set_t** comp_set_ret);

/**
 * Calculate betweenness centrality scores for unweighted graphs using the
 * successors stack implementation.
 * @param[in] graph the graph to use
 * @param[out] centrality_score the output list of betweenness centrality scores
 *             per vertex
 * @return generic success or failure
 */
error_t betweenness_unweighted_cpu(const graph_t* graph,
                                   weight_t** centrality_score);
error_t betweenness_unweighted_gpu(const graph_t* graph,
                                   weight_t** centrality_score);

/**
 * Calculate betweenness centrality scores for unweighted graphs using the
 * predecessors map implementation.
 * @param[in] graph the graph to use
 * @param[out] centrality_score the output list of betweenness centrality scores
 *             per vertex
 * @return generic success or failure
 */
error_t betweenness_unweighted_shi_gpu(const graph_t* graph,
                                       weight_t** centrality_score);

/**
 * Calculate betweenness centrality scores for graphs using the algorithm
 * described in Chapter 2 of GPU Computing Gems (Algorithm 1)
 * @param[in] graph the graph for which the centrality measure is calculated
 * @param[out] centrality_score the output list of betweenness centrality
 *             scores per vertex
 * @return generic success or failure
 */
error_t betweenness_cpu(const graph_t* graph, weight_t** centrality_score);

/**
 * Implements the parallel Brandes closeness centrality algorithm using
 * predecessor maps as described in "Fast Network Centrality Analysis Using
 * GPUs" [Shi11]
 */
error_t closeness_unweighted_cpu(const graph_t* graph,
                                 weight_t** centrality_score);
error_t closeness_unweighted_gpu(const graph_t* graph,
                                 weight_t** centrality_score);

/**
 * Calculate stress centrality scores for unweighted graphs.
 * @param[in] graph the graph
 * @param[out] centrality_score the output list of stress centrality scores  for
 *                              each vertex
 * @return generic success or failure
 */
error_t stress_unweighted_cpu(const graph_t* graph,
                              weight_t** centrality_score);
error_t stress_unweighted_gpu(const graph_t* graph,
                              weight_t** centrality_score);

/**
 * De-allocates a component_set_t object
 * @param[in] comp_set a reference to component set type to be de-allocated
 * @return generic success or failure
 */
error_t finalize_component_set(component_set_t* comp_set);

#endif  // TOTEM_GRAPH_H
