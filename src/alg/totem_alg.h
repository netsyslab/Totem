/**
 * Declarations of the algorithms implemented using Totem
 *
 *  Created on: 2013-03-24
 *  Author: Abdullah Gharaibeh
 */
#ifndef TOTEM_ALG_H
#define TOTEM_ALG_H

// totem includes
#include "totem_bitmap.cuh"
#include "totem_comdef.h"
#include "totem_comkernel.cuh"
#include "totem_graph.h"
#include "totem_mem.h"
#include "totem_partition.h"

/**
 * A type for the cost in traversal-based algorithms.
 */
typedef uint16_t cost_t;
const cost_t INF_COST = (cost_t)INFINITE;

/**
 * A type for BFS tree type.
 */
typedef int64_t bfs_tree_t;

/**
 * For traversal-based algorithms, this constant determines the threshold
 * (as percentage of the total number of vertices) below which the frontier
 * is considered sparse. This is used to tune the graph algorithm (for example)
 * to choose between iterating over all the vertices (when the frontier is 
 * not sparse), or build a frontier and iterate over only the vertices in the
 * frontier when it is sparse.
 */
const double TRV_FRONTIER_SPARSE_THRESHOLD = .1;

/*
 * For traversal-based algorithms, this constant determines the threshold
 * (as a fraction of the total number of vertices) that determines the
 * maximum space to be allocated for the frontier array. Since  the GPU has
 * limited memory, this threshold is used by GPU-based partitions to limit
 * the space allocated for the frontier array. Note that if the frontier 
 * in a specific level was longer, then the algorithm should not build a
 * frontier array, and should iterate over all the vertices. This value has
 * been determined experimentally.
 */
const double TRV_MAX_FRONTIER_LEN = .2;

/**
 * A type for page rank. This is useful to allow changes in precision.
 */
typedef float rank_t;

/**
 * Used to define the number of rounds: a static convergance condition
 * for PageRank
 */
const int PAGE_RANK_ROUNDS = 5;

/**
 * A probability used in the PageRank algorithm that models the behavior of the 
 * random surfer when she moves from one page to another without following the 
 * links on the current page.
 * TODO(abdullah): The variable could be passed as a parameter in the entry
 * function to enable more flexibility and experimentation. This however 
 * increases register usage and may affect performance
 */
const rank_t PAGE_RANK_DAMPING_FACTOR = 0.85;

/**
 * Specifies a type for centrality scores. This is useful to allow future
 * changes in the precision and value range that centrality scores can hold.
 */
typedef float score_t;

/**
 * The Centrality algorithms accepts an epsilon value to determine the amount
 * of error that is tolerable, along with how long the algorithm will take to
 * complete. A value of 0.0 will indicate that the algorithm should compute an
 * exact metric. For approximate Betweenness Centrality, we are currently using
 * a value of 1.0, which could change. This value was initially selected as it
 * allows the algorithm to complete in a more reasonable amount of time.
 * The CENTRALITY_SINGLE value configures the centrality algorithm to run a
 * single iteration, which is used for benchmarking purposes.
 */
const double CENTRALITY_EXACT = 0.0;
const double CENTRALITY_APPROXIMATE = 1.0;
const double CENTRALITY_SINGLE = -1.0;

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
 * @return generic success or failure
*/
error_t bfs_cpu(graph_t* graph, vid_t src_id, cost_t* cost);
error_t bfs_bu_cpu(graph_t* graph, vid_t src_id, cost_t* cost);
error_t bfs_queue_cpu(graph_t* graph, vid_t source_id, cost_t* cost);
error_t bfs_gpu(graph_t* graph, vid_t src_id, cost_t* cost);
error_t bfs_bu_gpu(graph_t* graph, vid_t src_id, cost_t* cost);
error_t bfs_vwarp_gpu(graph_t* graph, vid_t src_id, cost_t* cost);
error_t bfs_hybrid(vid_t src_id, cost_t* cost);
error_t bfs_bu_hybrid(vid_t src_id, cost_t* cost);

/**
 * Given an undirected, unweighted graph and a source vertex, compute the 
 * corresponding BFS tree.
 *
 * @param[in]  graph  the graph to perform BFS on
 * @param[in]  src    id of the source vertex
 * @param[out] tree   the BFS tree (the parent of each vertex)
 * @return generic success or failure
*/
error_t graph500_cpu(graph_t* graph, vid_t src, bfs_tree_t* tree);
error_t graph500_hybrid(vid_t src, bfs_tree_t* tree);
void graph500_free(partition_t* par);
void graph500_alloc(partition_t* par);

/**
 * Given a graph, compute the clustering coefficient of each vertex.
 *
 * @param[in] graph the input graph
 * @param[out] coefficients array containing computed coefficinets
 */
 // TODO(treza): change the data type from weight_t to something specific
 // to clustering coefficient (e.g., clustering_coefficient_t) or even
 // change this to a template.
error_t clustering_coefficient_cpu(const graph_t* graph,
                                   weight_t** coefficients);
error_t clustering_coefficient_gpu(const graph_t* graph,
                                   weight_t** coefficients);
error_t clustering_coefficient_sorted_neighbours_cpu(const graph_t* graph,
                                                     weight_t** coefficients);
error_t clustering_coefficient_sorted_neighbours_gpu(const graph_t* graph,
                                                     weight_t** coefficients);

/*
 * Given a weighted graph \f$G = (V, E, w)\f$ and a source vertex \f$v\inV\f$,
 * Dijkstra's algorithm computes the distance from \f$v\f$ to every other
 * vertex in a directed, weighted graph, where the edges have non-negative
 * weights (i.e., \f$\forall (u,v) \in E, w(u,v) \leq 0\f$).
 *
 * @param[in] graph an instance of the graph structure
 * @param[in] source_id vertex id for the source
 * @param[out] shortest_distances the length of the computed shortest paths
 * @return generic success or failure
 */
error_t dijkstra_cpu(const graph_t* graph, vid_t src_id, weight_t* distance);
error_t dijkstra_gpu(const graph_t* graph, vid_t src_id, weight_t* distance);
error_t dijkstra_vwarp_gpu(const graph_t* graph, vid_t src_id,
                           weight_t* distance);
error_t sssp_hybrid(vid_t src_id, weight_t* distance);

/**
 * Given a weighted graph \f$G = (V, E, w)\f$, the All Pairs Shortest Path
 * algorithm computes the distance from every vertex to every other vertex
 * in a weighted graph with no negative cycles.
 * The distances array must be of size vertex_count^2. It mimics a static 
 * array to avoid the overhead of creating an array of pointers. Thus, 
 * accessing index [i][j] will be done as distances[(i * vertex_count) + j]
 *
 * @param[in] graph an instance of the graph structure
 * @param[out] distances the length of the computed shortest paths for each
 *                      vertex
 * @return generic success or failure
 */
error_t apsp_cpu(graph_t* graph, weight_t** distances);
error_t apsp_gpu(graph_t* graph, weight_t** distances);

/**
 * Implements a version of the Label Propagation algorithm described in
 * [Xie 2013] for CPU. Algorithm details are described in 
 * totem_label_propagation.cu.
 * @param[in] graph an instance of the graph structure
 * @param[out] labels the computed labels of each vertex
 * @return generic success or failure
 * 
 */
 // TODO(tanuj): Declare a data type label_t.
error_t label_propagation_cpu(const graph_t* graph, vid_t* labels);

/**
 * Implements a version of the simple PageRank algorithm described in
 * [Malewicz 2010] for both CPU and CPU. Algorithm details are described in
 * totem_page_rank.cu. Note that the "incoming" postfixed funtions take into
 * consideration the incoming edges, while the first two consider the outgoing
 * edges.
 * @param[in]  graph the graph to run PageRank on
 * @param[in]  rank_i the initial rank for each node in the graph (NULL
 *                    indicates uniform initial rankings as default)
 * @param[out] rank the PageRank output array
 * @return generic success or failure
 */
error_t page_rank_cpu(graph_t* graph, rank_t* rank_i, rank_t* rank);
error_t page_rank_gpu(graph_t* graph, rank_t* rank_i, rank_t* rank);
error_t page_rank_vwarp_gpu(graph_t* graph, rank_t* rank_i, rank_t* rank);
error_t page_rank_incoming_cpu(graph_t* graph, rank_t* rank_i, rank_t* rank);
error_t page_rank_incoming_gpu(graph_t* graph, rank_t* rank_i, rank_t* rank);
error_t page_rank_hybrid(rank_t* rank_i, rank_t* rank);
error_t page_rank_incoming_hybrid(rank_t* rank_i, rank_t* rank);


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
 * @return generic success or failure.
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
 * @return generic success or failure.
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
 * @return generic success or failure.
*/
error_t node_degree_cpu(const graph_t* graph, uint32_t** node_degree);
error_t node_degree_gpu(const graph_t* graph, uint32_t** node_degree);

/**
 * Calculate betweenness centrality scores for unweighted graphs using the
 * successors stack implementation.
 * @param[in] graph the graph to use
 * @param[out] centrality_score the output list of betweenness centrality scores
 *             per vertex
 * @return generic success or failure
 */
error_t betweenness_unweighted_cpu(const graph_t* graph,
                                   score_t* centrality_score);
error_t betweenness_unweighted_gpu(const graph_t* graph,
                                   score_t* centrality_score);

/**
 * Calculate betweenness centrality scores for unweighted graphs using the
 * predecessors map implementation.
 * @param[in] graph the graph to use
 * @param[out] centrality_score the output list of betweenness centrality scores
 *             per vertex
 * @return generic success or failure
 */
error_t betweenness_unweighted_shi_gpu(const graph_t* graph,
                                       score_t* centrality_score);

/**
 * Calculate betweenness centrality scores for graphs using the algorithm
 * described in Chapter 2 of GPU Computing Gems (Algorithm 1)
 * @param[in] graph the graph for which the centrality measure is calculated
 *            this is not used for the hybrid version of the algorithm
 * @param[in] epsilon determines how precise the results of the algorithm will
 *            be, and thus also how long it will take to compute
 * @param[out] centrality_score the output list of betweenness centrality
 *             scores per vertex
 * @return generic success or failure
 */
error_t betweenness_cpu(const graph_t* graph, double epsilon, 
                        score_t* centrality_score);
error_t betweenness_gpu(const graph_t* graph, double epsilon, 
                        score_t* centrality_score);
error_t betweenness_hybrid(double epsilon, score_t* centrality_score);

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


typedef struct frontier_state_s {
  bitmap_t current;         // current frontier bitmap
  bitmap_t visited_last;    // a bitmap of the visited vertices before the 
                            // start of the previous round. This is used to
                            // compute the frontier bitmap of the current
                            // round by diffing this bitmap with the visited
                            // bitmap (a bitmap of the visited untill after the
                            // end of the previous round
  vid_t len;                // frontier bitmaps length
  vid_t* list;              // maintains the list of vertices that belong to the
                            // current frontier being processed (GPU only)
  vid_t  list_len;          // maximum number of vertices that the frontier 
                            // list can hold (GPU only)
  vid_t* count;             // used to calculate the current number of vertices
                            // in the frontier (GPU only)
  vid_t* boundaries;        // thread scheduling boundaries (GPU only)
} frontier_state_t;

#ifdef FEATURE_SM35
PRIVATE const int FRONTIER_BOUNDARY_COUNT = 6;
#endif /* FEATURE_SM35 */

/**
 * Initializes a frontier data structure internal state
 * @param[in] frontier reference to the frontier data structure
 */

void frontier_init_gpu(frontier_state_t* state, vid_t vertex_count);
void frontier_init_cpu(frontier_state_t* state, vid_t vertex_count);

/**
 * Frees space allocated for a frontier data structure
 * @param[in] frontier reference to the frontier data structure
 */
void frontier_finalize_gpu(frontier_state_t* state);
void frontier_finalize_cpu(frontier_state_t* state);

/**
 * Resets the state of the frontier
 * @param[in] frontier reference to the frontier data structure
 */
void frontier_reset_gpu(frontier_state_t* state);
void frontier_reset_cpu(frontier_state_t* state);

/**
 * Updates the frontier bitmap 
 * @param[in] frontier reference to the frontier data structure
 * @param[in] visited a bitmap representing  all the vertices that has been
 * visited untill now
 */
vid_t frontier_update_bitmap_cpu(frontier_state_t* state, 
                                 const bitmap_t visited);
vid_t frontier_update_bitmap_gpu(frontier_state_t* state, 
                                 const bitmap_t visited,
                                 cudaStream_t stream);

/**
 * Updates the frontier list with the vertex ids of the vertices in the 
 * frontier. It also defines the scheduling boundaries in the case
 * the vertices are sorted by degree
 * @param[in] frontier reference to the frontier data structure
 */
void frontier_update_list_gpu(frontier_state_t* state,
                              vid_t level, const cost_t* cost, 
                              const cudaStream_t stream);
void frontier_update_list_gpu(frontier_state_t* state, 
                              const cudaStream_t stream);

#ifdef FEATURE_SM35
/**
 * Updates the scheduling boundaries if the vertices are sorted by degree
 * @param[in] frontier reference to the frontier data structure
 * @param[in] graph a reference to the graph to be processed
 */
void frontier_update_boundaries_gpu(frontier_state_t* state, 
                                    const graph_t* graph,
                                    const cudaStream_t stream);
#endif /* FEATURE_SM35 */

/**
 * Returns the number of vertices in the frontier
 * @param[in] frontier the frontier 
 * @return number of vertices in the frontier
 */
inline vid_t frontier_count_cpu(frontier_state_t* state) {
  return bitmap_count_cpu(state->current, state->len);
}
inline vid_t frontier_count_gpu(frontier_state_t* state, cudaStream_t stream) {
  return bitmap_count_gpu(state->current, state->len, state->count, stream);
}

#endif  // TOTEM_ALG_H
