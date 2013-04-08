/**
 *  Header for common centrality functions and algorithms.
 *
 *  Created on: 2012-05-24
 *  Author: Greg Redekop
 */

#ifndef TOTEM_CENTRALITY_H
#define TOTEM_CENTRALITY_H

// totem includes
#include "totem_alg.h"

/**
 * Checks for input parameters and special cases. This is invoked at the
 * beginning of public interfaces (GPU and CPU).
 */
error_t betweenness_check_special_cases(const graph_t* graph, bool* finished, 
                                        score_t* betweenness_score);

/**
 * Determine the number of sample nodes to use based on the total number
 * of nodes in the graph and the value of epsilon provided.
 * Number of Samples Nodes = Log2(Total Number of Nodes) / Epsilon^2
 */
int centrality_get_number_sample_nodes(vid_t vertex_count, double epsilon);

/**
 * Populate the sampling nodes for approximate centrality.
 * It randomly selects nodes within the graph. The nodes are not duplicates and
 * each one have at least one outgoing edge.
 */
vid_t* centrality_select_sampling_nodes(const graph_t* graph, 
                                        int number_samples);

/**
 * Unweighted BFS single source shortest path kernel using a successor stack.
 * Nodes are visited in a BFS order and the shortest paths are held in a list of
 * successors for each node.
 * @param[in] graph the graph
 * @param[in] phase the current 'phase' of the BFS
 * @param[out] sigma the number of shortest paths between the source and the
 *                   indexed vertex
 * @param[out] dist the BFS distance from the source to the given node
 * @param[out] succ the list of shortest path successor nodes for each node
 * @param[out] succ_count the number of SP successor nodes for each node
 * @param[out] stack a stack of vertices to process for each node
 * @param[out] stack_count the size of each node's stack
 * @param[out] finished flag for whether the algorithm has completed or not
 */
__global__
void unweighted_sssp_succs_kernel(graph_t graph, int64_t phase, vid_t* sigma,
                                  int32_t* dist, vid_t* succ,
                                  uint32_t* succ_count, vid_t* stack,
                                  uint32_t* stack_count, bool* finished);
/**
 * Unweighted BFS single source shortest path kernel with predecessor map.
 * Nodes are visited in a BFS order and the shortest paths are held in a list of
 * predecessors for each node.
 * @param[in] graph the graph
 * @param[in] r_edges the source vertex for each edge, indexed by edge id
 * @param[in] dist the current BFS phase distance
 * @param[out] dists the BFS distance from the source to the node
 * @param[out] sigma the number of shortest paths between the source and the
 *                   indexed vertex
 * @param[out] preds the predecessor adjacency map
 * @param[out] finished flag for whether the algorithm has completed or not
 */
__global__
void unweighted_sssp_preds_kernel(graph_t graph, vid_t* r_edges, int32_t dist,
                                  int32_t* dists, vid_t* sigma, bool* preds,
                                  bool* finished);
/**
 * Unweighted centrality back propagation kernel for predecessor map
 * implementation. Calculates dependence (delta) for each vertex in the graph.
 * @param[in] graph the graph
 * @param[in] r_edges the source vertex for each edge, indexed by edge id
 * @param[in] dists the BFS distance from the source to the node
 * @param[in] sigma the number of shortest paths between the source and the node
 * @param[in] preds the predecessor adjacency map
 * @param[in] dist the current BFS phase distance
 * @param[out] delta the dependence of each node
 */
__global__
void unweighted_back_prop_kernel(graph_t graph, vid_t* r_edges, int32_t* dists,
                                 vid_t* sigma, bool* preds, int32_t dist,
                                 score_t* delta);

/**
 * Construct reverse edges so that we can easily find the source vertex for each
 * edge in the graph.
 * @param[in] graph the graph
 * @param[out] r_edges_p the pointer to the list of source vertices, indexed by
 *                       edge id.
 * @returns generic success or failure
 */
error_t centrality_construct_r_edges(const graph_t* graph, vid_t** r_edges_p);

#endif // TOTEM_CENTRALITY_H
