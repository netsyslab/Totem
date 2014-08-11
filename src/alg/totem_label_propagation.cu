/**
 *
 * Implements Label Propagation algorithm for CPU. It follows a version 
 * of the  algorithm described in [Xie 2013]. Jierui Xie; Szymanski, B.K., 
 * "LabelRank: A stabilized label propagation algorithm for community detection
 * in networks," Network Science Workshop (NSW), 2013 IEEE 2nd , vol., no., 
 * pp.138,143, April 29 2013-May 1 2013.
 * doi: 10.1109/NSW.2013.6609210 
 *
 * Created on: 2014-08-08
 * Author: Tanuj Kr Aasawat 
 */

// totem includes
#include "totem_alg.h"

const int LABEL_PROPAGATION_MAX_ITERATIONS = 25;
const int LABEL_PROPAGATION_MAX_LABEL_NOT_CHANGED_COUNT = 5;

// Checks for input parameters and special cases. This is invoked at the
// beginning of public interfaces (CPU and GPU).

PRIVATE error_t check_special_cases(const graph_t* graph, bool* finished,
                            vid_t* labels) {
  *finished =  true;
  // Check whether the graph is null or vertex set is empty
  if ((graph == NULL) || (graph->vertex_count == 0) ||
      (labels == NULL)) {
    return FAILURE;
  }

  // Check if this is a single vertex graph
  if (graph->vertex_count == 1) {
    labels[0] = 0;
    return SUCCESS;
  }

  // Check whether the edge set is empty
  if (graph->edge_count == 0) {
    for (vid_t vertex_id = 0; vertex_id < graph->vertex_count; vertex_id++) {
      labels[vertex_id] = vertex_id;
    }
    return SUCCESS;
  }

  *finished = false;
  return SUCCESS;
}

/**
 * Implements a version of the Label Propagation algorithm described in
 * [Xie 2013] for CPU.
 *
 * This algorithm simulates the propagation of labels in the network. Here, we
 * use vertex IDs as labels. Therefore, the number of unique labels is equal
 * to the number of vertices. Initially, each vertex in the graph is assigned
 * a unique numeric label (its own vertex ID). The algorithm consists of
 * three main steps. In the initialization step, based on its neighbours, each
 * vertex computes the probability of receiving each label. An n x m matrix P, 
 * where n is the number of vertices in the graph and m is the number of 
 * labels, holds current probability of vertices observing different labels.
 * In the propagation step, each vertex broadcasts the probability distribution
 * of all the labels to its neighbours. Based on received distribution, each
 * vertex computes the new distribution of each label (sum of received 
 * distributions normalized by its vertex degree). This operation increases 
 * probability of labels that were assigned high probability during propagation
 * at the cost of labels that in propagation received low probabilities. At any
 * given time, the label with the highest probability becomes the label of a 
 * vertex. The propagation phase does not always converge to a state in which
 * all vertices have the same label in successive iteration. To ensure that the
 * propagation phase terminates, we verify if the labels are being updated in
 * successive iterations. For all vertices in the graph, if their labels do not
 * change for a predefined successive number of iterations, the algorithm stops
 * propagating labels and terminates. Please note that this algorithm can only
 * detect disjoint communities.
 * 
 * @param[in] graph an instance of the graph structure
 * @param[out] labels the computed labels of each vertex
 * @return generic success or failure
 *
 */
PRIVATE void label_probability_initialisation(vid_t knumLabels, const graph_t*
                  graph, weight_t** ProbMatrix, vid_t*
                  max_prb_label_not_changed_count) {
  // TODO(tanuj): Declare data types label_t and probability_t.
  OMP(omp parallel for schedule(runtime))
  for (vid_t v = 0; v < graph->vertex_count; v++) {
    max_prb_label_not_changed_count[v] = 0;
    vid_t num_of_neighbours = graph->vertices[v+1] - graph->vertices[v];
    for (vid_t l = 0; l < knumLabels; l++) {  // iterate over each label
      ProbMatrix[v][l] = (v == l) ? 1.0 : 0.0;
      for (eid_t e = graph->vertices[v]; e < graph->vertices[v+1]; e++) {
        vid_t nbr = graph->edges[e];
        ProbMatrix[v][nbr] = 1.0 / (weight_t)num_of_neighbours;
      }  // for
    }  // for
  }  // for
}

PRIVATE void label_propagation(const graph_t* graph, weight_t** ProbMatrix,
                               weight_t** ProbMatrix_new, vid_t*
                               max_prb_label_not_changed_count, vid_t
                               knumLabels) {
  OMP(omp parallel for schedule(runtime))
  for (vid_t v = 0; v < graph->vertex_count; v++) {
    vid_t num_of_neighbours = graph->vertices[v+1] - graph->vertices[v];
    for (vid_t l = 0; l < knumLabels; l++) {  // iterate over each label
      weight_t prb_l = 0.0;
      for (vid_t e = graph->vertices[v]; e < graph->vertices[v+1]; e++) {
        vid_t nbr = graph->edges[e];
        prb_l+= ProbMatrix[nbr][l];  //  ProbMatrix[nbr][l] is the probability
                                     //  from the previous round
      }  //  for
      //  normalize sum of probabilities of a label by number of neighbours
      ProbMatrix_new[v][l] = prb_l / (weight_t)num_of_neighbours;
    }  //  for
  }  //  for
}

PRIVATE void update_labels(const graph_t* graph, vid_t* labels, weight_t**
                          ProbMatrix, weight_t** ProbMatrix_new, vid_t*
                          max_prb_label_not_changed_count, vid_t knumLabels) {
  OMP(omp parallel for schedule(runtime))
  for (vid_t v = 0; v < graph->vertex_count; v++) {
    weight_t max_prb = 0.0;
    int max_label = 0;
    for (vid_t l = 0; l < knumLabels; l++) {  // iterate over each label
      ProbMatrix[v][l] = ProbMatrix_new[v][l];
      if (ProbMatrix[v][l] > max_prb) {
        max_prb = ProbMatrix[v][l];
        max_label = l;
      }
    }  // for
    vid_t old_label = labels[v];
    vid_t new_label = max_label;
    if (old_label == new_label) {
    // label has not changed since last iteration so increase the counter
      max_prb_label_not_changed_count[v]++;
    } else {
    // label has changed; so, reset the counter
      max_prb_label_not_changed_count[v] = 0;
    }
    labels[v] = max_label;
  }  // for
}

PRIVATE bool convergence_termination_checking(const graph_t* graph, vid_t*
                          max_prb_label_not_changed_count, bool finished) {
  OMP(omp parallel for schedule(runtime) reduction(& : finished))
  for (vid_t v = 0; v < graph->vertex_count; v++) {
    if (max_prb_label_not_changed_count[v] <
      LABEL_PROPAGATION_MAX_LABEL_NOT_CHANGED_COUNT) {
      finished = false;
    }  // for
  }
  return finished;
}

error_t label_propagation_cpu(const graph_t* graph, vid_t* labels) {
  // Check for special cases
  bool finished = false;
  error_t rc = check_special_cases(graph, &finished, labels);
  if (finished) return rc;

  // Initialize the labels
  OMP(omp parallel for schedule(runtime))
  for (vid_t vertex_id = 0; vertex_id < graph->vertex_count; vertex_id++) {
    labels[vertex_id] = vertex_id;
  }

  const vid_t knumLabels = graph->vertex_count;

  // P is a n x m matrix, where n is the number of vertices in the graph and
  // m is the number of labels

  weight_t **ProbMatrix = reinterpret_cast<weight_t**>(malloc(
                                    graph->vertex_count * sizeof(weight_t*)));
  weight_t **ProbMatrix_new = reinterpret_cast<weight_t**>
                             (malloc(graph->vertex_count * sizeof(weight_t*)));

  for (vid_t vertex_id = 0; vertex_id < graph->vertex_count; vertex_id++) {
    ProbMatrix[vertex_id] = reinterpret_cast<weight_t *>
                             (malloc(graph->vertex_count * sizeof(weight_t)));
    ProbMatrix_new[vertex_id] = reinterpret_cast<weight_t *>
                             (malloc(graph->vertex_count * sizeof(weight_t)));
  }

  // ProbMatrix_new contains the updated probabilities
  vid_t *max_prb_label_not_changed_count = reinterpret_cast<vid_t*>
                             (malloc(graph->vertex_count * sizeof(vid_t)));
  // Initialize label probabilities
  label_probability_initialisation(knumLabels, graph, ProbMatrix,
                                            max_prb_label_not_changed_count);

  vid_t iteration_count = 0;
  finished = false;
  while (!finished) {
    finished = true;

    // Label propagation
    label_propagation(graph, ProbMatrix, ProbMatrix_new,
                                max_prb_label_not_changed_count, knumLabels);

    // Update lable propability and label selection
    update_labels(graph, labels, ProbMatrix, ProbMatrix_new,
                                max_prb_label_not_changed_count, knumLabels);

    // Verify convergence and termination criterion
    finished = convergence_termination_checking(graph,
                                 max_prb_label_not_changed_count, &finished);
    iteration_count++;
    if (iteration_count >= LABEL_PROPAGATION_MAX_ITERATIONS) {
      finished = true;
    }
  }  // while
  return SUCCESS;
}
