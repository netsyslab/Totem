/**
 * Implements the core logic of the handlers to commands supported by the tool.
 *
 *  Created on: 2014-02-28
 *  Author: Abdullah Gharaibeh
 */

// system includes
#include <sstream>
#include <string>

// totem includes
#include "totem_graph.h"
#include "totem_generator.h"

// The maximum log of the number of vertices.
const int kMaxVertexScale = sizeof(vid_t) * 8;

// The maximum log of the number of edges.
const eid_t kMaxEdgeScale =  sizeof(eid_t) * 8;

PRIVATE error_t create_init(generator_config_t* config, vid_t* vertex_count,
                            eid_t* edge_count, vid_t** src, vid_t** dst) {
  assert(config->scale > 0 && config->edge_factor > 0 && src && dst);

  CHK(config->scale < kMaxVertexScale, err_overflow);
  // A -2 is necessary to avoid overflow and reserve space for the last "fake"
  // vertex when scale is set to the maximum supported value.
  *vertex_count = ((uint64_t)1 << config->scale) - 2;

  CHK(ceil(log2(config->edge_factor)) + config->scale <= kMaxEdgeScale,
      err_overflow);
  *edge_count = ((eid_t)config->edge_factor) * (*vertex_count);

  // The src and dst arrays form the edge list that will be produced by the
  // graph generation algorithm.
  *src = reinterpret_cast<vid_t*>(calloc(*edge_count, sizeof(vid_t)));
  *dst = reinterpret_cast<vid_t*>(calloc(*edge_count, sizeof(vid_t)));
  assert(*src && *dst);

  // Seed the random number generator, which will be used by the graph
  // generation algorithm.
  srand48(GLOBAL_SEED);

  return SUCCESS;

err_overflow:
  fprintf(stderr, "Vertex or edge count overflow scale:%d, edge_factor:%d!\n",
          config->scale, config->edge_factor);
  return FAILURE;
}

PRIVATE void edgelist_to_graph(vid_t* src, vid_t* dst, vid_t vertex_count,
                               vid_t edge_count, bool weighted, bool directed,
                               graph_t** graph_ret) {
  // Third, compute the degree of each vertex.
  eid_t* degree = reinterpret_cast<eid_t*>(calloc(vertex_count, sizeof(eid_t)));
  assert(degree);
  for (eid_t i = 0; i < edge_count; i++) {
    degree[src[i]]++;
  }

  // Fourth, setup the graph's data structure.
  graph_t* graph;
  graph_allocate(vertex_count, edge_count, directed, weighted,
                 false /* No values associated with the vertices */ , &graph);
  graph->vertices[0] = 0;
  for (vid_t i = 1; i <= vertex_count; i++) {
    graph->vertices[i] = graph->vertices[i - 1] + degree[i - 1];
  }

  for (eid_t i = 0; i < edge_count; i++) {
    vid_t u = src[i];
    vid_t v = dst[i];
    eid_t pos = degree[u]--;
    eid_t eid = graph->vertices[u] + pos - 1;
    graph->edges[eid] = v;
    if (weighted) {
      // Assign random weights to the edges. For no particular reason, the
      // weights are chosen from the range [0, vertex_count].
      graph->weights[eid] = drand48() * vertex_count;
    }
  }
  free(degree);
  *graph_ret = graph;
}

PRIVATE void graph_to_edgelist(graph_t* graph, vid_t** src, vid_t** dst) {
  *src = reinterpret_cast<vid_t*>(calloc(graph->edge_count, sizeof(vid_t)));
  *dst = reinterpret_cast<vid_t*>(calloc(graph->edge_count, sizeof(vid_t)));
  assert(*src && *dst);
  vid_t index = 0;
  for (vid_t v = 0; v < graph->vertex_count; v++) {
    for (eid_t e = graph->vertices[v]; e < graph->vertices[v+1]; e++) {
      (*src)[index] = v;
      (*dst)[index] = graph->edges[e];
      index++;
    }
  }
}

// Permutates the vertices so that one can't know the characteristics of the
// vertex from its vertex id.
PRIVATE void permute_edgelist(vid_t* src, vid_t* dst, vid_t vertex_count,
                              vid_t edge_count) {
  vid_t* p = reinterpret_cast<vid_t*>(calloc(vertex_count, sizeof(vid_t)));
  for (vid_t i = 0; i < vertex_count; i++) { p[i] = i; }
  for (vid_t i = 0; i < vertex_count; i++) {
    vid_t j = vertex_count * drand48();
    vid_t temp = p[j];
    p[j] = p[i];
    p[i] = temp;
  }
  for (eid_t i = 0; i < edge_count; i++) {
    src[i] = p[src[i]];
    dst[i] = p[dst[i]];
  }
  free(p);
}

// Checks that the number of edges is correct.
PRIVATE error_t check_edge_and_vertex_count(const graph_t* graph,
                                            std::string* report) {
  printf("Checking edge count... "); fflush(stdout);
  eid_t edge_count = 0;
  OMP(omp parallel for reduction(+ : edge_count))
  for (vid_t vid = 0; vid < graph->vertex_count; vid++) {
    edge_count += (graph->vertices[vid + 1] - graph->vertices[vid]);
  }
  printf("done\n"); fflush(stdout);

  std::ostringstream stringStream;
  if (edge_count != graph->edge_count) {
    stringStream << "Edge count: Mismatch. Found " << edge_count
                 << ", Expected: " << graph->edge_count << "\n";
    report->append(stringStream.str());
    return FAILURE;
  } else {
    stringStream << "Vertex count: " << graph->vertex_count << "\nEdge count: "
                 << graph->edge_count << "\n";
    report->append(stringStream.str());
  }
  return SUCCESS;
}

// Checks that the neighbors ids are within the id space of the graph, and
// report whether the neighbours are sorted or not.
PRIVATE error_t check_neighbours_sorted(const graph_t* graph,
                                        std::string* report) {
  printf("Checking neighbours id space and order... "); fflush(stdout);
  bool sorted_asc = true;
  bool sorted_dsc = true;
  bool failed = false;
  vid_t invalid_nbr_id = 0;
  OMP(omp parallel for schedule(guided) reduction(| : failed)
      reduction(& : sorted_asc) reduction(& : sorted_dsc))
  for (vid_t vid = 0; vid < graph->vertex_count; vid++) {
    for (eid_t i = graph->vertices[vid]; i < graph->vertices[vid + 1]; i++) {
      failed |= (graph->edges[i] >= graph->vertex_count);
      if (failed) { invalid_nbr_id = graph->edges[i]; }
      if (i != (graph->vertices[vid + 1] - 1)) {
        sorted_asc &= graph->edges[i] <= graph->edges[i + 1];
        sorted_dsc &= graph->edges[i] >= graph->edges[i + 1];
      }
    }
  }
  printf("done\n"); fflush(stdout);

  if (failed) {
    std::ostringstream stringStream;
    stringStream << "Invalid neighbour id: " << invalid_nbr_id << ", it should "
                 << "be less than " << graph->vertex_count << "\n";
    report->append(stringStream.str());
    return FAILURE;
  }

  report->append("Neighbours sorted: ");
  if (sorted_asc) { report->append("ASCENDING\n");
  } else if (sorted_dsc) { report->append("DESCENDING\n");
  } else { report->append("FALSE\n"); }
  return SUCCESS;
}

PRIVATE bool check_nbr_exist(const graph_t* graph, vid_t vertex, vid_t nbr) {
  bool exist = false;
  OMP(omp parallel for reduction(| : exist))
  for (eid_t i = graph->vertices[vertex];
       i < graph->vertices[vertex + 1]; i++) {
    exist |= (graph->edges[i] == nbr);
  }
  return exist;
}

// Checks whether the graph is directed or undirected. It reports failure in one
// case: the graph is labelled undirected while it is directed.
PRIVATE error_t check_direction(const graph_t* graph, bool detailed_check,
                                std::string* report) {
  printf("Checking edge direction... "); fflush(stdout);
  if (!detailed_check) {
    printf("skipped\n"); fflush(stdout);
    report->append("Direction: ");
    if (graph->directed) { report->append("Labelled DIRECTED\n");
    } else { report->append("Labelled UNDIRECTED\n"); }
    return SUCCESS;
  }

  // Check if the graph is undirected by checking if each edge exist in both
  // directions.
  bool directed = false;
  OMP(omp parallel for schedule(guided) reduction(| : directed))
  for (vid_t vid = 0; vid < graph->vertex_count; vid++) {
    for (eid_t i = graph->vertices[vid]; i < graph->vertices[vid + 1]; i++) {
      vid_t nbr = graph->edges[i];
      directed |= !check_nbr_exist(graph, nbr, vid);
    }
  }
  printf("done\n"); fflush(stdout);

  if (!graph->directed && directed) {
    report->append("Direction: Invalid. The graph is labelled UNDIRECTED, "
                   "but not every edge is present in both directions\n");
    return FAILURE;
  } else if (graph->directed && !directed) {
    report->append("Direction: The graph is labelled DIRECTED, but each edge "
                   "is present in both directions, so it is practically "
                   "UNDIRECTED\n");
  } else {
    report->append("Direction: ");
    if (directed) { report->append("Labelled and verified DIRECTED\n");
    } else { report->append("Labelled and verified UNDIRECTED\n"); }
  }

  return SUCCESS;
}

PRIVATE error_t check_vertices_sorted(const graph_t* graph,
                                      std::string* report) {
  printf("Checking if vertices are sorted by degree... "); fflush(stdout);
  bool sorted_asc = true;
  bool sorted_dsc = true;
  OMP(omp parallel for reduction(& : sorted_asc) reduction(& : sorted_dsc))
  for (vid_t vid = 0; vid < graph->vertex_count - 1; vid++) {
    vid_t nbr_count = graph->vertices[vid + 1] - graph->vertices[vid];
    vid_t next_nbr_count = graph->vertices[vid + 2] - graph->vertices[vid + 1];
    sorted_asc &= nbr_count <= next_nbr_count;
    sorted_dsc &= nbr_count >= next_nbr_count;
  }
  printf("done\n"); fflush(stdout);

  report->append("Vertices sorted by degree: ");
  if (sorted_asc) { report->append("ASCENDING\n");
  } else if (sorted_dsc) { report->append("DESCENDING\n");
  } else { report->append("FALSE\n"); }
  return SUCCESS;
}

// This function modifies the graph by sorting the neighbours. This is done to
// make it easy to calculate the number of repeated edges.
PRIVATE void count_repeated_edges(graph_t* graph, std::string* report) {
  graph_sort_nbrs(graph);
  eid_t repeated_edges = 0;
  OMP(omp parallel for reduction(+ : repeated_edges))
  for (vid_t vid = 0; vid < graph->vertex_count; vid++) {
    for (eid_t i = graph->vertices[vid]; i < graph->vertices[vid + 1]; i++) {
      if ((i > graph->vertices[vid]) &&
          (graph->edges[i] == graph->edges[i - 1])) {
        repeated_edges++;
      }
    }
  }

  std::ostringstream stringStream;
  stringStream << "Repeated edges: " << repeated_edges << " ("
               << 100 * ((double)repeated_edges/graph->edge_count)
               << "% of edges)\n";
  report->append(stringStream.str());
}

PRIVATE void mark_non_singleton_vertices(const graph_t* graph,
                                           bool** mask_ret) {
  assert(graph && mask_ret);
  if (graph->vertex_count == 0) { return; }
  bool* mask = reinterpret_cast<bool*>(calloc(graph->vertex_count,
                                              sizeof(bool)));
  OMP(omp parallel for schedule(guided))
  for (vid_t vid = 0; vid < graph->vertex_count; vid++) {
    for (eid_t i = graph->vertices[vid]; i < graph->vertices[vid + 1]; i++) {
      mask[graph->edges[i]] = true;
      mask[vid] = true;
    }
  }
  *mask_ret = mask;
}

// Singletons are defined as the vertices with no outgoing and no incoming
// edges. Leafs are defined as the vertices with no outgoing edges, but at least
// one incoming edge.
PRIVATE void count_singletons_and_leafs(const graph_t* graph,
                                        std::string* report) {
  bool* mask = NULL;
  mark_non_singleton_vertices(graph, &mask);

  vid_t singleton_count = 0;
  OMP(omp parallel for reduction(+ : singleton_count))
  for (vid_t v = 0; v < graph->vertex_count; v++) {
    if (!mask[v]) { singleton_count++; }
  }

  // Mark the leaf nodes.
  OMP(omp parallel for schedule(guided))
  for (vid_t vid = 0; vid < graph->vertex_count; vid++) {
    vid_t nbr_count = graph->vertices[vid + 1] - graph->vertices[vid];
    if (nbr_count != 0) { mask[vid] = false; }
  }

  vid_t leaf_count = 0;
  OMP(omp parallel for reduction(+ : leaf_count))
  for (vid_t v = 0; v < graph->vertex_count; v++) {
    if (mask[v]) { leaf_count++; }
  }

  std::ostringstream stringStream;
  stringStream << "Singletons: " << singleton_count << " ("
               << 100 * ((double)singleton_count/graph->vertex_count)
               << "% of vertices)\n";
  stringStream << "Leaf vertices: " << leaf_count << " ("
               << 100 * ((double)leaf_count/graph->vertex_count)
               << "% of vertices)\n";
  report->append(stringStream.str());
  free(mask);
}

// The implementation of the following RMAT generation algorithm is based on the
// one available in the SNAP library.
error_t generator_create_rmat(generator_config_t* config, double a, double b,
                              double c, graph_t** graph_ret) {
  vid_t vertex_count = 0;
  eid_t edge_count = 0;
  vid_t* src = NULL;
  vid_t* dst = NULL;
  if (create_init(config, &vertex_count, &edge_count, &src, &dst) != SUCCESS) {
    return FAILURE;
  }

  printf("Generating an RMAT graph with %llu vertices and %llu edges\n",
         (uint64_t)vertex_count, (uint64_t)edge_count);

  // First, generate the edges.
  double d = 1 - (a + b + c);
  for (eid_t i = 0; i < edge_count; i++) {
    vid_t u = 1;
    vid_t v = 1;
    vid_t step = vertex_count / 2;
    double av = a;
    double bv = b;
    double cv = c;
    double dv = d;
    double p = drand48();
    if (p < av) {
    } else if (p < (av + bv)) {
      v += step;
    } else if (p < (av + bv + cv)) {
      u += step;
    } else {
      v += step;
      u += step;
    }

    for (int j = 1; j < config->scale; j++) {
      step = step / 2;
      double var = 0.1;
      av *= 0.95 + var * drand48();
      bv *= 0.95 + var * drand48();
      cv *= 0.95 + var * drand48();
      dv *= 0.95 + var * drand48();

      double s = av + bv + cv + dv;
      av = av / s;
      bv = bv / s;
      cv = cv / s;
      dv = dv / s;

      // Choose partition.
      p = drand48();
      if (p < av) {
        // Do nothing.
      } else if (p < (av + bv)) {
        v += step;
      } else if (p < (av+bv+cv)) {
        u += step;
      } else {
        v += step;
        u += step;
      }
    }
    src[i] = u - 1;
    dst[i] = v - 1;

    // Avoid self edges.
    if (src[i] == dst[i]) { i = i - 1; continue; }

    // Print out progress so far.
    if (i % (1024 * 1024) == 0) {
      fprintf(stderr, "%dM edges created\n", i / (1024*1024));
      fflush(stderr);
    }
  }

  // Second, permutate the vertices so that one can't know what are the
  // high-degree edges from the vertex id.
  permute_edgelist(src, dst, vertex_count, edge_count);

  // Third, create the totem graph.
  edgelist_to_graph(src, dst, vertex_count, edge_count, config->weighted,
                    true /* Directed graph */, graph_ret);

  free(src);
  free(dst);
  return SUCCESS;
}

error_t generator_create_uniform(generator_config_t* config,
                                 graph_t** graph_ret) {
  vid_t vertex_count = 0;
  eid_t edge_count = 0;
  vid_t* src = NULL;
  vid_t* dst = NULL;
  if (create_init(config, &vertex_count, &edge_count, &src, &dst) != SUCCESS) {
    return FAILURE;
  }

  printf("Generating a uniform graph with %llu vertices and %llu edges\n",
         (uint64_t)vertex_count, (uint64_t)edge_count);

  for (eid_t i = 0; i < edge_count; i++) {
    src[i] = ((vid_t)mrand48()) % vertex_count;
    dst[i] = ((vid_t)mrand48()) % vertex_count;
    if (src[i] == dst[i]) { i = i - 1; continue; }
  }

  edgelist_to_graph(src, dst, vertex_count, edge_count, config->weighted,
                    true /* Directed graph */, graph_ret);

  free(src);
  free(dst);
  return SUCCESS;
}

error_t generator_check_and_summarize(generator_config_t* config,
                                      std::string* report) {
  printf("Loading graph from disk... "); fflush(stdout);
  graph_t* graph = NULL;
  CALL_SAFE(graph_initialize(config->input_graph_file.c_str(), false, &graph));
  printf("done");
  CHK_SUCCESS(check_edge_and_vertex_count(graph, report), error);
  CHK_SUCCESS(check_direction(graph, config->check_direction, report), error);
  CHK_SUCCESS(check_neighbours_sorted(graph, report), error);
  CHK_SUCCESS(check_vertices_sorted(graph, report), error);
  count_singletons_and_leafs(graph, report);
  // IMPORTANT: count_repeated_edges must be the last to call because it
  // modifies the graph.
  count_repeated_edges(graph, report);
  CALL_SAFE(graph_finalize(graph));
  return SUCCESS;

error:
  return FAILURE;
}

error_t generator_degree_distribution(generator_config_t* config,
                                      eid_t** degree_distribution_out,
                                      eid_t* highest_degree_out) {
  graph_t* graph = NULL;
  CALL_SAFE(graph_initialize(config->input_graph_file.c_str(), false, &graph));

  // Get the highest degree.
  eid_t highest_degree = 0;
  for (vid_t v = 0; v < graph->vertex_count; v++) {
    eid_t degree = graph->vertices[v + 1] - graph->vertices[v];
    if (degree > highest_degree) {
      highest_degree = degree;
    }
  }
  highest_degree++;

  eid_t* degree_distribution =
      reinterpret_cast<eid_t*>(calloc(highest_degree, sizeof(eid_t)));
  assert(degree_distribution);
  for (vid_t v = 0; v < graph->vertex_count; v++) {
    eid_t degree = graph->vertices[v + 1] - graph->vertices[v];
    degree_distribution[degree]++;
  }

  *highest_degree_out = highest_degree;
  *degree_distribution_out = degree_distribution;
  CALL_SAFE(graph_finalize(graph));
  return SUCCESS;
}

error_t generator_permute(generator_config_t* config,
                          graph_t** permuted_graph) {
  // TODO(abdullah): Take weights into consideration.
  graph_t* graph;
  CALL_SAFE(graph_initialize(config->input_graph_file.c_str(),
                             false /* Ignore weights */ , &graph));
  vid_t* src = NULL;
  vid_t* dst = NULL;
  graph_to_edgelist(graph, &src, &dst);
  vid_t vertex_count = graph->vertex_count;
  vid_t edge_count = graph->edge_count;
  bool directed = graph->directed;
  graph_finalize(graph);
  permute_edgelist(src, dst, vertex_count, edge_count);
  edgelist_to_graph(src, dst, vertex_count, edge_count,
                    false /* Ignore weights */, directed, permuted_graph);

  free(src);
  free(dst);
  return SUCCESS;
}

error_t generator_reverse(generator_config_t* config,
                          graph_t** reversed_graph) {
  // TODO(abdullah): Take weights into consideration.
  graph_t* graph;
  CALL_SAFE(graph_initialize(config->input_graph_file.c_str(),
                             false /* Ignore weights */ , &graph));

  if (!graph->directed) {
    printf("The graph is labelled as undirected, nothing to do\n");
    return FAILURE;
  }

  vid_t* src = NULL;
  vid_t* dst = NULL;
  graph_to_edgelist(graph, &src, &dst);
  vid_t vertex_count = graph->vertex_count;
  vid_t edge_count = graph->edge_count;
  graph_finalize(graph);

  for (eid_t eid = 0; eid < edge_count; eid++) {
    vid_t tmp = src[eid];
    src[eid] = dst[eid];
    dst[eid] = tmp;
  }

  edgelist_to_graph(src, dst, vertex_count, edge_count,
                    false /* Ignore weights */, true /* Directed graph */,
                    reversed_graph);

  free(src);
  free(dst);
  return SUCCESS;
}

error_t generator_undirected(generator_config_t* config,
                             graph_t** undirected_graph) {
  // TODO(abdullah): Take weights into consideration.
  graph_t* graph;
  CALL_SAFE(graph_initialize(config->input_graph_file.c_str(),
                             false /* Ignore weights */ , &graph));

  if (!graph->directed) {
    printf("The graph is labelled as undirected, nothing to do\n");
    return FAILURE;
  }

  if (log2(graph->vertex_count) + 1 > kMaxVertexScale ||
      log2(graph->edge_count) + 1 > kMaxEdgeScale) {
    fprintf(stderr, "Vertex or edge count overflow. Scale:%d, Edge "
            "factor:%d!\n", config->scale, config->edge_factor);
    return FAILURE;
  }

  vid_t* src = reinterpret_cast<vid_t*>(calloc((graph->edge_count * 2),
                                               sizeof(vid_t)));
  vid_t* dst = reinterpret_cast<vid_t*>(calloc((graph->edge_count * 2),
                                               sizeof(vid_t)));
  assert(src && dst);

  eid_t i = 0;
  for (vid_t v = 0; v < graph->vertex_count; v++) {
    for (eid_t e = graph->vertices[v];
         e < graph->vertices[v+1]; e++) {
      src[i] = v;
      dst[i] = graph->edges[e];
      src[i + 1] = graph->edges[e];
      dst[i + 1] = v;
      i += 2;
    }
  }
  vid_t vertex_count = graph->vertex_count * 2;
  vid_t edge_count = graph->edge_count * 2;
  graph_finalize(graph);

  edgelist_to_graph(src, dst, vertex_count, edge_count,
                    false /* Ignore weights */, false /* Undirected graph */,
                    undirected_graph);

  free(src);
  free(dst);
  return SUCCESS;
}

error_t generator_sort_vertices_by_degree(generator_config_t* config,
                                          graph_t** sorted_graph) {
  // TODO(abdullah): Take weights into consideration.
  graph_t* graph;
  CALL_SAFE(graph_initialize(config->input_graph_file.c_str(),
                             false /* Ignore weights */ , &graph));

  // Prepare the degree-sorted list of vertices
  typedef struct {
    vid_t id;
    vid_t degree;
  } vertex_degree_t;
  vertex_degree_t* degree = reinterpret_cast<vertex_degree_t*>(calloc(
      graph->vertex_count, sizeof(vertex_degree_t)));
  assert(degree);
  OMP(omp parallel for)
  for (vid_t v = 0; v < graph->vertex_count; v++) {
    degree[v].id = v;
    degree[v].degree = graph->vertices[v + 1] - graph->vertices[v];
  }
  // The comparison function is implemented as a lambda function (a new feature
  // in C++ that allows defining anonymous functions).
  tbb::parallel_sort(degree, degree + graph->vertex_count,
                     [] (const vertex_degree_t& d1,
                         const vertex_degree_t& d2) {
                       return (d1.degree < d2.degree);
                     });

  // Create a map of old to new vertex id.
  vid_t* map = reinterpret_cast<vid_t*>(calloc(graph->vertex_count,
                                               sizeof(vid_t)));
  assert(map);
  OMP(omp parallel for)
  for (vid_t v = 0; v < graph->vertex_count; v++) {
    map[degree[v].id] = v;
  }
  free(degree);

  // Prepare an edge list of the graph using the new ids.
  vid_t* src = reinterpret_cast<vid_t*>(calloc(graph->edge_count,
                                               sizeof(vid_t)));
  vid_t* dst = reinterpret_cast<vid_t*>(calloc(graph->edge_count,
                                               sizeof(vid_t)));
  eid_t index = 0;
  for (vid_t v = 0; v < graph->vertex_count; v++) {
    for (eid_t e = graph->vertices[v]; e < graph->vertices[v + 1]; e++) {
      src[index] = map[v];
      dst[index] = map[graph->edges[e]];
      index++;
    }
  }
  free(map);

  vid_t vertex_count = graph->vertex_count;
  vid_t edge_count = graph->edge_count;
  bool directed = graph->directed;
  graph_finalize(graph);

  edgelist_to_graph(src, dst, vertex_count, edge_count,
                    false /* Ignore weights */, directed, sorted_graph);

  free(src);
  free(dst);
  return SUCCESS;
}
