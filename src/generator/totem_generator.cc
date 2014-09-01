/**
 * Implements the core logic of the handlers to commands supported by the tool.
 *
 *  Created on: 2014-02-28
 *  Author: Abdullah Gharaibeh
 */

// system includes
#include <map>
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
  printf("Vertex or edge count overflow scale:%d, edge_factor:%d!\n",
         config->scale, config->edge_factor);
  return FAILURE;
}

PRIVATE void get_output_file_with_extension(generator_config_t* config,
                                            const std::string& ext,
                                            std::string* output_file) {
  if (config->output_directory.empty()) {
    output_file->assign(config->input_graph_file);
  } else {
    const std::string& input_graph_file = config->input_graph_file;
    std::string basename = input_graph_file.rfind("/") != std::string::npos ?
        input_graph_file.substr(input_graph_file.rfind("/") + 1) :
        input_graph_file;
    output_file->assign(config->output_directory);
    output_file->append("/");
    output_file->append(basename);
  }
  output_file->append(ext);
}

PRIVATE void write_graph(graph_t* graph, const std::string& graph_path) {
  printf("Writing graph file %s ", graph_path.c_str()); fflush(stdout);
  CALL_SAFE(graph_store_binary(graph, graph_path.c_str()));
  printf("done.\n"); fflush(stdout);
  CALL_SAFE(graph_finalize(graph));
}

PRIVATE void write_graph_with_extension(
    generator_config_t* config, graph_t* graph, const std::string& ext) {
  std::string output_graph_file;
  get_output_file_with_extension(config, ext, &output_graph_file);
  write_graph(graph, output_graph_file);
}

PRIVATE void load_graph(
    const std::string& graph_path, bool weighted, graph_t** graph) {
  printf("Loading graph file %s ", graph_path.c_str()); fflush(stdout);
  CALL_SAFE(graph_initialize(graph_path.c_str(), weighted, graph));
  printf("done.\n"); fflush(stdout);
}

PRIVATE void edgelist_to_graph(
    vid_t* src, vid_t* dst, vid_t vertex_count, eid_t edge_count, bool weighted,
    bool directed, graph_t** graph_ret) {
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

PRIVATE void graph_to_edgelist(const graph_t* graph, vid_t** src, vid_t** dst) {
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
PRIVATE void permute_edgelist(
    vid_t* src, vid_t* dst, vid_t vertex_count, eid_t edge_count) {
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
PRIVATE error_t check_edge_and_vertex_count(
    const graph_t* graph, std::string* report) {
  printf("Checking edge count... "); fflush(stdout);
  eid_t edge_count = 0;
  OMP(omp parallel for reduction(+ : edge_count))
  for (vid_t vid = 0; vid < graph->vertex_count; vid++) {
    edge_count += (graph->vertices[vid + 1] - graph->vertices[vid]);
  }
  printf("done.\n"); fflush(stdout);

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
PRIVATE error_t check_neighbours_sorted(
    const graph_t* graph, std::string* report) {
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
  printf("done.\n"); fflush(stdout);

  if (failed) {
    std::ostringstream stringStream;
    stringStream << "Invalid neighbour id: " << invalid_nbr_id << ", it should "
                 << "be less than " << graph->vertex_count << ".\n";
    printf("%s", stringStream.str().c_str());
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
PRIVATE error_t check_direction(
    const graph_t* graph, bool detailed_check, std::string* report) {
  printf("Checking edge direction... "); fflush(stdout);
  if (!detailed_check) {
    printf("skipped.\n"); fflush(stdout);
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
  printf("done.\n"); fflush(stdout);

  if (!graph->directed && directed) {
    printf("Invalid direction, the graph is labelled UNDIRECTED, "
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

PRIVATE error_t check_vertices_sorted(
    const graph_t* graph, std::string* report) {
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
  printf("done.\n"); fflush(stdout);

  report->append("Vertices sorted by degree: ");
  if (sorted_asc) { report->append("ASCENDING\n");
  } else if (sorted_dsc) { report->append("DESCENDING\n");
  } else { report->append("FALSE\n"); }
  return SUCCESS;
}

// This function modifies the graph by sorting the neighbours. This is done to
// make it easy to calculate the number of repeated edges.
PRIVATE void count_repeated_edges(graph_t* graph, std::string* report) {
  printf("Counting repeated edges... "); fflush(stdout);
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
  printf("done.\n"); fflush(stdout);

  std::ostringstream stringStream;
  stringStream << "Repeated edges: " << repeated_edges << " ("
               << 100 * (static_cast<double>(repeated_edges)/graph->edge_count)
               << "% of edges)\n";
  report->append(stringStream.str());
}

PRIVATE void mark_non_singleton_vertices(
    const graph_t* graph, bool** mask_ret) {
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
PRIVATE void count_singletons_and_leafs(
    const graph_t* graph, std::string* report) {
  printf("Counting singletons and leaf nodes... "); fflush(stdout);
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
  printf("done.\n"); fflush(stdout);

  std::ostringstream stringStream;
  stringStream << "Singletons: " << singleton_count << " ("
               << 100 * (static_cast<double>(singleton_count) /
                         graph->vertex_count)
               << "% of vertices)\n";
  stringStream << "Leaf vertices: " << leaf_count << " ("
               << 100 * (static_cast<double>(leaf_count) / graph->vertex_count)
               << "% of vertices)\n";
  report->append(stringStream.str());
  free(mask);
}

// The implementation of the following RMAT generation algorithm is based on the
// one available in the SNAP library.
PRIVATE error_t create_rmat_handler(
    generator_config_t* config, graph_t** graph_ret) {
  const double kA = 0.57;
  const double kB = 0.19;
  const double kC = 0.19;
  const double kD = 1 - (kA + kB + kC);

  vid_t vertex_count = 0;
  eid_t edge_count = 0;
  vid_t* src = NULL;
  vid_t* dst = NULL;
  if (create_init(config, &vertex_count, &edge_count, &src, &dst) != SUCCESS) {
    return FAILURE;
  }

  printf("Generating an RMAT graph with %llu vertices and %llu edges.\n",
         (uint64_t)vertex_count, (uint64_t)edge_count);

  // First, generate the edges.
  for (eid_t i = 0; i < edge_count; i++) {
    vid_t u = 1;
    vid_t v = 1;
    vid_t step = vertex_count / 2;
    double av = kA;
    double bv = kB;
    double cv = kC;
    double dv = kD;
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
      printf("%dM edges created\n", i / (1024*1024));
      fflush(stdout);
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

PRIVATE error_t create_uniform_handler(
    generator_config_t* config, graph_t** graph_ret) {
  vid_t vertex_count = 0;
  eid_t edge_count = 0;
  vid_t* src = NULL;
  vid_t* dst = NULL;
  if (create_init(config, &vertex_count, &edge_count, &src, &dst) != SUCCESS) {
    return FAILURE;
  }

  printf("Generating a uniform graph with %llu vertices and %llu edges.\n",
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

// Performs sanity check on the graph and produces summary information regarding
// its characteristics.
PRIVATE void analyze_summary_handler(generator_config_t* config) {
  graph_t* graph = NULL;
  load_graph(config->input_graph_file, false, &graph);
  std::string report = "";
  CHK_SUCCESS(check_edge_and_vertex_count(graph, &report), error);
  CHK_SUCCESS(check_direction(graph, config->check_direction, &report), error);
  CHK_SUCCESS(check_neighbours_sorted(graph, &report), error);
  CHK_SUCCESS(check_vertices_sorted(graph, &report), error);
  count_singletons_and_leafs(graph, &report);
  // IMPORTANT: count_repeated_edges must be the last to call because it
  // modifies the graph.
  count_repeated_edges(graph, &report);
  CALL_SAFE(graph_finalize(graph));
  printf("Passed!\n");
  printf("\nSummary Report:\n===============\n%s", report.c_str());
  return;

error:
  printf("Failed!\n");
}

PRIVATE error_t generator_degree_distribution(generator_config_t* config,
                                              eid_t** degree_distribution_out,
                                              eid_t* highest_degree_out) {
  graph_t* graph = NULL;
  load_graph(config->input_graph_file, false, &graph);

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

PRIVATE void analyze_degree_distribution_handler(generator_config_t* config) {
  eid_t highest_degree = 0;
  eid_t* degree_distribution = NULL;
  generator_degree_distribution(config, &degree_distribution, &highest_degree);
  if (!degree_distribution) { return; }

  std::string degree_file;
  get_output_file_with_extension(config, ".degreeDist", &degree_file);

  printf("Writing file %s ", degree_file.c_str());
  FILE* file_handler = fopen(degree_file.c_str(), "w");
  fprintf(file_handler, "degree\tvertex_count\n");
  for (eid_t degree = 0; degree < highest_degree; degree++) {
    if (degree_distribution[degree]) {
      fprintf(file_handler, "%llu\t%llu\n", (uint64_t)degree,
              (uint64_t)degree_distribution[degree]);
    }
  }
  fclose(file_handler);
  printf("done.\n");
  free(degree_distribution);
}

// Creates a new graph from an existing one after permuting the ids of its
// vertices.
// TODO(abdullah): Take weights into consideration.
PRIVATE error_t alter_permute_handler(
    generator_config_t* config, graph_t* graph, graph_t** permuted_graph) {
  vid_t* src = NULL;
  vid_t* dst = NULL;
  graph_to_edgelist(graph, &src, &dst);
  vid_t vertex_count = graph->vertex_count;
  eid_t edge_count = graph->edge_count;
  bool directed = graph->directed;
  graph_finalize(graph);
  permute_edgelist(src, dst, vertex_count, edge_count);
  edgelist_to_graph(src, dst, vertex_count, edge_count,
                    false /* Ignore weights */, directed, permuted_graph);

  free(src);
  free(dst);
  return SUCCESS;
}

PRIVATE void get_reverse_edgelist(
    const graph_t* graph, vid_t** src, vid_t** dst) {
  graph_to_edgelist(graph, src, dst);
  for (eid_t eid = 0; eid < graph->edge_count; eid++) {
    vid_t tmp = (*src)[eid];
    (*src)[eid] = (*dst)[eid];
    (*dst)[eid] = tmp;
  }
}

// Creates a new graph from an existing one after reversing the direction of
// each edge.
// TODO(abdullah): Take weights into consideration.
PRIVATE error_t alter_reverse_handler(
    generator_config_t* config, graph_t* graph, graph_t** reversed_graph) {
  if (!graph->directed) {
    printf("The graph is labelled as undirected, nothing to do.\n");
    return FAILURE;
  }

  vid_t* src = NULL;
  vid_t* dst = NULL;
  get_reverse_edgelist(graph, &src, &dst);

  vid_t vertex_count = graph->vertex_count;
  eid_t edge_count = graph->edge_count;
  graph_finalize(graph);
  edgelist_to_graph(src, dst, vertex_count, edge_count,
                    false /* Ignore weights */, true /* Directed graph */,
                    reversed_graph);

  free(src);
  free(dst);
  return SUCCESS;
}

// TODO(abdullah): Take weights into consideration.
PRIVATE void get_undirected_edgelist(
    const graph_t* graph, vid_t** src, vid_t** dst) {
  *src = reinterpret_cast<vid_t*>(
      calloc((graph->edge_count * 2), sizeof(vid_t)));
  *dst = reinterpret_cast<vid_t*>(
      calloc((graph->edge_count * 2), sizeof(vid_t)));
  assert(*src && *dst);

  eid_t i = 0;
  for (vid_t v = 0; v < graph->vertex_count; v++) {
    for (eid_t e = graph->vertices[v];
         e < graph->vertices[v+1]; e++) {
      (*src)[i] = v;
      (*dst)[i] = graph->edges[e];
      (*src)[i + 1] = graph->edges[e];
      (*dst)[i + 1] = v;
      i += 2;
    }
  }
}

// Creates a new undirected graph from an existing directed one.
// TODO(abdullah): Take weights into consideration.
PRIVATE error_t alter_undirected_handler(
    generator_config_t* config, graph_t* graph, graph_t** undirected_graph) {
  if (!graph->directed) {
    printf("The graph is labelled as undirected, nothing to do.\n");
    return FAILURE;
  }

  if (log2(graph->edge_count) + 1 > kMaxEdgeScale) {
    printf("Vertex or edge count overflow");
    return FAILURE;
  }

  vid_t* src = NULL;
  vid_t* dst = NULL;
  get_undirected_edgelist(graph, &src, &dst);

  vid_t vertex_count = graph->vertex_count;
  eid_t edge_count = graph->edge_count * 2;
  graph_finalize(graph);
  edgelist_to_graph(src, dst, vertex_count, edge_count,
                    false /* Ignore weights */, false /* Undirected graph */,
                    undirected_graph);

  free(src);
  free(dst);
  return SUCCESS;
}

PRIVATE void get_sorted_vertices_map(const graph_t* graph, vid_t** map) {
  // Prepare the degree-sorted list of vertices
  typedef struct { vid_t id; vid_t degree; } vertex_degree_t;
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
                     [] (const vertex_degree_t& d1, const vertex_degree_t& d2) {
                       return (d1.degree < d2.degree);
                     });

  // Create a map of old to new vertex id.
  *map = reinterpret_cast<vid_t*>(calloc(graph->vertex_count, sizeof(vid_t)));
  assert(*map);
  OMP(omp parallel for)
  for (vid_t v = 0; v < graph->vertex_count; v++) { (*map)[degree[v].id] = v; }
  free(degree);
}

PRIVATE void get_sorted_vertices_edgelist(
    const graph_t* graph, vid_t** src, vid_t** dst) {
  // Create a map of old to new vertex id.
  vid_t* map;
  get_sorted_vertices_map(graph, &map);

  *src = reinterpret_cast<vid_t*>(calloc(graph->edge_count, sizeof(vid_t)));
  *dst = reinterpret_cast<vid_t*>(calloc(graph->edge_count, sizeof(vid_t)));
  eid_t index = 0;
  for (vid_t v = 0; v < graph->vertex_count; v++) {
    for (eid_t e = graph->vertices[v]; e < graph->vertices[v + 1]; e++) {
      (*src)[index] = map[v];
      (*dst)[index] = map[graph->edges[e]];
      index++;
    }
  }
  free(map);
}

// Creates a new graph from an existing one after permuting the vertex ids such
// that they are sorted by degree.
// TODO(abdullah): Take weights into consideration.
PRIVATE error_t alter_sort_vertices_handler(
    generator_config_t* config, graph_t* graph, graph_t** sorted_graph) {

  // Prepare an edge list of the graph using the new ids.
  vid_t* src = NULL;
  vid_t* dst = NULL;
  get_sorted_vertices_edgelist(graph, &src, &dst);

  vid_t vertex_count = graph->vertex_count;
  eid_t edge_count = graph->edge_count;
  bool directed = graph->directed;
  graph_finalize(graph);
  edgelist_to_graph(src, dst, vertex_count, edge_count,
                    false /* Ignore weights */, directed, sorted_graph);

  free(src);
  free(dst);
  return SUCCESS;
}

PRIVATE error_t alter_binary_handler(
    generator_config_t* config, graph_t* graph, graph_t** ret_graph) {
  *ret_graph = graph;
  return SUCCESS;
}

// TODO(abdullah): Take weights into consideration.
PRIVATE error_t alter_remove_singletons_handler(
    generator_config_t* config, graph_t* graph, graph_t** graph_no_singletons) {
  error_t err = graph_remove_singletons(graph, graph_no_singletons);
  graph_finalize(graph);
  return err;
}

PRIVATE error_t alter_sort_neighbours_handler(
    generator_config_t* config, graph_t* graph, graph_t** sorted_graph) {
  // TODO(scott): Add an option to the generator to allow descending order here.
  graph_sort_nbrs(graph, false /* Sort in ascending order. */);
  *sorted_graph = graph;
  return SUCCESS;
}

// TODO(abdullah): Take weights into consideration.
void alter_handler(generator_config_t* config) {
  graph_t* graph;
  load_graph(config->input_graph_file, false /* Ignore weights */ , &graph);

  // Defines the signature of the alert sub-commands handlers.
  typedef error_t(*alter_sub_command_handler_t)
      (generator_config_t*, graph_t*, graph_t**);

  // Maps each alter sub-command with its handler.
  const std::map<std::string, alter_sub_command_handler_t> dispatch_map = {
    {kBinarySubCommand, alter_binary_handler},
    {kPermuteSubCommand, alter_permute_handler},
    {kRemoveSingletonsSubCommand, alter_remove_singletons_handler},
    {kReverseSubCommand, alter_reverse_handler},
    {kSortNeighboursSubCommand, alter_sort_neighbours_handler},
    {kSortVerticesSubCommand, alter_sort_vertices_handler},
    {kUndirectedSubCommand, alter_undirected_handler},
  };

  // Maps each alter sub-command with the extension to be used to store
  // the generated altered graph.
  const std::map<std::string, std::string> extensions_map = {
    {kBinarySubCommand, ".tbin"},
    {kPermuteSubCommand, ".permuted"},
    {kRemoveSingletonsSubCommand, ".noSingletons"},
    {kReverseSubCommand, ".reversed"},
    {kSortNeighboursSubCommand, ".sortedNbrs"},
    {kSortVerticesSubCommand, ".sortedVertices"},
    {kUndirectedSubCommand, ".undirected"},
  };

  assert(dispatch_map.find(config->sub_command) != dispatch_map.end());
  const alter_sub_command_handler_t handler =
      dispatch_map.find(config->sub_command)->second;

  printf("Invoking %s sub command handler.\n", config->sub_command.c_str());
  graph_t* altered_graph = NULL;
  if (handler(config, graph, &altered_graph) == SUCCESS) {
    const std::string& ext = extensions_map.find(config->sub_command)->second;
    write_graph_with_extension(config, altered_graph, ext);
  }
}

void create_handler(generator_config_t* config) {
  // Defines the signature of the create sub-commands handlers.
  typedef error_t(*create_sub_command_handler_t)
      (generator_config_t*, graph_t**);

  // Maps each alter sub-command with its handler.
  const std::map<std::string, create_sub_command_handler_t> dispatch_map = {
      {kRmatSubCommand, create_rmat_handler},
      {kUniformSubCommand, create_uniform_handler}
  };

  const auto& handler = dispatch_map.find(config->sub_command);
  assert(handler != dispatch_map.end());
  printf("Invoking %s sub command handler.\n", config->sub_command.c_str());
  graph_t* created_graph = NULL;
  if (handler->second(config, &created_graph) == SUCCESS) {
    write_graph(created_graph, config->input_graph_file.c_str());
  } else {
    printf("Creating a graph failed!\n");
  }
}

void analyze_handler(generator_config_t* config) {
  // Defines the signature of the command/sub-command handler function.
  typedef void(*analyze_sub_command_handler_t)(generator_config_t*);

  // Maps each command/sub-command with its handler.
  const std::map<std::string, analyze_sub_command_handler_t> dispatch_map = {
    {kSummarySubCommand, analyze_summary_handler},
    {kDegreeDistributionSubCommand, analyze_degree_distribution_handler}
  };

  const auto& handler = dispatch_map.find(config->sub_command);
  assert(handler != dispatch_map.end());
  printf("Invoking %s sub command handler.\n", config->sub_command.c_str());
  handler->second(config);
}
