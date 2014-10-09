/**
 * Implements the graph interface defined in totem_graph.h
 *
 *  Created on: 2011-02-28
 *  Author: Abdullah Gharaibeh
 */

// totem includes
#include "totem_graph.h"
#include "totem_mem.h"
#include "totem_util.h"

// Common logistics for parsing.
const uint32_t MAX_LINE_LENGTH = 100;
PRIVATE const char delimiters[] = " \t\n:";
PRIVATE uint64_t line_number = 0;
PRIVATE char line[MAX_LINE_LENGTH];

// Common binary parameters.
const uint32_t BINARY_MAGIC_WORD = 0x10102048;

/**
 * parses the metadata at the very beginning of the graph file
 * @param[in] file_handler a handler to an opened graph file
 * @param[out] vertex_count number of vertices
 * @param[out] edges_count number of edges
 * @param[out] directed set to true if directed
 * @return generic success or failure
 */
PRIVATE error_t parse_metadata(FILE* file_handler, vid_t* vertex_count,
                               eid_t* edge_count, bool* directed,
                               bool* valued) {
  assert(file_handler && vertex_count && edge_count && directed && valued);

  // We assume a directed graph without vertex list unless otherwise set.
  *directed = true;
  *valued   = false;

  // The following are the keywords we expect in the metadata.
  char keywords[][15] = {"NODES", "EDGES", "DIRECTED", "UNDIRECTED"};
  enum {
    KEYWORD_START = 0,
    NODES = 0,
    EDGES = 1,
    DIRECTED = 2,
    UNDIRECTED = 3,
    KEYWORD_COUNT = 4
  };

  // Indicates which keywords we got.
  bool keywords_found[KEYWORD_COUNT] = {false, false, false, false};

  // Get the metadata, which includes the vertex and edge counts and whether the
  // graph is directed or not. The assumption is that the metadata exists is the
  // first four lines demonstrated below. Note that the flag [Y] after
  // vertex_count indicates that a vertex list should be expected.
  // # Nodes: vertex_count [Y]
  // # Edges: edge_count
  // # DIRECTED|UNDIRECTED
  uint32_t metadata_lines = 3;
  while (metadata_lines--) {
    CHK(fgets(line, sizeof(line), file_handler) != NULL, err_format);
    line_number++;

    // The metadata lines must start with a "#"
    CHK(line[0] == '#', err_format);

    // The first token is one of the keywords, start after the # (hence +1).
    char* token;
    char* saveptr;
    CHK((token = strtok_r(line + 1, delimiters, &saveptr)) != NULL, err_format);
    to_upper(token);

    int keyword;
    for (keyword = KEYWORD_START; keyword < KEYWORD_COUNT; keyword++) {
      if (strcmp(token, keywords[keyword]) == 0) {
        keywords_found[keyword] = true;
        break;
      }
    }
    CHK(keyword != KEYWORD_COUNT, err_format);

    switch (keyword) {
      case NODES:
        // The second token is the value.
        CHK((token = strtok_r(NULL, delimiters, &saveptr)) != NULL, err_format);
        CHK(is_numeric(token), err_format);
        *vertex_count = atoi(token);
        if (((token = strtok_r(NULL, delimiters, &saveptr)) != NULL) &&
            tolower(*token) == 'y') {
          *valued = true;
        }
        break;
      case EDGES:
        CHK((token = strtok_r(NULL, delimiters, &saveptr)) != NULL, err_format);
        CHK(is_numeric(token), err_format);
        *edge_count = atoi(token);
        break;
      case DIRECTED:
        *directed = true;
        break;
      case UNDIRECTED:
        *directed = false;
        break;
      default:
        // We should not be here.
        assert(false);
    }
  }

  CHK(keywords_found[NODES] && keywords_found[EDGES], err_format);

  return SUCCESS;

  err_format:
    fprintf(stderr, "Error in metadata format (i.e., the first three lines)");
    return FAILURE;
}

/**
 * Parse the vertex list. The vertex list assigns values to each vertex. The
 * list must be sorted. Although a value is not needed for each and every
 * vertex,a value for the last vertex (i.e., vertex id graph->vertex_count - 1)
 * is required as it is used as an end-of-list signal.
 * @param[in] file_handler a handler to an opened graph file
 * @param[in|out] graph reference to graph type to store the vertex list values
 * @return generic success or failure
 */
PRIVATE error_t parse_vertex_list(FILE* file_handler, graph_t* graph) {
  if (!graph->valued) { return SUCCESS; }
  vid_t vertex_index = 0;
  while (vertex_index < graph->vertex_count) {
    if (fgets(line, sizeof(line), file_handler) == NULL) break;
    line_number++;
    if (line[0] == '#') { continue; }

    // Start tokenizing: first, the vertex id.
    char* token;
    char* saveptr;
    CHK((token = strtok_r(line, delimiters, &saveptr)) != NULL, err);
    CHK(is_numeric(token), err);
    uint64_t token_num  = atoll(token);
    CHK((token_num < VERTEX_ID_MAX), err_id_overflow);
    vid_t vertex_id = token_num;

    // Second, get the value.
    CHK((token = strtok_r(NULL, delimiters, &saveptr)) != NULL, err);
    // TODO(abdullah): Use isnumeric to verify the value.
    weight_t value = (weight_t)atof(token);

    if (vertex_id != vertex_index) {
      // Vertices must be in increasing order and less than the maximum count.
      CHK(((vertex_id > vertex_index) &&
           (vertex_id < graph->vertex_count)), err);

      // Vertices without values will be assigned a default one.
      while (vertex_index < vertex_id) {
        graph->values[vertex_index++] = DEFAULT_VERTEX_VALUE;
      }
    }
    graph->values[vertex_index++] = value;
  }

  return SUCCESS;

  err_id_overflow:
    fprintf(stderr, "The type used for vertex ids does not support the range of"
            " values in this file.\n");
  err:
    fprintf(stderr, "parse_vertex_list\n");
    return FAILURE;
}

/**
 * parses the edge list
 * @param[in] file_handler a handler to an opened graph file
 * @param[in|out] graph reference to graph type to store the edge list
 * @return generic success or failure
 */
PRIVATE error_t parse_edge_list(FILE* file_handler, graph_t* graph) {
  vid_t  vertex_index = 0;
  eid_t  edge_index   = 0;
  while (fgets(line, sizeof(line), file_handler) != NULL) {
    line_number++;

    // Comments start with '#', skip them.
    if (line[0] == '#') { continue; }

    // Start tokenizing: first, the source node.
    char* token;
    char* saveptr;
    CHK((token = strtok_r(line, delimiters, &saveptr)) != NULL, err);
    CHK(is_numeric(token), err);
    uint64_t token_num  = atoll(token);
    CHK((token_num < VERTEX_ID_MAX), err_id_overflow);
    vid_t src_id = token_num;

    // Second, the destination node.
    CHK((token = strtok_r(NULL, delimiters, &saveptr)) != NULL, err);
    CHK(is_numeric(token), err);
    token_num  = atoll(token);
    CHK(token_num < VERTEX_ID_MAX, err_id_overflow);
    vid_t dst_id = token_num;

    // Third, get the weight if any.
    weight_t weight = DEFAULT_EDGE_WEIGHT;
    if (graph->weighted && ((token = strtok_r(NULL, delimiters, &saveptr))
                            != NULL)) {
      // TODO(abdullah): Use isnumeric to verify the value.
      weight = (weight_t)atof(token);
    }

    if (src_id != vertex_index - 1) {
      // Vertices must be in increasing order and less than the maximum count.
      CHK(((src_id >= vertex_index) && (src_id < graph->vertex_count)), err);

      // IMPORTANT: vertices without edges have the same index in the vertices
      // array as their next vertex, hence their number of edges as zero would
      // be calculated in the same way as every other vertex. hence the
      // following loop.
      while (vertex_index <= src_id) {
        graph->vertices[vertex_index++] = edge_index;
      }
    }

    // Add the edge and its weight if any.
    CHK((edge_index < graph->edge_count), err);
    graph->edges[edge_index] = dst_id;
    if (graph->weighted) {
      graph->weights[edge_index] = weight;
    }
    edge_index++;
  }

  CHK((vertex_index <= graph->vertex_count), err);
  CHK((edge_index == graph->edge_count), err);

  // Make sure we set the vertices that do not exist at the end.
  while (vertex_index <= graph->vertex_count) {
    graph->vertices[vertex_index++] = edge_index;
  }

  return SUCCESS;

  err_id_overflow:
    fprintf(stderr, "The type used for vertex ids does not support the range "
            "of values in this file.\n");
  err:
    fprintf(stderr, "parse_edge_list\n");
    return FAILURE;
}

PRIVATE error_t graph_initialize_binary(FILE* fh, bool load_weights,
                                        graph_t** graph) {
  // Read vertex and edge id sizes and make sure they comply with the compiled
  // version of Totem.
  // TODO(abdullah): Have a portable way of reading the graph that does not
  // depend on the compiled types of edge and vertex id.
  uint32_t vid_size;
  CHK(fread(&vid_size, sizeof(uint32_t), 1, fh) == 1, err);
  uint32_t eid_size;
  CHK(fread(&eid_size, sizeof(uint32_t), 1, fh) == 1, err);
  CHK((vid_size == sizeof(vid_t)) && eid_size == sizeof(eid_t), err);
  vid_t vertex_count;
  CHK(fread(&vertex_count, sizeof(vid_t), 1, fh) == 1, err);
  eid_t edge_count;
  CHK(fread(&edge_count, sizeof(eid_t), 1, fh) == 1, err);
  bool valued;
  CHK(fread(&valued, sizeof(bool), 1, fh) == 1, err);
  bool weighted;
  CHK(fread(&weighted, sizeof(bool), 1, fh) == 1, err);
  bool directed;
  CHK(fread(&directed, sizeof(bool), 1, fh) == 1, err);
  graph_allocate(vertex_count, edge_count, directed, load_weights, valued,
                 graph);

  // Load the vertices and their values if any.
  CHK(fread((*graph)->vertices, sizeof(eid_t), vertex_count + 1, fh) ==
      (vertex_count + 1), err_free);
  if (valued) {
    CHK(fread((*graph)->values, sizeof(weight_t), vertex_count, fh) ==
        vertex_count, err_free);
  }

  // Load the edges and their weights if any.
  CHK(fread((*graph)->edges, sizeof(vid_t), edge_count, fh) ==
      edge_count, err_free);
  if (load_weights) {
    if (weighted) {
      CHK(fread((*graph)->weights, sizeof(weight_t), edge_count, fh) ==
          edge_count, err_free);
    } else {
      // Set weights to the default value.
      OMP(omp parallel for)
      for (eid_t e = 0; e < edge_count; e++) {
        (*graph)->weights[e] = DEFAULT_EDGE_WEIGHT;
      }
    }
  }

  fclose(fh);
  return SUCCESS;

 err_free:
  graph_finalize(*graph);
 err:
  printf("Error invalid binary file format\n");
  fclose(fh);
  return FAILURE;
}

PRIVATE error_t graph_initialize_text(FILE* file_handler, bool weighted,
                                      graph_t** graph_ret) {
  // We had to define those variables here, not within the code, to overcome a
  // compilation problem with using "goto" (used to emulate exceptions).
  graph_t* graph        = NULL;
  vid_t    vertex_count = 0;
  eid_t    edge_count   = 0;
  bool     directed     = true;
  bool     valued       = false;

  // Get graph characteristics.
  CHK(parse_metadata(file_handler, &vertex_count, &edge_count,
                     &directed, &valued) == SUCCESS, err);

  // Allocate the graph and its buffers.
  graph_allocate(vertex_count, edge_count, directed, weighted, valued, &graph);

  // Parse the vertex list.
  CHK(parse_vertex_list(file_handler, graph) == SUCCESS, err_format_clean);

  // Parse the edge list.
  CHK(parse_edge_list(file_handler, graph) == SUCCESS, err_format_clean);

  fclose(file_handler);
  *graph_ret = graph;
  return SUCCESS;

  // Error handling.
  err_format_clean:
    fclose(file_handler);
    graph_finalize(graph);
    fprintf(stderr, "Incorrect file format at line number %d.\n"
            "Check the file format described in totem_graph.h\n",
            line_number);
  err:
    return FAILURE;
}

/**
 * Allocates space for a graph structure and its buffers. It also, sets the
 * various members of the structure.
 * @param[in] vertex_count number of vertices
 * @param[in] edge_count number of edges
 * @param[in] weighted indicates if the edge weights are to be loaded
 * @param[in] valued indicates if the vertex values are to be loaded
 * @param[out] graph reference to allocated graph type to store the edge list
 * @return generic success or failure
 */
void graph_allocate(vid_t vertex_count, eid_t edge_count, bool directed,
                    bool weighted, bool valued, graph_t** graph_ret) {
  graph_t* graph = reinterpret_cast<graph_t*>(calloc(1, sizeof(graph_t)));
  assert(graph);
  // Allocate the buffers. An extra slot is allocated in the vertices array to
  // make it easy to calculate the number of neighbors of the last vertex.
  graph->vertices = reinterpret_cast<eid_t*>(malloc((vertex_count + 1) *
                                                    sizeof(eid_t)));
  graph->edges = reinterpret_cast<vid_t*>(malloc(edge_count * sizeof(vid_t)));
  graph->weights = weighted ?
      reinterpret_cast<weight_t*>(malloc(edge_count * sizeof(weight_t))) : NULL;
  graph->values = valued ?
      reinterpret_cast<weight_t*>(malloc(vertex_count * sizeof(weight_t))) :
      NULL;
  assert((graph->vertices && graph->edges) &&
         ((valued && graph->values) || (!valued && !graph->values)) &&
         ((weighted && graph->weights) || (!weighted && !graph->weights)));
  // Set the member variables.
  graph->vertex_count = vertex_count;
  graph->edge_count = edge_count;
  graph->directed = directed;
  graph->weighted = weighted;
  graph->valued = valued;
  *graph_ret = graph;
}

error_t graph_initialize(const char* graph_file, bool weighted,
                         graph_t** graph) {
  assert(graph_file);
  FILE* file_handler = fopen(graph_file, "r");
  CHK(file_handler != NULL, err);

  uint32_t magic_word;
  CHK(fread(&magic_word, sizeof(uint32_t), 1, file_handler) == 1, err_close);

  if (magic_word == BINARY_MAGIC_WORD) {
    return graph_initialize_binary(file_handler, weighted, graph);
  }

  fseek(file_handler, 0, 0);
  return graph_initialize_text(file_handler, weighted, graph);

 err_close:
  fclose(file_handler);
 err:
  fprintf(stderr, "Can't open file %s\n", graph_file);
  return FAILURE;
}

error_t get_subgraph(const graph_t* graph, bool* mask, graph_t** subgraph_ret) {
  assert(graph && mask);

  // Used to map vertices in the graph to the subgraph to maintain the
  // requirement that vertex ids start from 0 to vertex_count.
  vid_t* map = reinterpret_cast<vid_t*>(calloc(graph->vertex_count,
                                               sizeof(vid_t)));

  // Get the number of vertices and edges of the subgraph and build the map.
  vid_t subgraph_vertex_count = 0;
  eid_t subgraph_edge_count = 0;
  for (vid_t vertex_id = 0; vertex_id < graph->vertex_count; vertex_id++) {
    if (mask[vertex_id]) {
      map[vertex_id] = subgraph_vertex_count;
      subgraph_vertex_count++;
      for (eid_t i = graph->vertices[vertex_id];
           i < graph->vertices[vertex_id + 1]; i++) {
        if (mask[graph->edges[i]]) subgraph_edge_count++;
      }
    }
  }

  assert(subgraph_vertex_count <= graph->vertex_count &&
         subgraph_edge_count <= graph->edge_count);

  // Allocate the subgraph and its buffers.
  graph_t* subgraph = NULL;
  graph_allocate(subgraph_vertex_count, subgraph_edge_count, graph->directed,
                 graph->weighted, graph->valued, &subgraph);

  // Build the vertex and edge lists.
  eid_t subgraph_edge_index = 0;
  vid_t subgraph_vertex_index = 0;
  for (vid_t vid = 0; vid < graph->vertex_count; vid++) {
    if (mask[vid]) {
      subgraph->vertices[subgraph_vertex_index] = subgraph_edge_index;
      if (subgraph->valued) {
        subgraph->values[subgraph_vertex_index] = graph->values[vid];
      }
      subgraph_vertex_index++;

      for (eid_t i = graph->vertices[vid]; i < graph->vertices[vid + 1]; i++) {
        if (mask[graph->edges[i]]) {
          subgraph->edges[subgraph_edge_index] = map[graph->edges[i]];
          if (subgraph->weighted) {
            subgraph->weights[subgraph_edge_index] = graph->weights[i];
          }
          subgraph_edge_index++;
        }
      }
    }
  }
  subgraph->vertices[subgraph_vertex_index] = subgraph_edge_index;

  free(map);
  *subgraph_ret = subgraph;
  return SUCCESS;
}

error_t graph_remove_singletons(const graph_t* graph, graph_t** subgraph) {
  // TODO(abdullah): Change the signature to graph_get_k_degree_nodes.
  assert(graph);
  bool* mask = reinterpret_cast<bool*>(
      calloc(graph->vertex_count, sizeof(bool)));
  OMP(omp parallel for schedule(guided))
  for (vid_t vid = 0; vid < graph->vertex_count; vid++) {
    for (eid_t i = graph->vertices[vid]; i < graph->vertices[vid + 1]; i++) {
      mask[graph->edges[i]] = true;
      mask[vid] = true;
    }
  }

  error_t err = get_subgraph(graph, mask, subgraph);
  free(mask);
  return err;
}

PRIVATE
void graph_match_bidirected_edges(graph_t* graph, eid_t** reverse_indices) {
  // Calculate the array of indexes matching each edge to its counterpart
  // reverse edge.
  totem_malloc(graph->edge_count * 2 * sizeof(eid_t), TOTEM_MEM_HOST,
               reinterpret_cast<void**>(reverse_indices));
  for (vid_t v = 0; v < graph->vertex_count; v++) {
    for (eid_t edge_id = graph->vertices[v];
         edge_id < graph->vertices[v + 1]; edge_id++) {
      for (eid_t rev_edge_id = graph->vertices[graph->edges[edge_id]];
           rev_edge_id < graph->vertices[graph->edges[edge_id] + 1];
           rev_edge_id++) {
        if (graph->edges[rev_edge_id] == v) {
          (*reverse_indices)[edge_id] = rev_edge_id;
          break;
        }
      }
    }
  }
}

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
graph_t* graph_create_bidirectional(graph_t* graph, eid_t** reverse_indices) {
  // Create the new graph with the new data.
  graph_t* new_graph;
  graph_allocate(graph->vertex_count, 2 * graph->edge_count, graph->directed,
                 graph->weighted, graph->valued, &new_graph);

  eid_t new_edge_index = 0;
  for (vid_t v = 0; v < graph->vertex_count; v++) {
    new_graph->vertices[v] = new_edge_index;

    // Add the forward graph edges in order and any reverse edges that might
    // come before it. Note that this assumes the given edge list is already
    // in order.
    // TODO(abdullah): Relax this assumption.
    eid_t rev_id = 0;
    vid_t rev_src_v = 0;
    for (eid_t edge_id = graph->vertices[v]; edge_id < graph->vertices[v + 1];
         edge_id++) {
      while (rev_id < edge_id) {
        // If we found a reverse edge, determine its source vertex.
        if (graph->edges[rev_id] == v) {
          while (!(rev_id >= graph->vertices[rev_src_v] &&
                   rev_id < graph->vertices[rev_src_v + 1]) &&
                   rev_src_v < graph->vertex_count)
            rev_src_v++;
          new_graph->edges[new_edge_index] = rev_src_v;
          new_graph->weights[new_edge_index] = 0;
          new_edge_index++;
        }
        rev_id++;
      }

      // Add the forward edge.
      new_graph->edges[new_edge_index] = graph->edges[edge_id];
      new_graph->weights[new_edge_index] = graph->weights[edge_id];
      new_edge_index++;
    }

    // Handle reverse edges that may come after all forward edges in the graph.
    // (eg., if (3,2) and (2,1) were forward edges in the graph, (2,3) would
    // have to be added here).
    while (rev_id < graph->edge_count) {
      // If we found a reverse edge, determine its source vertex.
      if (graph->edges[rev_id] == v) {
        while (!(rev_id >= graph->vertices[rev_src_v] &&
                 rev_id < graph->vertices[rev_src_v + 1]) &&
                 rev_src_v < graph->vertex_count)
          rev_src_v++;
        new_graph->edges[new_edge_index] = rev_src_v;
        new_graph->weights[new_edge_index] = 0;
        new_edge_index++;
      }
      rev_id++;
    }
  }

  // Add the upper bound to the vertices array.
  new_graph->vertices[graph->vertex_count] = new_edge_index;
  assert(new_edge_index == new_graph->edge_count);

  // Index the reverse edges.
  graph_match_bidirected_edges(new_graph, reverse_indices);

  return new_graph;
}

error_t graph_finalize(graph_t* graph) {
  assert(graph);
  if (graph->vertex_count != 0) free(graph->vertices);
  if (graph->edge_count != 0) free(graph->edges);
  if (graph->weighted && graph->edge_count != 0) free(graph->weights);
  if (graph->valued && graph->vertex_count != 0) free(graph->values);
  free(graph);
  return SUCCESS;
}

PRIVATE eid_t get_device_edge_count_limit(const graph_t* graph) {
  // TODO(abdullah): The following constants have been determined haphazardly,
  // we need a better way to set them.
  const eid_t max_edge_count_limit = 1024 * 1024 * 1024 * 1.3;
  const size_t state_per_vertex = 4;
  size_t available = 0; size_t total = 0;
  CALL_CU_SAFE(cudaMemGetInfo(&available, &total));
  assert(available > (graph->vertex_count * state_per_vertex));
  size_t state_edges = available -
         (graph->vertex_count * state_per_vertex);

  eid_t gpu_edge_count_limit = ((int64_t)state_edges / sizeof(vid_t));
  if (gpu_edge_count_limit > max_edge_count_limit)
    gpu_edge_count_limit = max_edge_count_limit;

  return gpu_edge_count_limit;
}

PRIVATE void initialize_device_partitioned_edges(
    const graph_t* graph_h, graph_t* graph_d) {
  eid_t gpu_edge_count_limit = get_device_edge_count_limit(graph_h);
  assert(graph_h->edge_count > gpu_edge_count_limit);

  // Calculate the boundaries, vertex id at which the extended edge list should
  // be used, and the number of edges placed on the device.
  graph_d->vertex_ext = 0;
  while (graph_h->vertices[graph_d->vertex_ext] < gpu_edge_count_limit) {
    graph_d->vertex_ext++;
  }
  graph_d->edge_count_ext = graph_h->vertices[graph_d->vertex_ext];
  assert(graph_d->edge_count_ext < graph_d->edge_count);

  // Device memory partition.
  CALL_SAFE(totem_malloc(graph_d->edge_count_ext * sizeof(vid_t),
                         TOTEM_MEM_DEVICE,
                         reinterpret_cast<void**>(&graph_d->edges)));
  CALL_SAFE(cudaMemcpy(graph_d->edges, graph_h->edges,
                       graph_d->edge_count_ext * sizeof(vid_t),
                       cudaMemcpyDefault));

  // Mapped memory partition.
  eid_t mapped_edge_count = graph_d->edge_count - graph_d->edge_count_ext;
  assert(mapped_edge_count);
  CALL_SAFE(totem_malloc(mapped_edge_count * sizeof(vid_t),
                         TOTEM_MEM_HOST_MAPPED,
                         reinterpret_cast<void**>(&graph_d->mapped_edges)));
  CALL_CU_SAFE(cudaHostGetDevicePointer(reinterpret_cast<void**>
                                        (&(graph_d->edges_ext)),
                                        graph_d->mapped_edges, 0));
  CALL_SAFE(cudaMemcpy(graph_d->edges_ext,
                       &graph_h->edges[graph_d->edge_count_ext],
                       mapped_edge_count * sizeof(vid_t),
                       cudaMemcpyDefault));
}

PRIVATE void initialize_device_vertices(
    const graph_t* graph_h, graph_t* graph_d) {
  if (graph_d->compressed_vertices) {
    assert(graph_d->gpu_graph_mem != GPU_GRAPH_MEM_MAPPED &&
           graph_d->gpu_graph_mem != GPU_GRAPH_MEM_MAPPED_VERTICES);

    eid_device_t* vertices_h;
    CALL_SAFE(totem_malloc((graph_h->vertex_count + 1) * sizeof(eid_device_t),
                           TOTEM_MEM_HOST,
                           reinterpret_cast<void**>(&vertices_h)));
    OMP(omp parallel for)
    for (vid_t i = 0; i < graph_h->vertex_count + 1; i++) {
      vertices_h[i] = static_cast<eid_device_t>(graph_h->vertices[i]);
    }

    CALL_SAFE(totem_malloc((graph_h->vertex_count + 1) * sizeof(eid_device_t),
                           TOTEM_MEM_DEVICE,
                           reinterpret_cast<void**>(&graph_d->vertices_d)));
    CALL_SAFE(cudaMemcpy(graph_d->vertices_d, vertices_h,
                         (graph_h->vertex_count + 1) * sizeof(eid_device_t),
                         cudaMemcpyDefault));
    totem_free(vertices_h, TOTEM_MEM_HOST);
  } else {
    vid_t vertex_count_allocated =
        vwarp_default_state_length(graph_d->vertex_count);

    switch (graph_d->gpu_graph_mem) {
      case GPU_GRAPH_MEM_DEVICE:
      case GPU_GRAPH_MEM_MAPPED_EDGES:
      case GPU_GRAPH_MEM_PARTITIONED_EDGES:
        CALL_SAFE(totem_malloc((vertex_count_allocated + 1) * sizeof(eid_t),
                               TOTEM_MEM_DEVICE,
                               reinterpret_cast<void**>(&graph_d->vertices)));
        break;
      case GPU_GRAPH_MEM_MAPPED:
      case GPU_GRAPH_MEM_MAPPED_VERTICES:
        CALL_SAFE(totem_malloc((vertex_count_allocated + 1) * sizeof(eid_t),
                               TOTEM_MEM_HOST_MAPPED,
                               reinterpret_cast<void**>
                               (&graph_d->mapped_vertices)));
        CALL_CU_SAFE(cudaHostGetDevicePointer(reinterpret_cast<void**>
                                              (&(graph_d->vertices)),
                                              graph_d->mapped_vertices, 0));
        break;
      default:
        fprintf(stderr, "Not supported graph memory type %d\n",
                graph_d->gpu_graph_mem);
        assert(false);
    }
    CALL_SAFE(cudaMemcpy(graph_d->vertices, graph_h->vertices,
                         (graph_h->vertex_count + 1) * sizeof(eid_t),
                         cudaMemcpyDefault));
    // Set the index of the extra vertices to the last actual vertex. This
    // renders the padded fake vertices with zero edges.
    int pad_size = vwarp_default_state_length(graph_h->vertex_count) -
        graph_h->vertex_count;
    if (pad_size > 0) {
      totem_memset(&(graph_d->vertices[graph_h->vertex_count + 1]),
                   graph_h->vertices[graph_h->vertex_count], pad_size,
                   TOTEM_MEM_DEVICE);
    }
  }
}

PRIVATE void initialize_device_edges(const graph_t* graph_h, graph_t* graph_d) {
  if (graph_h->edge_count == 0) { return; }

  gpu_graph_mem_t gpu_graph_mem = graph_d->gpu_graph_mem;
  graph_d->vertex_ext = graph_d->vertex_count;
  if ((gpu_graph_mem == GPU_GRAPH_MEM_PARTITIONED_EDGES) &&
      (graph_h->edge_count <= get_device_edge_count_limit(graph_h))) {
    gpu_graph_mem = GPU_GRAPH_MEM_DEVICE;
  }

  switch (gpu_graph_mem) {
    case GPU_GRAPH_MEM_DEVICE:
    case GPU_GRAPH_MEM_MAPPED_VERTICES:
      CALL_SAFE(totem_malloc(graph_d->edge_count * sizeof(vid_t),
                             TOTEM_MEM_DEVICE,
                             reinterpret_cast<void**>(&graph_d->edges)));
      break;
    case GPU_GRAPH_MEM_MAPPED:
    case GPU_GRAPH_MEM_MAPPED_EDGES:
      CALL_SAFE(totem_malloc(graph_d->edge_count * sizeof(vid_t),
                             TOTEM_MEM_HOST_MAPPED,
                             reinterpret_cast<void**>
                             (&graph_d->mapped_edges)));
      CALL_CU_SAFE(cudaHostGetDevicePointer(reinterpret_cast<void**>
                                            (&(graph_d->edges)),
                                            graph_d->mapped_edges, 0));
      break;
    case GPU_GRAPH_MEM_PARTITIONED_EDGES:
      initialize_device_partitioned_edges(graph_h, graph_d);
      return;
    default:
      fprintf(stderr, "Not supported graph memory type %d\n",
              graph_d->gpu_graph_mem);
      assert(false);
  }

  CALL_SAFE(cudaMemcpy(graph_d->edges, graph_h->edges,
                       graph_h->edge_count * sizeof(vid_t),
                       cudaMemcpyDefault));
}

error_t graph_initialize_device(const graph_t* graph_h, graph_t** graph_d,
                                gpu_graph_mem_t gpu_graph_mem,
                                bool compressed_vertices_supported) {
  assert(graph_h);

  // Allocate the graph struct that will host references to device buffers.
  CALL_SAFE(totem_malloc(sizeof(graph_t), TOTEM_MEM_HOST,
                         reinterpret_cast<void**>(graph_d)));

  // Copy basic data types within the structure, the buffers pointers will be
  // overwritten next with device pointers.
  **graph_d = *graph_h;
  (*graph_d)->gpu_graph_mem = gpu_graph_mem;

  const eid_t kMaxEdgeCount =
      (static_cast<int64_t>(4) * 1024 * 1024 * 1024) - 1;
#ifdef FEATURE_64BIT_EDGE_ID
  (*graph_d)->compressed_vertices = compressed_vertices_supported &&
      (graph_h->edge_count <= kMaxEdgeCount);
#else
  (*graph_d)->compressed_vertices = false;
#endif

  // Nothing to be done if this is an empty graph.
  if (graph_h->vertex_count > 0) {
    // Allocate device buffers and copy data to the GPU.
    initialize_device_vertices(graph_h, *graph_d);
    initialize_device_edges(graph_h, *graph_d);
    if (graph_h->weighted) {
      CALL_SAFE(totem_malloc(graph_h->edge_count * sizeof(weight_t),
                             TOTEM_MEM_DEVICE,
                             reinterpret_cast<void**>(&(*graph_d)->weights)));
      CALL_CU_SAFE(cudaMemcpy((*graph_d)->weights, graph_h->weights,
                              graph_h->edge_count * sizeof(weight_t),
                              cudaMemcpyDefault));
    }
  }

  return SUCCESS;
}

void graph_finalize_device(graph_t* graph_d) {
  assert(graph_d);
  if (graph_d->vertex_count) {
    if (graph_d->compressed_vertices) {
      totem_free(graph_d->vertices_d, TOTEM_MEM_DEVICE);
    } else if (graph_d->gpu_graph_mem == GPU_GRAPH_MEM_MAPPED ||
               graph_d->gpu_graph_mem == GPU_GRAPH_MEM_MAPPED_VERTICES) {
      totem_free(graph_d->mapped_vertices, TOTEM_MEM_HOST_MAPPED);
    } else {
      totem_free(graph_d->vertices, TOTEM_MEM_DEVICE);
    }
  }

  if (graph_d->edge_count) {
    if ((graph_d->gpu_graph_mem == GPU_GRAPH_MEM_MAPPED) ||
        (graph_d->gpu_graph_mem == GPU_GRAPH_MEM_MAPPED_EDGES)) {
      totem_free(graph_d->mapped_edges, TOTEM_MEM_HOST_MAPPED);
    } else if ((graph_d->gpu_graph_mem == GPU_GRAPH_MEM_DEVICE) ||
               ((graph_d->gpu_graph_mem == GPU_GRAPH_MEM_PARTITIONED_EDGES) &&
                (graph_d->vertex_ext < graph_d->vertex_count))) {
      totem_free(graph_d->edges, TOTEM_MEM_DEVICE);
    } else if (graph_d->gpu_graph_mem == GPU_GRAPH_MEM_PARTITIONED_EDGES) {
      totem_free(graph_d->edges, TOTEM_MEM_DEVICE);
      totem_free(graph_d->mapped_edges, TOTEM_MEM_HOST_MAPPED);
    }
  }

  if (graph_d->weighted) totem_free(graph_d->weights, TOTEM_MEM_DEVICE);
  totem_free(graph_d, TOTEM_MEM_HOST);
}

void graph_print(graph_t* graph) {
  assert(graph);
  printf("#Nodes:%d\n#Edges:%d\n", graph->vertex_count, graph->edge_count);
  if (graph->directed) {
    printf("#Directed\n");
  } else {
    printf("#Undirected\n");
  }
  for (vid_t vid = 0; vid < graph->vertex_count; vid++) {
    for (eid_t i = graph->vertices[vid]; i < graph->vertices[vid + 1]; i++) {
      fprintf(stdout, "%d %d", vid, graph->edges[i]);
      if (graph->weighted) {
        fprintf(stdout, " %f\n", graph->weights[i]);
      } else {
        fprintf(stdout, "\n");
      }
    }
  }
}

error_t graph_store_binary(graph_t* graph, const char* filename) {
  assert(graph);
  FILE* fh = fopen(filename, "wb");
  if (fh == NULL) return FAILURE;

  uint32_t word = BINARY_MAGIC_WORD;
  CHK(fwrite(&word, sizeof(uint32_t), 1, fh) == 1, err);

  // Write the graph's parameters.
  word = sizeof(vid_t);
  CHK(fwrite(&word, sizeof(uint32_t), 1, fh) == 1, err);
  word = sizeof(eid_t);
  CHK(fwrite(&word, sizeof(uint32_t), 1, fh) == 1, err);
  CHK(fwrite(&(graph->vertex_count), sizeof(vid_t), 1, fh) == 1, err);
  CHK(fwrite(&(graph->edge_count), sizeof(eid_t), 1, fh) == 1, err);
  CHK(fwrite(&(graph->valued), sizeof(bool), 1, fh) == 1, err);
  CHK(fwrite(&(graph->weighted), sizeof(bool), 1, fh) == 1, err);
  CHK(fwrite(&(graph->directed), sizeof(bool), 1, fh) == 1, err);

  // Write the vertices array and the vertices values if any.
  CHK(fwrite(graph->vertices, sizeof(eid_t), graph->vertex_count + 1, fh) ==
      (graph->vertex_count + 1), err);
  if (graph->valued) {
    CHK(fwrite(graph->values, sizeof(weight_t), graph->vertex_count, fh) ==
        graph->vertex_count, err);
  }

  // Write the edges array and the edge weights if any.
  CHK(fwrite(graph->edges, sizeof(vid_t), graph->edge_count, fh) ==
      graph->edge_count, err);
  if (graph->weighted) {
    CHK(fwrite(graph->weights, sizeof(weight_t), graph->edge_count, fh) ==
        graph->edge_count, err);
  }

  fclose(fh);
  return SUCCESS;

 err:
  return FAILURE;
}

void graph_sort_nbrs(graph_t* graph, bool edge_sort_dsc) {
  OMP(omp parallel for schedule(guided))
  for (vid_t v = 0; v < graph->vertex_count; v++) {
    vid_t* nbrs = &graph->edges[graph->vertices[v]];

    // Sort based off of direction given.
    qsort(nbrs, graph->vertices[v+1] - graph->vertices[v], sizeof(vid_t),
          edge_sort_dsc ? compare_ids_dsc : compare_ids_asc);
    // TODO(treza): Required updates for edge-weights.
  }
}

PRIVATE graph_t* graph_g = NULL;
PRIVATE bool edge_sort_dsc_g = false;
PRIVATE int compare_ids_by_degree_dsc(const void* a, const void* b) {
  vid_t v1 = *(reinterpret_cast<const vid_t*>(a));
  vid_t v2 = *(reinterpret_cast<const vid_t*>(b));
  vid_t v1_nbrs = graph_g->vertices[v1 + 1] - graph_g->vertices[v1];
  vid_t v2_nbrs = graph_g->vertices[v2 + 1] - graph_g->vertices[v2];
  if (edge_sort_dsc_g) { return v2_nbrs - v1_nbrs; }
  return v1_nbrs - v2_nbrs;
}

void graph_sort_nbrs_by_degree(graph_t* graph, bool edge_sort_dsc) {
  // TODO(abdullah): this function is not reentrant as it uses global shared
  // variables, make it reentrant.
  graph_g = graph;
  edge_sort_dsc_g = edge_sort_dsc;
  OMP(omp parallel for schedule(guided))
  for (vid_t v = 0; v < graph->vertex_count; v++) {
    vid_t* nbrs = &graph->edges[graph->vertices[v]];
    qsort(nbrs, graph->vertices[v + 1] - graph->vertices[v], sizeof(vid_t),
          compare_ids_by_degree_dsc);
  }
}
