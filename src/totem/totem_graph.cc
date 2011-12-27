/**
 * Implements the graph interface defined in totem_graph.h
 *
 *  Created on: 2011-02-28
 *  Author: Abdullah Gharaibeh
 */

// totem includes
#include "totem_graph.h"
#include "totem_mem.h"

// common logistics for parsing
#define MAX_LINE_LENGTH      100
PRIVATE const char delimiters[] = " \t\n:";
PRIVATE uint64_t line_number = 0;
PRIVATE char line[MAX_LINE_LENGTH];

/**
 * parses the metadata at the very beginning of the graph file
 * @param[in] file_handler a handler to an opened graph file
 * @param[out] vertex_count number of vertices
 * @param[out] edges_count number of edges
 * @param[out] directed set to true if directed
 * @return generic success or failure
 */
PRIVATE error_t parse_metadata(FILE* file_handler, uint64_t* vertex_count,
                               uint64_t* edge_count, bool* directed, 
                               bool* valued) {
  // logistics for parsing
  char*         token          = NULL;
  uint32_t      metadata_lines = 3;

  assert(file_handler);
  assert(vertex_count);
  assert(edge_count);
  assert(directed);

  // we assume a directed graph without vertex list unless otherwise set
  *directed = true;
  *valued   = false;

  // the following are the keywords we expect in the metadata
  char keywords[][15] = {"NODES", "EDGES", "DIRECTED", "UNDIRECTED"};
  enum {
    KEYWORD_START = 0,
    NODES = 0,
    EDGES = 1,
    DIRECTED = 2,
    UNDIRECTED = 3,
    KEYWORD_COUNT = 4
  };

  // indicates which keywords we got
  bool keywords_found[KEYWORD_COUNT] = {false, false, false, false};

  /* get the metadata, the vertex and edge counts and whether the graph is
     directed or not. The assumption is that the metadata exists is the
     first four lines demonstrated below. 
     Note that the flag [Y] after vertex_count indicates that a vertex list 
     should be expected.
     # Nodes: vertex_count [Y]
     # Edges: edge_count
     # DIRECTED|UNDIRECTED
  */
  while (metadata_lines--) {
    // get the line
    CHK(fgets(line, sizeof(line), file_handler) != NULL, err_format);
    line_number++;

    // must be a comment
    CHK(line[0] == '#', err_format);

    // first token is one of the keywords. start after the # (hence +1)
    CHK((token = strtok(line + 1, delimiters)) != NULL, err_format);
    to_upper(token);

    int keyword;
    for (keyword = KEYWORD_START; keyword < KEYWORD_COUNT; keyword++) {
      if (strcmp(token, keywords[keyword]) == 0) {
        keywords_found[keyword] = true;
        break;
      }
    }
    CHK(keyword != KEYWORD_COUNT, err_format);

    // take action based on the keyword
    switch (keyword) {
      case NODES:
        // the second token is the value
        CHK((token = strtok(NULL, delimiters)) != NULL, err_format);
        CHK(is_numeric(token), err_format);
	*vertex_count = atoi(token);
	if (((token = strtok(NULL, delimiters)) != NULL) && 
	    tolower(*token) == 'y') {
	  *valued = true;
	}
        break;
      case EDGES:
        // the second token is the value
        CHK((token = strtok(NULL, delimiters)) != NULL, err_format);
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
        // we should not be here
        assert(0);
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
  // logistics for parsing
  char*         token       = NULL;
  id_t		vertex_index = 0;

  if (!graph->valued)
    return SUCCESS;

  // read line by line
  while (vertex_index < graph->vertex_count) {

    if (fgets(line, sizeof(line), file_handler) == NULL) break;
    line_number++;
    if (line[0] == '#') continue;

    // start tokenizing: first, the vertex id
    CHK((token = strtok(line, delimiters)) != NULL, err);
    CHK(is_numeric(token), err);
    uint64_t token_num  = atoll(token);
    CHK((token_num < ID_MAX), err_id_overflow);
    id_t vertex_id = token_num;

    // second, get the value
    CHK((token = strtok(NULL, delimiters)) != NULL, err);
    // TODO(abdullah): isnumeric returns false for negatives, fix this.
    //CHK(is_numeric(token), err);
    weight_t value = (weight_t)atof(token);

    if (vertex_id != vertex_index) {
      // vertices must be in increasing order and less than the maximum count
      CHK(((vertex_id > vertex_index) && 
           (vertex_id < graph->vertex_count)), err);

      // vertices without values will be assigned a default one
      while (vertex_index < vertex_id) {
        graph->values[vertex_index++] = DEFAULT_VERTEX_VALUE;
      }
    }

    // set the value
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

  // logistics for parsing
  char* token        = NULL;
  id_t  vertex_index = 0;
  id_t  edge_index   = 0;

  // read line by line to create the graph
  while (fgets(line, sizeof(line), file_handler) != NULL) {
    // we got a new line
    line_number++;

    // comments start with '#', skip them
    if (line[0] == '#')
      continue;

    // start tokenizing: first, the source node
    CHK((token = strtok(line, delimiters)) != NULL, err);
    CHK(is_numeric(token), err);
    uint64_t token_num  = atoll(token);
    CHK((token_num < ID_MAX), err_id_overflow);
    id_t src_id = token_num;

    // second, the destination node
    CHK((token = strtok(NULL, delimiters)) != NULL, err);
    CHK(is_numeric(token), err);
    token_num  = atoll(token);
    CHK(token_num < ID_MAX, err_id_overflow);
    id_t dst_id = token_num;

    // third, get the weight if any
    weight_t weight = DEFAULT_EDGE_WEIGHT;
    if (graph->weighted && ((token = strtok(NULL, delimiters)) != NULL)) {
      // TODO(abdullah): isnumeric returns false for negatives, fix this.
      //CHK(is_numeric(token), err);
      weight = (weight_t)atof(token);
    }

    if (src_id != vertex_index - 1) {
      // add new vertex

      // vertices must be in increasing order and less than the maximum count
      CHK(((src_id >= vertex_index) && (src_id < graph->vertex_count)), err);

      /* IMPORTANT: vertices without edges have the same index in the vertices
         array as their next vertex, hence their number of edges as zero would
         be calculated in the same way as every other vertex. hence the
         following loop. */
      while (vertex_index <= src_id) {
        graph->vertices[vertex_index++] = edge_index;
      }
    }

    // add the edge and its weight if any
    CHK((edge_index < graph->edge_count), err);
    graph->edges[edge_index] = dst_id;
    if (graph->weighted) {
      graph->weights[edge_index] = weight;
    }
    edge_index++;
  }

  CHK((vertex_index <= graph->vertex_count), err);
  CHK((edge_index == graph->edge_count), err);

  // make sure we set the vertices that do not exist at the end
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
PRIVATE void allocate_graph(uint64_t vertex_count, uint64_t edge_count, 
                            bool directed, bool weighted, bool valued,
                            graph_t** graph_ret) {
  // Allocate the main structure
  graph_t* graph = (graph_t*)calloc(1, sizeof(graph_t));
  assert(graph);
  // Allocate the buffers. An extra slot is allocated in the vertices array to
  // make it easy to calculate the number of neighbors of the last vertex.
  graph->vertices = (id_t*)mem_alloc((vertex_count + 1) * sizeof(id_t));
  graph->edges    = (id_t*)mem_alloc(edge_count * sizeof(id_t));
  graph->weights  = weighted ? 
    (weight_t*)mem_alloc(edge_count * sizeof(weight_t)) : NULL;
  graph->values  = valued ?
    (weight_t*)mem_alloc(vertex_count * sizeof(weight_t)) : NULL;
  // Ensure buffer allocation
  assert((graph->vertices && graph->edges) && 
         ((valued && graph->values) || (!valued && !graph->values)) && 
         ((weighted && graph->weights) || (!weighted && !graph->weights)));
  // Set the member variables
  graph->vertex_count = vertex_count;
  graph->edge_count = edge_count;
  graph->directed = directed;
  graph->weighted = weighted;
  graph->valued = valued;

  *graph_ret = graph;
}

error_t graph_initialize(const char* graph_file, bool weighted,
                         graph_t** graph_ret) {

  /* we had to define those variables here, not within the code, to overcome a
     compilation problem with using "goto" (used to emulate exceptions). */
  graph_t* graph        = NULL;
  uint64_t vertex_count = 0;
  uint64_t edge_count   = 0;
  bool     directed     = true;
  bool     valued       = false;

  assert(graph_file);
  FILE* file_handler = fopen(graph_file, "r");
  CHK(file_handler != NULL, err_openfile);

  // get graph characteristics
  CHK(parse_metadata(file_handler, &vertex_count, &edge_count, 
		     &directed, &valued) == SUCCESS, err);

  // allocate the graph and its buffers
  allocate_graph(vertex_count, edge_count, directed, weighted, valued, &graph);

  // parse the vertex list
  CHK(parse_vertex_list(file_handler, graph) == SUCCESS, err_format_clean);

  // parse the edge list
  CHK(parse_edge_list(file_handler, graph) == SUCCESS, err_format_clean);

  // we are done
  fclose(file_handler);
  *graph_ret = graph;
  return SUCCESS;

  // error handling
  err_openfile:
    fprintf(stderr, "Can't open file %s\n", graph_file);
    graph_finalize(graph);
    goto err;
  err_format_clean:
    fclose(file_handler);
    graph_finalize(graph);
    fprintf(stderr, "Incorrect file format at line number %d.\n" 
            "Check the file format described in totem_graph.h\n", 
            line_number);
  err:
    return FAILURE;
}

error_t get_subgraph(const graph_t* graph, bool* mask, graph_t** subgraph_ret) {

  assert(graph && mask);

  // Used to map vertices in the graph to the subgraph to maintain the 
  // requirement that vertex ids start from 0 to vertex_count
  id_t* map = (id_t*)calloc(graph->vertex_count, sizeof(id_t));

  // get the number of vertices and edges of the subgraph and build the map
  uint32_t subgraph_vertex_count = 0;
  uint32_t subgraph_edge_count = 0;
  for (id_t vertex_id = 0; vertex_id < graph->vertex_count; vertex_id++) {
    if (mask[vertex_id]) {
      map[vertex_id] = subgraph_vertex_count;
      subgraph_vertex_count++;
      for (uint32_t i = graph->vertices[vertex_id]; 
           i < graph->vertices[vertex_id + 1]; i++) {
        if (mask[graph->edges[i]]) subgraph_edge_count++;     
      }
    }
  }

  // allocate the subgraph and its buffers
  graph_t* subgraph = NULL;
  allocate_graph(subgraph_vertex_count, subgraph_edge_count, graph->directed, 
                 graph->weighted, graph->valued, &subgraph);

  // build the vertex and edge lists
  uint32_t subgraph_edge_index = 0;
  uint32_t subgraph_vertex_index = 0;
  for (id_t vertex_id = 0; vertex_id < graph->vertex_count; vertex_id++) {
    if (mask[vertex_id]) {
      subgraph->vertices[subgraph_vertex_index] = subgraph_edge_index;
      if (subgraph->valued) {
        subgraph->values[subgraph_vertex_index] = graph->values[vertex_id];
      }
      subgraph_vertex_index++;

      for (uint32_t i = graph->vertices[vertex_id]; 
           i < graph->vertices[vertex_id + 1]; i++) {
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
  // set the last (fake) vertex
  subgraph->vertices[subgraph_vertex_index] = subgraph_edge_index;

  // clean up
  free(map);

  // set output
  *subgraph_ret = subgraph;
  return SUCCESS;
}

error_t graph_remove_singletons(const graph_t* graph, graph_t** subgraph) {
  // TODO(abdullah): change the signature to graph_get_k_degree_nodes
  if (!graph) return FAILURE;
  bool* mask = (bool*)calloc(graph->vertex_count, sizeof(bool));
  for (id_t vid = 0; vid < graph->vertex_count; vid++) {
    for (id_t i = graph->vertices[vid]; i < graph->vertices[vid + 1]; i++) {
      mask[graph->edges[i]] = true;
      mask[vid] = true;
    }
  }
  error_t err = get_subgraph(graph, mask, subgraph);
  free(mask);
  return err;
}

PRIVATE
void graph_match_bidirected_edges(graph_t* graph, id_t** reverse_indices) {
  // Calculate the array of indexes matching each edge to its
  // counterpart reverse edge
  (*reverse_indices) = (id_t*)mem_alloc(graph->edge_count * 2 * sizeof(id_t));
  for (id_t v = 0; v < graph->vertex_count; v++) {
    for (id_t edge_id = graph->vertices[v];
         edge_id < graph->vertices[v + 1]; edge_id++) {
      for (id_t rev_edge_id = graph->vertices[graph->edges[edge_id]];
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
graph_t* graph_create_bidirectional(graph_t* graph, id_t** reverse_indices) {
  // Create the new graph with the new data
  graph_t* new_graph;
  allocate_graph(graph->vertex_count, 2 * graph->edge_count, graph->directed,
                 graph->weighted, graph->valued, &new_graph);

  id_t new_edge_index = 0;
  for (id_t v = 0; v < graph->vertex_count; v++) {
    new_graph->vertices[v] = new_edge_index;

    // Add the forward graph edges in order and any reverse edges that might
    // come before it. Note that this assumes the given edge list is already
    // in order.
    // TODO: relax this assumption
    id_t rev_id = 0;
    id_t rev_src_v = 0;
    for (id_t edge_id = graph->vertices[v]; edge_id < graph->vertices[v + 1];
         edge_id++) {
      while (rev_id < edge_id) {
        // If we found a reverse edge, determine its source vertex
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

      // Add the forward edge
      new_graph->edges[new_edge_index] = graph->edges[edge_id];
      new_graph->weights[new_edge_index] = graph->weights[edge_id];
      new_edge_index++;
    }
    /**
     * Handle reverse edges that may come after all forward edges in the graph.
     * (eg., if (3,2) and (2,1) were forward edges in the graph, (2,3) would
     * have to be added here).
     */
    while (rev_id < graph->edge_count) {
      // If we found a reverse edge, determine its source vertex
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

  // Add the upper bound to the vertices array
  new_graph->vertices[graph->vertex_count] = new_edge_index;
  assert(new_edge_index == new_graph->edge_count);

  // Index the reverse edges
  graph_match_bidirected_edges(new_graph, reverse_indices);

  return new_graph;
}

error_t graph_finalize(graph_t* graph) {
  assert(graph);

  // those buffers are allocated via mem_alloc
  if (graph->vertex_count != 0) mem_free(graph->vertices);
  if (graph->edge_count != 0) mem_free(graph->edges);
  if (graph->weighted && graph->edge_count != 0) mem_free(graph->weights);
  if (graph->valued && graph->vertex_count != 0) mem_free(graph->values);

  // this is always allocated via malloc
  free(graph);

  return SUCCESS;
}

error_t graph_random_partition(graph_t* graph, int number_of_partitions,
                               unsigned int seed, id_t** partition_labels) {
  // Check pre-conditions
  if (graph == NULL) {
    // TODO(elizeu): Use Lauro's beautiful logging library.
    printf("ERROR: Graph object is NULL, cannot proceed with partitioning.\n");
    *partition_labels = NULL;
    return FAILURE;
  }
  // The requested number of partitions should be positive
  if ((number_of_partitions <= 0) || (graph->vertex_count == 0)) {
    printf("ERROR: Invalid number of partitions or empty graph: %d (|V|),",
           graph->vertex_count);
    printf(" %d (partitions).\n", number_of_partitions);
    *partition_labels = NULL;
    return FAILURE;
  }

  // Allocate the partition vector
  id_t* partitions = (id_t*)malloc((graph->vertex_count) * sizeof(id_t));

  // Initialize the random number generator
  srand(seed);

  for (uint64_t vertex_id = 0; vertex_id < graph->vertex_count; vertex_id++) {
    // Assign each vertex to a random partition within the range
    // (0, NUMBER_OF_PARTITIONS - 1)
    partitions[vertex_id] = rand() % number_of_partitions;
  }
  *partition_labels = partitions;
  return SUCCESS;
}
