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

error_t graph_initialize(const char* graph_file, bool weighted,
                         graph_t** out_graph) {

  /* we had to define those variables here, not within the code, to overcome a
     compilation problem with using "goto" (used to emulate exceptions). */
  uint64_t vertex_count = 0;
  uint64_t edge_count   = 0;
  graph_t* graph        = NULL;
  bool     directed     = true;
  bool     valued       = false;

  assert(graph_file);
  FILE* file_handler = fopen(graph_file, "r");
  CHK(file_handler != NULL, err_openfile);

  // parse metadata, will allow us to allocate the graph buffers
  CHK(parse_metadata(file_handler, &vertex_count, &edge_count, 
		     &directed, &valued) == SUCCESS, err);

  /* allocate graph buffers, we allocate an extra slot in the vertices array to
     make it easy to calculate the number of neighbors of the last vertex. */
  graph           = (graph_t*)calloc(1, sizeof(graph_t));
  assert(graph);
  graph->vertices = (id_t*)mem_alloc((vertex_count + 1) * sizeof(id_t));
  graph->edges    = (id_t*)mem_alloc(edge_count * sizeof(id_t));
  graph->weights  = weighted ?
    (weight_t*)mem_alloc(edge_count * sizeof(weight_t)) : NULL;
  graph->values  = valued ?
    (weight_t*)mem_alloc(vertex_count * sizeof(weight_t)) : NULL;

  /* set the rest of the members of the graph data structure.
     TODO(abdullah): verify that the graph is actually undirected if true */
  graph->vertex_count = vertex_count;
  graph->edge_count   = edge_count;
  graph->directed     = directed;
  graph->weighted     = weighted;
  graph->valued       = valued;

  // parse the vertex list
  CHK(parse_vertex_list(file_handler, graph) == SUCCESS, err_format_clean);

  // parse the edge list
  CHK(parse_edge_list(file_handler, graph) == SUCCESS, err_format_clean);

  // we are done
  fclose(file_handler);
  *out_graph = graph;
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
