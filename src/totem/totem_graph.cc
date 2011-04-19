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
#define DEFAULT_EDGE_WEIGHT  1
PRIVATE const char delimiters[] = " \t\n:";


/**
 * parses the metadata at the very beginning of the graph file
 * @param[in] file_handler a handler to an opened graph file
 * @param[out] vertex_count number of vertices
 * @param[out] edges_count number of edges
 * @param[out] directed set to true if directed
 * @return generic success or failure
 */
PRIVATE error_t parse_metadata(FILE* file_handler, uint64_t* vertex_count,
                               uint64_t* edge_count, bool* directed) {

  // logistics for parsing
  char          line[MAX_LINE_LENGTH];
  char*         token          = NULL;
  uint32_t      metadata_lines = 3;

  assert(file_handler);
  assert(vertex_count);
  assert(edge_count);
  assert(directed);

  // we assume a directed graph unless otherwise set
  *directed = true;

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
     first three lines with the following format:
     # Nodes: vertex_count
     # Edges: edge_count
     # DIRECTED|UNDIRECTED
  */
  while (metadata_lines--) {
    // get the line
    CHK(fgets(line, sizeof(line), file_handler) != NULL, err_format);

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
      case EDGES:
        // the second token is the value
        CHK((token = strtok(NULL, delimiters)) != NULL, err_format);
        CHK(is_numeric(token), err_format);
        if (keyword == NODES) *vertex_count = atoi(token);
        else *edge_count = atoi(token);
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

error_t graph_initialize(const char* graph_file, bool weighted,
                         graph_t** graph) {


  /* we had to define those variables here, not within the code, to overcome a
     compilation problem with using "goto" (used to emulate exceptions). */
  id_t vertex_index     = 0;
  id_t edge_index       = 0;
  uint64_t vertex_count = 0;
  uint64_t edge_count   = 0;
  graph_t* my_graph      = NULL;
  bool     directed     = true;

  // logistics for parsing
  char          line[MAX_LINE_LENGTH];
  char*         token       = NULL;
  uint64_t      line_number = 0;

  assert(graph_file);
  FILE* file_handler = fopen(graph_file, "r");
  CHK(file_handler != NULL, err_openfile);

  CHK(parse_metadata(file_handler, &vertex_count,
                     &edge_count, &directed) == SUCCESS, err);

  /* allocate graph buffers, we allocate an extra slot in the vertices array to
     make it easy to calculate the number of neighbors of the last vertex. */
  my_graph           = (graph_t*)calloc(1, sizeof(graph_t));
  assert(my_graph);
  my_graph->vertices = (id_t*)mem_alloc((vertex_count + 1) * sizeof(id_t));
  my_graph->edges    = (id_t*)mem_alloc(edge_count * sizeof(id_t));
  my_graph->weights  = weighted ?
    (weight_t*)mem_alloc(edge_count * sizeof(weight_t)) : NULL;

  // read line by line to create the graph
  while (fgets(line, sizeof(line), file_handler) != NULL) {
    // we got a new line
    line_number++;

    // comments start with '#', skip them
    if (line[0] == '#')
      continue;

    // start tokenizing: first, the source node
    CHK((token = strtok(line, delimiters)) != NULL, err_format_clean);
    CHK(is_numeric(token), err_format_clean);
    uint64_t token_num  = atoll(token);
    CHK((token_num < ID_MAX), err_id_overflow);
    id_t src_id = token_num;

    // second, the destination node
    CHK((token = strtok(NULL, delimiters)) != NULL, err_format_clean);
    CHK(is_numeric(token), err_format_clean);
    token_num  = atoll(token);
    CHK(token_num < ID_MAX, err_id_overflow);
    id_t dst_id = token_num;

    // third, get the weight if any
    weight_t weight = DEFAULT_EDGE_WEIGHT;
    if (weighted && ((token = strtok(NULL, delimiters)) != NULL)) {
      // TODO(abdullah): isnumeric returns false for negatives, fix this.
      //CHK(is_numeric(token), err_format_clean);
      weight = (weight_t)atof(token);
    }

    if (src_id != vertex_index - 1) {
      // add new vertex

      // vertices must be in increasing order and less than the maximum count
      CHK(((src_id >= vertex_index) && (src_id < vertex_count)),
          err_format_clean);

      /* IMPORTANT: vertices without edges have the same index in the vertices
         array as their next vertex, hence their number of edges as zero would
         be calculated in the same way as every other vertex. hence the
         following loop. */
      while (vertex_index <= src_id) {
        my_graph->vertices[vertex_index++] = edge_index;
      }
    }

    // add the edge and its weight if any
    CHK((edge_index < edge_count), err_format_clean);
    my_graph->edges[edge_index] = dst_id;
    if (weighted) {
      my_graph->weights[edge_index] = weight;
    }
    edge_index++;
  }
  fclose(file_handler);

  CHK((vertex_index <= vertex_count), err_format_clean);
  CHK((edge_index == edge_count), err_format_clean);

  // make sure we set the vertices that do not exist at the end
  while (vertex_index <= vertex_count) {
    my_graph->vertices[vertex_index++] = edge_index;
  }

  /* set the rest of the members of the graph data structure.
     TODO(abdullah): verify that the graph is actually undirected if true */
  my_graph->vertex_count = vertex_count;
  my_graph->edge_count   = edge_count;
  my_graph->directed     = directed;
  my_graph->weighted     = weighted;

  // we are done, set the output parameter and return
  *graph = my_graph;
  return SUCCESS;

  // error handling
 err_openfile:
  fprintf(stderr, "Can't open file %s\n", graph_file);
  graph_finalize(my_graph);
  goto err;
 err_id_overflow:
  fprintf(stderr, "The type used for vertex ids does not support the range of "
          "values in this file.\n");
 err_format_clean:
  graph_finalize(my_graph);
  fprintf(stderr, "Incorrect file format at line number %d, check the file "
          "format described in totem_graph.h\n", line_number);
 err:
  return FAILURE;
}

error_t graph_finalize(graph_t* graph) {
  assert(graph);

  // those buffers are allocated via mem_alloc
  if (graph->vertex_count != 0) mem_free(graph->vertices);
  if (graph->edge_count != 0) mem_free(graph->edges);
  if (graph->weighted && graph->edge_count != 0) mem_free(graph->weights);

  // this is always allocated via malloc
  free(graph);

  return SUCCESS;
}
