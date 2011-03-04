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
#define DEFAULT_EDGE_WEIGHT  0
PRIVATE const char delimiters[] = " \t\n:";


/**
 * parses the metadata at the very beginning of the graph file
 * @param[in] file_handler a handler to an opened graph file
 * @param[out] vertices_count number of vertices
 * @param[out] edgess_count number of edges
 * @param[out] directed set to true if directed
 * @return generic success or failure
 */
PRIVATE error_t parse_metadata(FILE* file_handler, uint32_t* vertices_count, 
                               uint32_t* edges_count, bool* directed) {

  // logistics for parsing
  char          line[MAX_LINE_LENGTH];
  char*         token          = NULL;
  uint32_t      metadata_lines = 3;

  assert(file_handler);
  assert(vertices_count);
  assert(edges_count);
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
     # Nodes: vertices_count
     # Edges: edges_count
     # DIRECTED|UNDIRECTED
  */
  while (metadata_lines--) {
    // get the line
    CHECK_ERR(fgets(line, sizeof(line), file_handler) != NULL, err_format);
   
    // must be a comment
    CHECK_ERR(line[0] == '#', err_format);

    // first token is one of the keywords. start after the # (hence +1)
    CHECK_ERR((token = strtok(line + 1, delimiters)) != NULL, err_format);
    to_upper(token);

    int keyword;
    for (keyword = KEYWORD_START; keyword < KEYWORD_COUNT; keyword++) {
      if (strcmp(token, keywords[keyword]) == 0) {
        keywords_found[keyword] = true;
        break;
      } 
    }
    CHECK_ERR(keyword != KEYWORD_COUNT, err_format);

    // take action based on the keyword
    switch (keyword) {
      case NODES:
      case EDGES:
        // the second token is the value
        CHECK_ERR((token = strtok(NULL, delimiters)) != NULL, err_format);
        CHECK_ERR(is_numeric(token), err_format);
        if (keyword == NODES) *vertices_count = atoi(token);
        else *edges_count = atoi(token);
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

  CHECK_ERR(keywords_found[NODES] && keywords_found[EDGES], err_format);

  return SUCCESS;

 err_format:
  fprintf(stderr, "Error in metadata format (i.e., the first three lines)");
  return FAILURE;
}

error_t graph_initialize(const char* graph_file, bool weighted, 
                         graph_t** graph) {


  /* we had to define those variables here, not within the code, to overcome a 
     compilation problem with using "goto" (used to emulate exceptions). */
  uint32_t vertex_index   = 0;
  uint32_t edge_index     = 0;
  uint32_t vertices_count = 0;
  uint32_t edges_count    = 0;
  graph_t* mygraph        = NULL;
  bool     directed       = true;

  // logistics for parsing
  char          line[MAX_LINE_LENGTH];
  char*         token       = NULL;
  uint32_t      line_number = 0;

  assert(graph_file);
  FILE* file_handler = fopen(graph_file, "r");
  CHECK_ERR(file_handler != NULL, err_openfile);

  CHECK_ERR(parse_metadata(file_handler, &vertices_count, 
                           &edges_count, &directed) == SUCCESS, err);

  /* allocate graph buffers, we allocate an extra slot in the vertices array to
     make it easy to calculate the number of neighbors of the last vertex. */
  mygraph           = (graph_t*)malloc(sizeof(graph_t));
  assert(mygraph);
  mygraph->vertices = (uint32_t*)mem_alloc((vertices_count + 1) * 
                                           sizeof(uint32_t));
  mygraph->edges    = (uint32_t*)mem_alloc(edges_count * sizeof(uint32_t));
  mygraph->weights  = weighted? 
    (int32_t*)mem_alloc(edges_count * sizeof(uint32_t)): NULL;

  // read line by line to create the graph
  while (fgets(line, sizeof(line), file_handler) != NULL) {
    // we got a new line
    line_number++;    

    // comments start with '#', skip them
    if (line[0] == '#')
      continue;
    
    // start tokenizing: first, the source node
    CHECK_ERR((token = strtok(line, delimiters)) != NULL, err_format_clean);
    CHECK_ERR(is_numeric(token), err_format_clean);
    uint32_t src_id = (uint32_t)atoi(token);

    // second, the destination node
    CHECK_ERR((token = strtok(NULL, delimiters)) != NULL, err_format_clean);
    CHECK_ERR(is_numeric(token), err_format_clean);
    uint32_t dst_id = (uint32_t)atoi(token);
    
    // third, get the weight if any
    int32_t weight = DEFAULT_EDGE_WEIGHT;
    if (weighted && ((token = strtok(NULL, delimiters)) != NULL)) {
      // TODO(abdullah): isnumeric returns false for negatives, fix this.
      //CHECK_ERR(is_numeric(token), err_format_clean);
      weight = atoi(token);
    }    

    if (src_id != vertex_index - 1) {
      // add new vertex

      // vertices must be in increasing order and less than the maximum count
      CHECK_ERR(((src_id >= vertex_index) && (src_id < vertices_count)), 
                err_format_clean);

      /* IMPORTANT: vertices without edges have the same index in the vertices 
         array as their next vertex, hence their number of edges as zero would
         be calculated in the same way as every other vertex. hence the 
         following loop. */
      while (vertex_index <= src_id) {
        mygraph->vertices[vertex_index++] = edge_index;
      }
    }

    // add the edge and its weight if any
    CHECK_ERR((edge_index < edges_count), err_format_clean);
    mygraph->edges[edge_index] = dst_id;
    if (weighted) {
      mygraph->weights[edge_index] = weight;
    }
    edge_index++;
  }
  fclose(file_handler);

  CHECK_ERR((vertex_index <= vertices_count), err_format_clean);
  CHECK_ERR((edge_index == edges_count), err_format_clean);

  // make sure we set the vertices that do not exist at the end
  while (vertex_index <= vertices_count) {
    mygraph->vertices[vertex_index++] = edge_index;
  }

  /* set the rest of the members of the graph data structure.
     TODO(abdullah): verify that the graph is actually undirected if true */
  mygraph->vertices_count = vertices_count;
  mygraph->edges_count    = edges_count;
  mygraph->directed       = directed;
  mygraph->weighted       = weighted;

  // we are done, set the output parameter and return
  *graph = mygraph;
  return SUCCESS;

  // error handling
 err_openfile:
  fprintf(stderr, "Can't open file %s\n", graph_file);
  graph_finalize(mygraph);
  goto err;
 err_format_clean:
  graph_finalize(mygraph);
  fprintf(stderr, "Incorrect file format at line number %d, check the file "
          "format described in totem_graph.h\n", line_number);
 err:
  return FAILURE;
}

error_t graph_finalize(graph_t* graph) {
  assert(graph);
  assert(graph->vertices);
  assert(graph->edges);

  // those buffers are allocated via mem_alloc
  mem_free(graph->vertices);
  mem_free(graph->edges);
  if (graph->weighted) {
    assert(graph->weights);
    mem_free(graph->weights);
  }

  // this is always allocated via malloc
  free(graph);

  return SUCCESS;
}
