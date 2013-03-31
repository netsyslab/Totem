/**
 * Defines the high-level interface of Totem framework. It offers an interface 
 * to the user of a totem-based algorithm to initialize/finalize the framework's
 * algorithm-agnostic state, and query profiling data recorded during the last
 * execution. This is basically a wrapper to the engine interface.
 *
 *  Created on: 2012-07-03
 *  Author: Abdullah Gharaibeh
 */

#include "totem_engine.cuh"

const totem_timing_t* totem_timing() {
  return &(context.timing);
}

void totem_timing_reset() {
  engine_reset_bsp_timers();
}

uint32_t totem_partition_count() {
  return engine_partition_count();
}

vid_t totem_par_vertex_count(uint32_t pid) {
  return context.vertex_count[pid];
}

eid_t totem_par_edge_count(uint32_t pid) {
  return context.edge_count[pid];
}

vid_t totem_par_rmt_vertex_count(uint32_t pid) {
  return context.rmt_vertex_count[pid];
}

eid_t totem_par_rmt_edge_count(uint32_t pid) {
  return context.rmt_edge_count[pid];
}

error_t totem_init(graph_t* graph, totem_attr_t* attr) {
  return engine_init(graph, attr);
}

void totem_finalize() {
  engine_finalize();
}
