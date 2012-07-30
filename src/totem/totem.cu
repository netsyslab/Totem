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

double totem_time_initialization() {
  return context.time_init;
}

double totem_time_partitioning() {
  return context.time_par;
}

double totem_time_execution() {
  return context.time_exec;
}

double totem_time_computation() {
  return context.time_comp;
}

double totem_time_gpu_computation() {
  return context.time_gpu_comp;
}

double totem_time_communication() {
  return context.time_comm;
}

double totem_time_scatter() {
  return context.time_scatter;
}

double totem_time_aggregation() {
  return context.time_aggr;
}

uint32_t totem_partition_count() {
  return engine_partition_count();
}

uint64_t totem_par_vertex_count(uint32_t pid) {
  return context.vertex_count[pid];
}

uint64_t totem_par_edge_count(uint32_t pid) {
  return context.edge_count[pid];
}

uint64_t totem_par_rmt_vertex_count(uint32_t pid) {
  return context.rmt_vertex_count[pid];
}

uint64_t totem_par_rmt_edge_count(uint32_t pid) {
  return context.rmt_edge_count[pid];
}

error_t totem_init(graph_t* graph, totem_attr_t* attr) {
  return engine_init(graph, attr);
}

void totem_finalize() {
  engine_finalize();
}
