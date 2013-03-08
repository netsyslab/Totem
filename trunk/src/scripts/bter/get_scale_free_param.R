#! /usr/bin/Rscript
## This script calculates the parameters of a scale-free network given a
## specific gamma, number of nodes and number of edges. The script assumes
## the following power-law relation between the degree k and the number of
## nodes with degree k: nnodes(k) = round((maxdeg/k)^gamma)
## where maxdeg is the maximum degree in the graph and gamma is the shape factor
##
## The script searches for values of max and min degrees that lead to the
## required gamma, number of nodes and edges. Therefore, the algorithm takes as
## input two additional optional parameters: the first is "starting max degree",
## which directs the search to start from a specific max degree, the second is
## "increment" which determines how much to increment max degree in each
## iteration during the search. Choosing good values for those two parameters
## helps speeding up the search process. Note that choosing a very high
## "increment" value may result in less accurate result (i.e., farther from the
## requested number of vertices and edges).
##
## In particular, the script takes five arguments, three required and two
## optional in the following order:
##   ./get_scale_free_param.R tnnodes tnedges gamma [maxdeg] [inc]
##   tnnodes: target number of nodes (required)
##   tnedges: target number of edges (required)
##   gamma  : scale-free network shape factor (required)
##   maxdeg : the starting max degree (optional)
##   inc    : the increment to maxdeg in each search iteration (options)
##
## The script prints out two max and min degrees ((max1,min1) (max2, min2))
## that result in the correct number of nodes (or very close to it), and a
## number of edges such that:
## nedges(max1, min1) < target_nedges < nedges(max1, min1)
## nnodes(max1, min1) < target_nnodes < nnodes(max1, min1)
##
## The user would typically use one of the two tuples to generate a scale-free
## graph using the scale-free.m script
##
## Requires: igraph
##
## Date  : 2013-02-26
## Author: Abdullah Gharaibeh

library(igraph);

## Power-law relation between node degree and number of nodes
scale_free <- function(x, gamma, maxdeg) {
  round((maxdeg/x)^gamma);
}

## Looks for a distribution with a specified number of nodes
get.ddist.nodes <- function(gamma, tnnodes, mindeg, imaxdeg, inc) {
  finished = FALSE;
  maxdeg = imaxdeg;
  while (!finished) {
    maxdeg = maxdeg + inc;
    ddist = sapply(c(mindeg:maxdeg), scale_free, gamma, maxdeg);
    nnodes = sum(ddist);
    if (nnodes >= tnnodes) finished = TRUE;
  }
  return(ddist);
}

## Check for minimum number of arguments
if (length(commandArgs(T)) <  3 || length(commandArgs(T)) > 5) {
  print("Error: missing degree file");
  print("Usage: Rscript get_scale_free_param.R <millions of nodes> ");
  print("<millions of edges> <gamma> [starting max degree] [increment]");
  quit("no");
}

# Get command line arguments
tnnodes_g = as.integer(commandArgs(T)[1]) * 1024 * 1024;
tnedges_g = as.integer(commandArgs(T)[2]) * 1024 * 1024;
gamma_g   = as.numeric(commandArgs(T)[3]);
maxdeg_g  = 100;
inc_g     = 5;
if (length(commandArgs(T)) > 3) {
  maxdeg_g = as.numeric(commandArgs(T)[4]);
}
if (length(commandArgs(T)) > 4) {
  inc_g = as.numeric(commandArgs(T)[5]);
}

## Print the input parameters
print("target_nnodes target_nedges gamma maxdeg inc");
print(sprintf("%.0f %.0f %.2f %.0f %.0f", tnnodes_g, tnedges_g, gamma_g,
              maxdeg_g, inc_g));

## Start the search, first
maxdeg = maxdeg_g;
mindeg = 0;
nedges = 0;
curr_result = rep(0, 8);
last_result = curr_result;
while (nedges < tnedges_g) {
  last_result = curr_result;
  mindeg = mindeg + 1;
  ddist = get.ddist.nodes(gamma_g, tnnodes_g, mindeg, maxdeg, inc_g);
  nnodes = sum(ddist);
  maxdeg = length(ddist) + mindeg - 1;
  nedges = sum(c(mindeg:maxdeg) * ddist);
  gamma = (power.law.fit(rep(c(mindeg:maxdeg), ddist)))$alpha;
  curr_result = c(gamma, mindeg, maxdeg, nnodes / (1024 * 1024),
                  nedges / (1024 * 1024), nedges/nnodes, nnodes/tnnodes_g,
                  nedges/tnedges_g);
}

## Print the best two results
names(curr_result) = names(last_result) = c("gamma", "mindeg", "maxdeg",
                                            "nnodes", "nedges", "degree",
                                            "nnodes/tnnodes", "nedges/tnedges");
if (sum(last_result)) {
  ## use signif to round to only two significant digits
  print(signif(last_result, digits = 2));
}
print(signif(curr_result, digits = 2));
