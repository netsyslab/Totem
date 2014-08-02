#! /usr/bin/Rscript
## This script loads the degree distribution of a set of graphs (can be a single
## graph), computes the power law coefficient of the distributions, and plots
## the distributions.
## The input is a two columns file, the first is the degree while the second
## is the number of nodes with that degree. The assumption is that the file
## has a header as follows:
## degree vertex_count
## x      y
## .      .
## .      .
## .      .
##
## Requirements: igraph, ggplo2
##
## Date: 2011-04-02
## Author: Elizeu Santos-Neto
##         Abdullah Gharaibeh

# Get the .degree filename from the command line
if (length(commandArgs(T)) < 1) {
  print("Error: missing degree file");
  print("Usage: Rscript degree_dist.R <degree file 1> [degree file 2] ...");
  quit("no");
}
dfiles = commandArgs(T);

# Load the igraph package
library("igraph");
library("ggplot2");

# Loads the data
ddist = data.frame(alpha = numeric(0), degree = numeric(0),
  vertex_count = numeric(0));
for (dfile in dfiles) {
  my_ddist = read.table(dfile, head=TRUE);
  a = power.law.fit(rep(my_ddist$degree, my_ddist$vertex_count))$alpha
  alpha = rep(signif(a, digits = 3), length(my_ddist$degree))
  my_ddist = cbind(alpha, my_ddist)
  ddist = rbind(ddist, my_ddist);
}

# log-log plot of vertex-degree distribution
imgfile = paste(dfiles[1],"_plot.png", sep="");
plot = ggplot(ddist, aes(degree, vertex_count)) +
  geom_point(aes(shape = factor(alpha)), size = 3) +
  scale_x_log10()     +
  scale_y_log10()     +
  xlab("Degree")      +
  ylab("# of Nodes")  +
  labs(shape='Alpha');
ggsave(imgfile, plot);
