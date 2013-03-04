#! /usr/bin/Rscript
## This script loads the node degrees of a graph, computes the power law
## coefficient of the empirical distribution, and plots the distribution.
## The input is a two columns file, the first is the degree while the second
## is the number of nodes with that degree. The assumption is that the file
## has a header as follows:
## degree nnodes
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
if (length(commandArgs(T)) != 1) {
  print("Error: missing degree file");
  print("Usage: Rscript degree_dist.R <degree file>");
  quit("no");
}
dfile = commandArgs(T)[1];

# Load the igraph package
library("igraph");
library("ggplot2");

# Loads the data and assumes that the node degree is in the
# first column.
ddist = read.table(dfile, head=TRUE);

# Fit the power law and get the alpha
a = power.law.fit(rep(ddist$degree, ddist$nnodes));

# log-log plot of vertex-degree distribution
imgfile = paste(dfile,"_plot.png", sep="");
ggplot(ddist, aes(nnodes, degree)) +
  scale_x_log10()     +
  scale_y_log10()     +
  stat_smooth()       +
  ylab("Degree")      +
  xlab("# of Nodes")  +
  ggtitle(paste("alpha = ", a$alpha)) +
  geom_point();
ggsave(imgfile);
