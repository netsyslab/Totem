#! /usr/bin/Rscript
## This script plots the performance of totem under a specific workload. The
## plot shows the performance in TEPS on the y-axis with 95% confidence 
## interval, while the x-axis varies alpha (the percentage of edges processed 
## by the CPU). the script takes as input a directory of the raw performance 
## data of the workload
##
## Requires: ggplot2
##
## Date: 2013-02-23
## Author: Abdullah Gharaibeh

library(ggplot2);

## Needed to pre-process the data
source("totem_summary.R")

## Check command line arguments
if (length(commandArgs(T)) < 1 | length(commandArgs(T)) > 2 ) {
  print("Error: Invalid number of argumens");
  print("Usage: Rscript plot_workload.R <raw data dir> [plots output dir]");
  q();
}

## Raw data directory
dir = commandArgs(T)[1];
if (!file.exists(dir)) {
  print(printf("Error: direcotry %s does not exist", dir));
  q();
}

## Plots output directory
plot_dir = dir;
if (length(commandArgs(T)) == 2) {
  plot_dir = commandArgs(T)[2];
}
dir.create(plot_dir, showWarnings = FALSE);


## Plots the performance for different partitioning algorithms for specific
## hardware configuration while varying on the x-axis the percentage of edges
## on the CPU (denoted alpha in the raw data). The plot will show the 1 and 2
## Sockets performance as horizantal lines
totem.plot.par <- function(data, filename, cpu_count = 1, gpu_count = 1) {
  ## Check input parameters
  if (cpu_count <= 0 || gpu_count <= 0) {
    stop("gpu_count and cpu_count must be larger than zero");
  }

  ## Get the data to plot
  data_hybrid = subset(data, CPU_COUNT == cpu_count & GPU_COUNT == gpu_count);
  if (length(data_hybrid$ALPHA) == 0) {
    print(sprintf("Warning: no data to plot configuration %dS%dG",
                  cpu_count, gpu_count));
    return();
  }

  ## Plot the data
  print(sprintf("Plotting configuration %dS%dG, figure at: %s", cpu_count,
                gpu_count, filename));

  ## The hybrid data layer
  plot = ggplot(data_hybrid, aes(ALPHA, RATE)) + aes(color = factor(PAR)) +
         geom_line() + geom_point(aes(shape = factor(PAR)), size = 4) +
         geom_errorbar(aes(ymin = RATE - RATE_CI, ymax = RATE + RATE_CI),
                       width = 2, color = "lightgray");

  ## The 1S and 2S layer
  data_1S = data[data$GPU_COUNT == 0 & data$CPU_COUNT == 1,]$RATE;
  data_2S = data[data$GPU_COUNT == 0 & data$CPU_COUNT == 2,]$RATE;
  plot = plot + geom_hline(yintercept = data_1S, color = "black", size = 1) +
    geom_text(x = 100, y = data_1S - .03, label = "1S", color = "black") +
    geom_hline(yintercept=data_2S, color = "orange", size = 1, linetype = 2) +
    geom_text(x = 100, y = data_2S + .03, label = "2S", color = "orange");

  ## The axis labels and limits
  plot = plot + scale_x_continuous("% of Edges on the CPU", limits=c(0, 100)) +
         scale_y_continuous("Billion Traversed Edges Per Second",
                            limits = c(0, max(data$RATE)));

  ## The theme of the plot
  theme_set(theme_bw());
  plot = plot + theme(panel.border = element_blank(),
                      legend.title = element_blank(),
                      legend.position = c(.9, .9),
                      legend.text = element_text(size = 15),
                      axis.line = element_line(size = 1),
                      axis.title = element_text(size = 15),
                      axis.ticks = element_line(size = 1),
                      axis.text = element_text(size = 15));
  print(plot);
  ggsave(filename);
}

## Plots the performance for different hardware configurations for the specified
## partitioning algorithm while varying on the x-axis the percentage of edges on
## the CPU (denoted alpha in the raw data). The plot will show the One and Two
## Sockets performance as horizantal lines with a label.
totem.plot.config <- function(data, filename, par = "LOW") {
  ## Check input parameters
  par_algs = c("LOW", "HIGH", "RAN");
  if (!(par %in% par_algs)) {
    stop(sprintf("partition algorithm must be one of: %s",
                 paste(par_algs, collapse = " ")));
  }

  ## Get the data to plot
  data_hybrid = subset(data, CPU_COUNT != 0 & GPU_COUNT != 0 & PAR == par);
  if (length(data_hybrid$ALPHA) == 0) {
    print(sprintf("Warning: no data to plot %s partitioning", par));
    return();
  }
  data_hybrid$CONFIG = paste(paste(data_hybrid$CPU_COUNT, data_hybrid$GPU_COUNT,
                             sep = "S"), "G", sep="");

  ## Plot the data
  print(sprintf("Plotting %s partitioning, figure at: %s", par, filename));

  ## The hybrid data layer
  plot = ggplot(data_hybrid, aes(ALPHA, RATE)) +
         aes(color = factor(CONFIG)) +
         geom_line() + geom_point(aes(shape = factor(CONFIG)), size = 4) +
         geom_errorbar(aes(ymin = RATE - RATE_CI, ymax = RATE + RATE_CI),
                       width = 2, color = "lightgray");

  ## The 1S and 2S layer
  data_1S = data[data$GPU_COUNT == 0 & data$CPU_COUNT == 1,]$RATE;
  data_2S = data[data$GPU_COUNT == 0 & data$CPU_COUNT == 2,]$RATE;
  plot = plot + geom_hline(yintercept = data_1S, color = "black", size = 1) +
    geom_text(x = 100, y = data_1S - .03, label = "1S", color = "black") +
    geom_hline(yintercept=data_2S, color = "orange", size = 1, linetype = 2) +
    geom_text(x = 100, y = data_2S + .03, label = "2S", color = "orange");

  ## The axis labels and limits
  plot = plot + scale_x_continuous("% of Edges on the CPU", limits=c(0, 100)) +
         scale_y_continuous("Billion Traversed Edges Per Second",
                            limits = c(0, max(data$RATE)));

  ## The theme of the plot
  theme_set(theme_bw());
  plot = plot + theme(panel.border = element_blank(),
                      legend.title = element_blank(),
                      legend.position = c(.9, .9),
                      legend.text = element_text(size = 15),
                      axis.line = element_line(size = 1),
                      axis.title = element_text(size = 15),
                      axis.ticks = element_line(size = 1),
                      axis.text = element_text(size = 15));
  print(plot);
  ggsave(filename);
}

## Pre-process the data to get averages and confidence intervals
data = totem.summary(dir);

## Use the last diretory in the path as a base for the plots names
path = unlist(strsplit(dir, "/"));
imgbase = paste(plot_dir, path[length(path)], sep="/");

## Plot different possible combinations
totem.plot.par(data, paste(imgbase, "1S1G.png", sep = "_"),
               cpu_count = 1, gpu_count = 1);
totem.plot.par(data, paste(imgbase, "2S1G.png", sep = "_"),
               cpu_count = 2, gpu_count = 1);
totem.plot.par(data, paste(imgbase, "1S2G.png", sep = "_"),
               cpu_count = 1, gpu_count = 2);
totem.plot.par(data, paste(imgbase, "2S2G.png", sep = "_"),
               cpu_count = 2, gpu_count = 2);

totem.plot.config(data, paste(imgbase, "RAN.png", sep = "_"), par = "RAN");
totem.plot.config(data, paste(imgbase, "LOW.png", sep = "_"), par = "LOW");
totem.plot.config(data, paste(imgbase, "HIGH.png", sep = "_"), par = "HIGH");
