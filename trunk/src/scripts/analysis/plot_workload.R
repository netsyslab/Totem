#! /usr/bin/Rscript
## This script plots the performance of totem under a specific workload. The
## plot shows the performance in TEPS on the y-axis with 95% confidence 
## interval, while the x-axis varies alpha (the percentage of edges processed 
## by the CPU). the script takes as input a directory of the raw performance 
## data of the workload
##
## Requires: ggplot2, grid
##
## Date: 2013-02-23
## Author: Abdullah Gharaibeh

library(ggplot2);
library(grid);

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


## Set the theme and save the plot
totem.plot.finalize <- function(plot, data, data_hybrid, filename,
                                legend_position) {
  
  ## The axes labels and limits
  ylimit = .2 * as.integer(5 * max(data_hybrid$RATE) + 2);
  plot = plot +
         scale_x_continuous("% of Edges on the CPU",
                            limits = c(min(data_hybrid$ALPHA),
                                       max(data_hybrid$ALPHA)),
                            breaks = seq(min(data_hybrid$ALPHA),
                                         max(data_hybrid$ALPHA), 5)) +
         scale_y_continuous("Billion Traversed Edges Per Second",
                            limits = c(0, ylimit),
                            breaks = seq(0, ylimit, .2));

    ## Plot the 1S and 2S lines
  data_1S = data[data$GPU_COUNT == 0 & data$CPU_COUNT == 1,]$RATE;
  data_2S = data[data$GPU_COUNT == 0 & data$CPU_COUNT == 2,]$RATE;
  plot = plot + geom_hline(yintercept = data_1S, color = "black", size = 1) +
    geom_text(x = min(data_hybrid$ALPHA), y = data_1S - .03, label = "1S",
              color = "black") +
    geom_hline(yintercept=data_2S, color = "orange", size = 1, linetype = 2) +
    geom_text(x = min(data_hybrid$ALPHA), y = data_2S + .03, label = "2S",
              color = "orange");
  
  ## The general theme of the plot
  theme_set(theme_bw());
  plot = plot + theme(panel.border = element_blank(),
                      legend.title = element_blank(),
                      legend.position = legend_position,
                      legend.text = element_text(size = 13),
                      legend.key.size = unit(1.5, "lines"),
    legend.direction = "horizontal",
                      axis.line = element_line(size = 1),
                      axis.title = element_text(size = 15),
                      axis.title.x = element_text(vjust = -0.5),
                      axis.title.y = element_text(vjust = 0.25),
                      axis.ticks = element_line(size = 1),
                      axis.text = element_text(size = 15));

  ## Save the plot
  ggsave(filename, plot, width = 7, height = 4.7);
}

## Plots the performance a hardware configuration for different partitioning
## algorithms while varying on the x-axis the percentage of edges on the CPU
## (denoted as alpha in the raw data). The plot will show the one and two
##  sockets performance as horizantal lines with a label
totem.plot.config <- function(data, filename, cpu_count = 1, gpu_count = 1,
                           legend_position = c(.5, .9)) {
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
  cat <- function(x) paste(cpu_count, "S", gpu_count, "G_", x, sep="");
  data_hybrid$LABEL = sapply(data_hybrid$PAR, cat)
  plot = ggplot(data_hybrid, aes(ALPHA, RATE)) + aes(color = factor(LABEL)) +
         geom_point(aes(shape = factor(LABEL)), size = 4) +
         geom_line(size = 1) +
         geom_errorbar(aes(ymin = RATE - RATE_CI, ymax = RATE + RATE_CI),
                       width = 1, color = "lightgray");

  totem.plot.finalize(plot, data, data_hybrid, filename, legend_position);
}

## Plots the performance of a partitioning algorithm for different hardware
## configurations while varying on the x-axis the percentage of edges on the
## CPU (denoted alpha in the raw data). The plot will show the one and two
## sockets performance as horizantal lines with a label
totem.plot.par <- function(data, filename, par = "LOW",
                              legend_position = c(.5, .9)) {
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
  plot = ggplot(data_hybrid, aes(ALPHA, RATE)) +
         aes(color = factor(CONFIG)) + geom_line(size = 1) +
         geom_point(aes(shape = factor(CONFIG)), size = 4) +
         geom_errorbar(aes(ymin = RATE - RATE_CI, ymax = RATE + RATE_CI),
                       width = 1, color = "lightgray");

  totem.plot.finalize(plot, data, data_hybrid, filename, legend_position);
}

## Plots the breakdown of exeuction time of a specific hardware configuration
## and alpha value
totem.plot.breakdown <- function(data, filename, alpha, cpu_count = 1,
                                 gpu_count = 1, legend_position = c(.5, .9)) {
  ## Get the data to plot
  data_points = subset(data, ALPHA == alpha & 
                      CPU_COUNT == cpu_count & GPU_COUNT == gpu_count);
  if (length(data_points$ALPHA) == 0) {
    print(sprintf("Warning: no data to plot breakdown for %dS%dG at %d",
                  cpu_count, gpu_count, alpha));
    return();
  }

  print(sprintf("Plotting time breakdown for %d alpha data point", alpha));
  
  ## Create a data frame such that each processing stage in a row
  data_all = data.frame(PHASE = character(0), TIME = numeric(0),
                        PAR = character(0));
  for (par in c("HIGH", "LOW", "RAN")) {
    data_point = subset(data_points, PAR == par);
    phases = c("Computation", "Communication", "Result Aggregation");
    d = data.frame(PHASE = factor(phases, levels = rev(phases)),
                   TIME = c(data_point$COMP, data_point$COMM,
                            data_point$AGGR),
                   GROUP = "Total");

    gpu = data.frame(PHASE = c("Computation"),
                     TIME = c(data_point$GPUCOMP),
                     GROUP = "GPU");
    d = rbind(d, gpu);

    d$PAR = par;
    data_all = rbind(data_all, d);
  }

  ## Plot as stacked bars
  plot = ggplot(data_all, aes(x = GROUP, y = TIME, fill = PHASE)) +
         geom_bar(stat = "identity", colour = "black", width = 1.5) +
         facet_grid(. ~ PAR);

  ## The axes labels and limits
  plot = plot + scale_x_discrete("", expand = c(.5,.5)) + scale_colour_grey() +
         scale_y_continuous("Time (ms)");

  ## Set the theme
  theme_set(theme_bw());
  plot = plot + theme(panel.border = element_blank(),
                      legend.title = element_blank(),
                      legend.text = element_text(size = 10),
                      legend.position = legend_position,
                      axis.line = element_line(size = 0),
                      axis.title = element_text(size = 15),
                      axis.ticks = element_line(size = 1),
                      axis.text = element_text(size = 15));

  plot = plot + theme(axis.text.x = element_text(angle = 45, hjust = 1),
                      panel.margin = unit(2, "lines"));

  ## Save the plot
  ggsave(filename, plot, width = 7, height = 4.7);
}

## Pre-process the data to get averages and confidence intervals
data = totem.summary(dir);

## Use the last diretory in the path as a base for the plots names
path = unlist(strsplit(dir, "/"));
imgbase = paste(plot_dir, path[length(path)], sep="/");

## Plot different possible combinations
totem.plot.config(data, paste(imgbase, "1S1G.png", sep = "_"),
                  cpu_count = 1, gpu_count = 1);
totem.plot.config(data, paste(imgbase, "2S1G.png", sep = "_"),
                  cpu_count = 2, gpu_count = 1);
totem.plot.config(data, paste(imgbase, "1S2G.png", sep = "_"),
                  cpu_count = 1, gpu_count = 2);
totem.plot.config(data, paste(imgbase, "2S2G.png", sep = "_"),
                  cpu_count = 2, gpu_count = 2);

totem.plot.par(data, paste(imgbase, "RAN.png", sep = "_"), par = "RAN");
totem.plot.par(data, paste(imgbase, "LOW.png", sep = "_"), par = "LOW");
totem.plot.par(data, paste(imgbase, "HIGH.png", sep = "_"), par = "HIGH");

## Plot the breakdown of execution time for the data point that represent
## a hybrid 1S1G configuration and minimum CPU partition size
alpha = min(subset(data, CPU_COUNT == 1 & GPU_COUNT == 1)$ALPHA);
totem.plot.breakdown(data, sprintf("%s_1S1G_%d_breakdown.png", imgbase, alpha),
                     alpha, legend_position = c(.5, .8));
