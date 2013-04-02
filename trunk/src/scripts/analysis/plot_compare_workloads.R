#! /usr/bin/Rscript
## This script plots the performance of totem under one or more workloads.
## The plot shows the performance in TEPS on the y-axis with 95% confidence 
## interval, while the x-axis varies the hardware configuration. The script
## takes as input the workloads' root directory. Each workload's raw data is
## assumed to be in a subdirectory under the root directory. To make things
## a bit more flexible, the script takes an optional "pattern" argument which
## can be used to filter out which workload (subdirectory) to be included
## in the plot.
##
## The scripts assumes that a workload's subdirectory name has the following
## format: name_num1M_num2M
##
## where "name" is a unique workload name that will be used in the plot's
##              legend to identify the workload (e.g., RMAT25)
##       "num1" is the number of vertices in millions
##       "num2" is the number of edges in millions
##
## Requires: ggplot2, grid
##
## Date: 2013-03-13
## Author: Abdullah Gharaibeh

library(ggplot2);
library(grid);

## Needed to pre-process the data
source("totem_summary.R")

## Check command line arguments
arg_list = commandArgs(T);
if (length(arg_list) < 3) {
  print("Error: Invalid number of arguments");
  print(paste("Usage: Rscript plot_workload.R <data directory> <image file>",
              "<x-axis: w = workload, h = hardware config> [pattern]"));
  q();
}
workloads_dir = arg_list[1];
img_file = arg_list[2];
x_axis = toupper(arg_list[3]);
if (x_axis != "W" & x_axis != "H") {
  print(sprintf("Error: Invalid x-axis argument %s", x_axis));
  print(paste("Usage: Rscript plot_workload.R <data directory> <image file>",
              "<x-axis: w = workload, h = hardware config> [pattern]"));
  q();
}

## Get the pattern if specified
workloads_pattern = NULL;
if (length(arg_list) == 4) {
  workloads_pattern = arg_list[4];
}

## Returns a data frame with the best processing rate for each possible hardware
## configuration.
totem.process.workload <- function(workload) {

  print(sprintf("Processing workload %s", workload));
  data = subset(totem.summary(workload), CPU_COUNT != 0);

  ## Add a column that represent the number of processing elements. This is
  ## useful to order the data later
  data$PROC_COUNT = data$CPU_COUNT + data$GPU_COUNT;

  ## Add a configuration column. The format is xSyG, where x is the number of
  ## CPU sockets and y is the number of GPU sockets
  data$CONFIG = factor(data$CPU_COUNT + 2 * data$GPU_COUNT,
    levels = c("1", "2", "3", "4", "5", "6"),
    lab = c("1S", "2S", "1S1G", "2S1G", "1S2G", "2S2G"));
  
  ## CPU-only configurations will have xS format
  data[data$GPU_COUNT == 0, ]$CONFIG =
    paste(data[data$GPU_COUNT == 0, ]$CPU_COUNT, "S", sep="");

  ## Get an identifier for the workload from its subdirectory name
  workload_name = unlist(strsplit(workload, "/"));
  workload_name = unlist(strsplit(workload_name[length(workload_name)], "_"));
  data$WL = toupper(workload_name[1]);

  ## Get the best rate for each hardware configuration
  data = data[order(data$RATE,decreasing=T),];
  data = data[!duplicated(data$CONFIG),];
  
  return(data[c("WL", "PROC_COUNT", "CPU_COUNT", "GPU_COUNT",
                "CONFIG", "RATE", "RATE_CI")]);
}

## Create an empty frame where the data of all workloads will be aggregated
data = data.frame(WL = character(0),
                  PROC_COUNT = numeric(0), CPU_COUNT = numeric(0),
                  GPU_COUNT = numeric(0), CONFIG = character(0),
                  RATE = numeric (0), RATE_CI = numeric (0));

## Get the list of subdirectories (workloads)
workloads = list.files(workloads_dir, workloads_pattern, full.names = T);
workloads = workloads[file.info(workloads)$isdir];
for (workload in workloads) {
  d = totem.process.workload(workload);
  if (levels(factor(d$WL)) %in% data$WL) {
    print(sprintf("Error: duplicate workload name %s", levels(factor(d$WL))));
    q();
  }
  data = rbind(data, d);
}

if (x_axis == "H") {
  ## plot configuration on the x-axis grouping by workload
  plot = ggplot(data, aes(CONFIG, RATE), aes(group = WL)) + aes(fill = WL);
  plot = plot + scale_x_discrete("Hardware Configuration");
} else {
  ## plot workloads on the x-axis grouping by configuration
  plot = ggplot(data, aes(WL, RATE), aes(group = CONFIG)) + aes(fill = CONFIG);
  plot = plot + scale_x_discrete("Workload");
}

## Plot the processing rate on the y axis
dodge = position_dodge(width=0.7);
plot = plot +
  geom_bar(colour = "black", position = dodge, width = .7, stat="identity") +
  geom_errorbar(aes(ymin = RATE - RATE_CI, ymax = RATE + RATE_CI),
                color = "lightgray", width = .4,
                position = dodge);

## Ensure a multiple of .2 limit on the y-axis
ylimit = .2 * as.integer(5 * max(data$RATE) + 2);
plot = plot + scale_y_continuous("Billion Traversed Edges Per Second",
  limits = c(0, ylimit), breaks = seq(0, ylimit, .2));

## Set the theme of the plot
theme_set(theme_bw());
plot = plot + theme(panel.border = element_blank(),
                    legend.title = element_blank(),
                    legend.position = c(.5, .95),
                    legend.text = element_text(size = 13),
                    axis.line = element_line(size = 1),
                    axis.title = element_text(size = 15),
                    axis.title.x = element_text(vjust = -0.5),
                    axis.title.y = element_text(vjust = 0.25),
                    axis.ticks = element_line(size = 1),
                    axis.text = element_text(size = 15)) +
  guides(fill = guide_legend(nrow = 2));

ggsave(img_file, plot, width = 7, height = 4.7);

