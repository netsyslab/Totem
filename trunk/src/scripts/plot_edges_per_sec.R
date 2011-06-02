## <add license>
##
## Plot the performance of each algorithm and platform in edges/second.
## It assumes that the input file has the following columns:
##
## edges: the number of edges in the graph
## time: the execution time
## graph: the name of the input graph
## algo: the name of the algorithm
## platform: a lable that identifies the platform for that execution
##
## Date: 2011-04-21
## Author: Elizeu Santos-Neto

# Requirements
library("lattice");

# Get the a data file 
graph_file = commandArgs()[4];
plot_file = commandArgs()[5];

# Load data
graph_data <- read.table(graph_file, header = T);

pdf(plot_file, height = 3.5, width = 7);

# Plot the barchart
a <- aggregate((graph_data$edges / (graph_data$time / 1000)) / 1000000, 
     	       list(algo = graph_data$algo, platform = graph_data$platform, 
	            graph = graph_data$graph), mean);

barchart( x ~ algo | graph, groups = platform, data = a, layout = c(2, 2), 
	  auto.key = list(points = FALSE, rectangles = TRUE, space = "top"),
	  scales = list(x = list(rot = 45)), 
	  ylab = "Millions of Edges / second");
dev.off();

