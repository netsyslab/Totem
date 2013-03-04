## <add license>
##
## Computes the correlations between the graph characteristics and the running
## time of a particular algorithm running on the GPU. It receives two input
## files. The first contains the following columns:
##
## edges: the number of edges in the graph
## time: the execution time
## graph: the name of the input graph
## algo: the name of the algorithm
## platform: a lable that identifies the platform for that execution
##
## The second file contains the characteristics of the graphs, and has the 
## following columns:
##
## graph: the name of the input graph
## kurtosis: the kurtosis of the node degree distribution for this graph
## alpha: the exponent of the power-law fitted on the empirial node degree 
##        distribution.
##
## This script also assumes that the column "algo" assume three values: BFS,
## Dijkstra, and PageRank. This is used to filter the data table.
##
## Date: 2011-04-21
## Author: Elizeu Santos-Neto


## Define a function that computes correlation between X and Y, and X and Z, and
## print the summaries of the respective linear regression models.
print_summary <- function(x, y, z) {
  # Correlations 
  cor(log(x), log(y)); 
  cor(log(x), log(z));

  # Linear regression models -- BFS
  model <- lm(x ~ y);
  summary(model);
  model <- lm(x ~ z);
  summary(model);
}

# Requirements
library("lattice");

# Get the a data file 
graph_file = commandArgs()[4];
dispersion_file = commandArgs()[5];

# Load data
graph_data <- read.table(graph_file, header = T);

# Agrregate
average_time <- aggregate(graph_data$edges / (graph_data$time * 10^-9), 
                          list(algo = graph_data$algo, 
                               platform = graph_data$platform, 
                               graph = graph_data$graph), mean);

# Load the file with the node degree distribution dispersion information
dispersion <- read.table(dispersion_file, header = T);

row_ids = (average_time[,1] == "PageRank" & average_time[,2] == "gpu");
pagerank_gpu = average_time[row_ids,];

row_ids = (average_time[,1] == "BFS" & average_time[,2] == "gpu");
bfs_gpu = average_time[row_ids,];

row_ids = (average_time[,1] == "Dijkstra" & average_time[,2] == "gpu");
dijkstra_gpu = average_time[row_ids,];

## Plot scatter plots.
pdf("scatter_pr_a.pdf");
plot(pagerank_gpu$x, dispersion$alph, log="xy");
dev.off();
pdf("scatter_pr_k.pdf");
plot(pagerank_gpu$x, dispersion$kurtosis, log="xy");
dev.off();

# Correlations -- BFS
print_summary(bfs_gpu$x, dispersion$alph, dispersion$kurtosis);

# Correlations -- Dijkstra
print_summary(dijkstra_gpu$x, dispersion$alph, dispersion$kurtosis);

# Correlations -- PageRank
print_summary(pagerank_gpu$x, dispersion$alph, dispersion$kurtosis);
