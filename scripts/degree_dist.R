## <add license>
##
## This script loads the node degrees of a graph, computes the power law
## coefficient of the empirical distribution, and plots the distribution.
## The input is a one column file with the node degree per line. Also,
## the script generates a plot with the distribution, so the second input
## paremeter is the title of the plot. 
##
## Requirements: igraph
##
## Date: 2011-04-02
## Author: Elizeu Santos-Neto

# Get the .degree filename from the command line
filename = commandArgs(T)[1];

# Get the title of the node degree distribution plot
main_title = commandArgs(T)[2];

# Load the igraph package
library("igraph");

# Loads the data and assumes that the node degree is in the
# first column.
data <- read.table(filename,head=FALSE);

# Fit the power law and gets the alpha.
a = power.law.fit(data$V1);
summary(a);

# Plot the rank-distribution.
imgfile <- paste(filename,"_plot.png", sep="");
png(filename=imgfile);
plot(sort(data$V1, decreasing=T), log="xy", ylab="Norm. Degree", xlab="Rank", 
     main=main_title);
dev.off()

# Plots the histogram.
imgfile <- paste(filename,"_hist.png", sep="");
png(filename=imgfile);
hist(log(data$V1), 100, ylab="# Nodes", xlab="Degree", main=main_title);
dev.off();
