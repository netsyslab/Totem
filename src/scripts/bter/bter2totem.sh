#!/bin/bash
#
# This script receives as input a txt file that contains a list of
# edges as produced by bter matlab implementation. The file should 
# have the following format:
# <source node id> <destination node id>  <connected>
# The file lists edges where the nodes are assumed to have
# contiguous IDs. The edges are undirected.
#
# Example:
# # Created by Octave 3.4.3
# # name: G
# # type: sparse matrix
# # nnz: 95225730
# # rows: 31639529
# # columns: 31639529
# 3 2 1
# 2 3 1
# 5 4 1
# 
# Created on: 2013-02-16
# Author: Abdullah Gharaibeh (abdullah@ece.ubc.ca)
#

if [ $# -gt 2 ]; then
    echo "Usage: " $0 " <txt graph input>"
    exit 1
fi

# Input file
BTER=$1
if [ ! -f $BTER ]; then
    echo "Could not open the file: " $BTER
    exit 1
fi

# Make sure it is UNIX txt format. 
dos2unix $BTER 1>&2 2> /dev/null

# Compute number of nodes/edges in the input graph and add a header.
NNODES=`head $BTER | grep rows | cut -d" " -f3`
# Deal with node id 0. Simply add node id 0 as an isolated node.
NNODES=$((NNODES+1))
NEDGES=`head $BTER | grep nnz | cut -d" " -f3`
echo "#Nodes:  $NNODES" > $BTER.totem
echo "#Edges:  $NEDGES" >> $BTER.totem
echo "#Undirected"      >> $BTER.totem

grep "^[[:digit:]]" $BTER | cut -d" " -f1,2 | \
    sort --key=1,1n --key=2n >> $BTER.totem
