#!/bin/bash
#
# This script receives as input a txt file that contains a list of
# edges as produced by bter matlab implementation. The file should 
# have the following format:
# (<source node id>, <destination node id>) -> <connected>
# The file lists edges where the nodes are assumed to have
# contiguous IDs. The edges are undirected.
#
# Example:
# (3, 2) ->  1
# (2, 3) ->  1
# (5, 4) ->  1
# (4, 5) ->  1
# (7, 6) ->  1
# (6, 7) ->  1
# (9, 8) ->  1
# 
# TODO(lauro): Deal with node id 0. None bter uses it. To be simpler, this
# script simply adds node id 0 as an isolated node.
# 
# Created on: 2012-02-28
# Author: Lauro Beltrao Costa (lauroc@ece.ubc.ca)
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

# Temp files' termination (to avoid name conflicts).
FTERM=$RANDOM

# TODO(lauro): It can be faster.
# Remove special charactes keeping just numbers. Delete initial spaces. 
# Delete trailing spaces and number 1.
sed 's/[-(),>]//g' < $BTER | sed 's/^ *//g' | sed 's/ *[1] *$//g' \
  | sort --key=1,1n --key=2n > EDGES.$FTERM

NNODES=`tail -n 1 EDGES.$FTERM | cut -d" " -f1`
NNODES=$((NNODES+1))

# Compute number of nodes/edges in the input graph and add a header.
echo "#Nodes:  $NNODES"
echo "#Edges: " `wc -l EDGES.$FTERM| cut -d" " -f1`
echo "#Undirected"
cat EDGES.$FTERM

# Remove temp files.
rm -f EDGES.$FTERM
