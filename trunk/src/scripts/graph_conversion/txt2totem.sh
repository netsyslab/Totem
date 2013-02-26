#!/bin/bash
#
# This script receives as input a txt file that contains a list of
# edges. The file should contain three columns separated by spaces:
# <source node id> <destination node id> [weight], these columns are 
# processed and converted into an edge list where the nodes are
# assumed to have contiguous IDs.
#
# Created on: 2011-03-02
# Author: Elizeu Santos-Neto (elizeus@ece.ubc.ca)
#         Lauro Beltrão Costa (lauroc@ece.ubc.ca)
#

if [ $# -gt 2 ]; then
    echo "Usage: " $0 " <txt graph input> [-u]"
    exit 1
fi

# Input file
TXT=$1
if [ ! -f $TXT ]; then
    echo "Could not open the file: " $TXT
    exit 1
fi

# Temp files' termination (to avoid name conflicts).
FTERM=$RANDOM

# Make sure it is UNIX txt format.
dos2unix $TXT 1>&2 2> /dev/null

# Declare an associative array to emmulate a map.
declare -a map 

# Current number of nodes (i.e., the next node id).
NNODES=0

# Populate the map (id from file --> id in a contiguous space).
for node in `grep -v \# $TXT | awk '{ print $1; print $2; }' | sort -nu`
do
  map[$node]=$NNODES
  NNODES=$((NNODES+1))
done

# Remove comments and translate tabs to spaces. 
grep -v \# $TXT | tr "\t" " " > $TXT.$FTERM.noheader

# Compute number of nodes/edges in the input graph and add a header.
echo "#Nodes:  $NNODES"
echo "#Edges: " `wc -l $TXT.$FTERM.noheader | cut -d" " -f1`

# Set the graph direction according to the command line parameter.
if [ "$#" -eq 2 -a "$2" == "-u" ]; then
    echo "#Undirected"
else 
    echo "#Directed"
fi 

# Break files' fields (source, destination and weight) into different files.
cut -s -d" " -f1 $TXT.$FTERM.noheader > SRC.$FTERM &
cut -s -d" " -f2 $TXT.$FTERM.noheader > DST.$FTERM &
cut -s -d" " -f3 $TXT.$FTERM.noheader > WEIGHT.$FTERM &
wait

# Maps nodes' ids from the original file to contiguous space.
for node in `cat SRC.$FTERM`
do
  echo ${map[$node]}
done > SRC.$FTERM.mapped &

for node in `cat DST.$FTERM`
do
  echo ${map[$node]}
done > DST.$FTERM.mapped &
wait

# Create output: put all fields together, remove spaces in the end of the lines
# (when there is no weight), and sort the edges.
paste -d" " SRC.$FTERM.mapped DST.$FTERM.mapped WEIGHT.$FTERM | \
  sed 's/ $//' | sort --key=1,1n --key=2n

# Remove temp files.
rm -f SRC.$FTERM DST.$FTERM SRC.$FTERM.mapped DST.$FTERM.mapped WEIGHT.$FTERM
