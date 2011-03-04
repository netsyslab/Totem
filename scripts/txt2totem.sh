#!/bin/bash
#
# This script receives as input a txt file that contains a list of
# edges. The file should contain three columns: <source node id>
# <destination node id> [weight], these columns are processed and
# converted into an edge list where the nodes are assumed to have
# contiguous IDs.
#
# Created on: 2011-03-02
# Author: Elizeu Santos-Neto (elizeus@ece.ubc.ca)
#

if [ $# -ne 2 ]; then
    echo "Usage: " $0 " <txt graph input> [-u]"
    exit 1
fi

# Input file
TXT=$1
if [ ! -f $TXT ]; then
    echo "Could not open the file: " $TXT
    exit 1
fi

# Make sure it is UNIX txt format.
dos2unix $TXT 1>2&> /dev/null

# Compute number of nodes/edges in the input graph and add a header.
echo "#Nodes: " `grep -v \# $TXT | awk '{ print $1; print $2; }' | sort -nu |\
   wc -l`
echo "#Edges: " `grep -v \# $TXT | wc -l`

# Set the graph direction according to the command line parameter.
if [ $2 == "-u" ]; then
    echo "#Undirected"
else 
    echo "#Directed"
fi 

# Current number of nodes (i.e., the next node id).
NNODES=0

# Change field separator
OLD_IFS=$IFS
IFS=$'\n'

# Declare an associative array to emmulate a hashmap.
declare -a hashmap 

## Produce the list of nodes in the graph
for EDGE in `grep -v \# $TXT`; do
  # Check whether the map contains the source node id
  SRC=`echo $EDGE | awk '{ print $1 }'`
  if [ -z "${hashmap[$SRC]}" ]; then
    hashmap[$SRC]=$NNODES
    NNODES=$((NNODES+1))
  fi

  # Check whether the map contains the source node id
  DST=`echo $EDGE | awk '{ print $2 }'`
  if [ -z "${hashmap[$DST]}" ]; then
    hashmap[$DST]=$NNODES
    NNODES=$((NNODES+1))
  fi
  WEIGHT=`echo $EDGE | awk '{ print $3 }'`
  echo ${hashmap[$SRC]} ${hashmap[$DST]} $WEIGHT
done | sort --key=1,1n --key=2n

# Reset the field separator
IFS=$OLD_IFS