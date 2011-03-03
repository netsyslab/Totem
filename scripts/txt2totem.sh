#!/bin/bash
#
# This script receives as input a txt file that contains a list of
# edges. The file should contain three columns: <source node id>
# <destination node id> [weight], these columns are processed and
# converted into an edge list where the nodes are assumed to have
# contiguous IDs.

if [ $# -ne 1 ]; then
    echo "Usage: " $0 " <txt graph input>"
    exit 1
fi

# Input file
TXT=$1

# Current number of nodes (i.e., the last node added).
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
done

# Reset the field separator
IFS=$OLD_IFS