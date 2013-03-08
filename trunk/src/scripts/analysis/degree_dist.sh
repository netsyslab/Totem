#!/bin/bash
# 
# Produces the node degree distribution of a graph. The script receives a
# totem-format graph file as input. The output is a two column file with the
# first column representing the out-degree and the second the number of nodes.
# <out-degree> <# of nodes> 
# 
# Author: Elizeu Santos-Neto 
# Date: 20 Apr 2011
#

if [ $# -ne 1 ]; then
    echo "Error: Missing Totem graph file"
    echo "Usage: $0 <graph file>"
    exit -1
fi

TOTEM_FILE=$1
DEGREE_FILE=${TOTEM_FILE}.degree

echo "degree nnodes" > ${DEGREE_FILE}
grep -v \# $1 | awk '{print $1}' | uniq -c | awk '{print $1}' | \
    sort -k1n | uniq -c | awk '{print $2,$1}' >> ${DEGREE_FILE}
