#!/bin/bash
# 
# Produces the node degree for each node in a graph. The script receives a
# totem-format graph file as input. The output is a two column file with the
# first column representing the out-degree and the second column the vertex id.
# <out-degree> <vertex-id>
# 
# Author: Elizeu Santos-Neto 
# Date: 20 Apr 2011
#
grep -v \# $1 | awk '{print $1}' | uniq -c