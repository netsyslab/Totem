#!/bin/bash
#
# This script receives as input a txt file that contains a list of edges.
# The file should contain two columns: <source node id> <destination node id>,
# these columns are processed and converted into a GraphML format. The 
# format consists of an XML file with a list of nodes, an indication whether
# the graph is directed or not, and a list of edges.

if [ $# -ne 3 ]; then
    echo "Usage: " $0 " <txt graph input> <graph name> -<d|u>"
    exit 1
fi

## Print the header
echo "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
echo "<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\""  
echo "         xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\""
echo "         xsi:schemaLocation=\"http://graphml.graphdrawing.org/xmlns"
echo "         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd\">"

TXT=$1

NODE="<node id="
NL="/>"
EDGE="<edge"
SRC="source="
DST="target="
ENDGRAPH="</graph>"
ENDFILE="</graphml>"

if [ $3 == "-d" ]; then
    DIRECTION=directed
elif [ $3 == "-u" ]; then
    DIRECTION=undirected	
else 
    echo "Edge direction undefined. Assuming undirected graph." 1>&2
    DIRECTION=undirected
fi

echo "<graph id=\"$2\" edgedefault=\"$DIRECTION\">"

## Produce the list of nodes in the graph
for U in `grep -v \# $TXT | awk '{ print $1 "\n" $2 }' | sort -n -u`; do
    echo $NODE \"$U\" $NL
done
grep -v \# $TXT |\
  awk '{ print "<edge source=\"" $1 "\" target=\"" $2 "\"/>" }' 

echo $ENDGRAPH
echo $ENDFILE
