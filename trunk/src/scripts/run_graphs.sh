#!/bin/bash

# Command
TOTEM=/home/elizeu/Dropbox/totem-graph/trunk/src/totem/graph${1}

# Input graphs
DATA=/local/data/
GRAPHS=`ls $DATA/*.totem | grep -v single | grep -v complete |\
                           grep -v chain | grep -v CA`
for G in $GRAPHS; do
    echo $G
    $TOTEM -w -s 1 $G
done

