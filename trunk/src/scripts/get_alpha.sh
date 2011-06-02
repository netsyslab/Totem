#!/bin/bash
#
# Invokes the R script that computes the node degree distribution and the alpha 
# coefficient. 
#
# Created on: 2011-04-02
# Author: Elizeu Santos-Neto (elizeus@ece.ubc.ca)

INPUT_FILE=$1
TITLE=`echo $INPUT_FILE | cut -d- -f2 | cut -d. -f1`;
R --silent --no-save --args $INPUT_FILE $TITLE < degree_dist.R | grep alpha |\
  awk '{print $2}';
