#!/bin/bash

TOTEM=/home/elizeu/Dropbox/totem-graph/trunk/scripts/run_graphs.sh
unset OMP_NUM_THREADS

I=1; 
while [ $I -le 30 ]; do 
    $TOTEM _seq_gpu >> /local/data/experiments_seq_gpu.res
    echo Round $((I++)) ...
done

export OMP_NUM_THREADS=128
I=1
while [ $I -le 30 ]; do 
    $TOTEM _omp >> /local/data/experiments_omp.res
    echo Round omp $((I++)) ...
done