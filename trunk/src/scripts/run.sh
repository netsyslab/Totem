#!/bin/bash
#
# This script makes it easier to run experiments for various configurations
#
# The script will produce an output file per configuration with the following
# format:
# 
# The first line describes the experiment configuration:
#  * file:<graph file>\tbenchmark:<BFS|PageRank|etc>\tvertices:<number>\t
#  * edges:<number>\tpartitioning:<RAND|HIGH|LOW>\tplatform:<CPU|GPU|HYBRID>\t
#  * alpha:<percentage of edges on the CPU>\trepeat:<number of runs>\t
#  * gpu_count:<number>\tthread_count:<CPU threads>\tthread_bind:<TRUE|FALSE>\t
#  * time_init:<Totem init time>\ttime_par:<Graph partitoining time>\t"
#  * rmt_vertex:<% of remote vertices>\trmt_edge:<% of remote edges>\t
#  * beta:<% of remote edges after aggregation>
#
# Multiple lines, each line details a run's timing breakdown. Note that the 
# first line is a header:
#  * total\texec\tinit\tcomp\tcomm\tfinalize\tgpu_comp\tscatter\tgather\taggr\t
#  * trv_edges\texec_rate
#
# where
#
# total: Total run time, including output buffer allocation
# exec: Execution time without buffer allocation (comp + comm + aggr)
# init: Algorithm initialization (mainly buffer allocations)
# comp: Compute phase
# comm: Communication phase (inlcudes scatter/gather)
# finalize: Algorithm finalization (buffer deallocations)
# gpu_comp: GPU computation (included in comp)
# scatter: The scatter step in communication (push mode)
# gather: The gather step in communication (pull mode)
# aggr: Final result aggregation
# trv_edges: Number of traversed edges
# exec_rate: Billion Traversed Edges Per Second (trv_edges/exec)
#
# The following is an example:
#
# * file:/ssd/graphs/bter_18_64.tbin	benchmark:BFS	vertices:63742081      \
# * edges:1011271202	partitioning:LOW	platform:HYBRID	alpha:5	       \
# * repeat:64\thread_count:12	thread_bind:true	time_init:55829.16     \
# * time_par:7383.63	rmt_vertex:46	rmt_edge:5	beta:3
# * total	exec	init	comp	comm	finalize	gpu_comp       \
# * scatter	gather	aggr	trv_edges	exec_rate
# * 1540.31	1309.85	27.92	950.76	220.93	3.70	950.24	206.09	0.00   \
# *	138.15	994293064	0.7591
# * 1515.95	1285.39	28.40	935.01	212.11	3.49	934.57	197.27	0.00   \
# * 138.25	994293064	0.7735
#
#
# Created on: 2013-02-15
# Author: Abdullah Gharaibeh

###########################################
# Display usage message and exit the script
###########################################
function usage() {
  echo "This script runs experiments for a specific workload and benchmark over"
  echo "a range of hardware and partitioning options."
  echo ""
  echo "Usage: $0 [options] <graph file>"
  echo "  -a  <minimum alpha> minimum value of alpha (the percentage of edges "
  echo "                      in the CPU partition) to use for experiments on"
  echo "                      hybrid platforms (default 5%)"
  echo "  -b  <benchmark> BFS=0, PageRank=1 (default BFS)"
  echo "  -g  <max gpu count> maximum number of GPUs to use (default 1)"
  echo "  -r  <results base directory> (default ../results)"
  echo "  -t  <totem executable> (default ../totem/totem)"
  echo "  -h  Print this usage message and exit"
}

###################
# Constants
###################
# The Benchmarks
BFS=0
PAGERANK=1
BENCHMARK_STR=("BFS" "PAGERANK")
BENCHMARK_REPEAT=(64 10)

# Platforms
CPU=0
GPU=1
HYBRID=2
PLATFORM_STR=("CPU" "GPU" "HYBRID")

# Graph partitioning algorithms
PAR_RAN="0"
PAR_HIGH="1"
PAR_LOW="2"
PAR_STR=("RAN" "HIGH" "LOW")

# Number of CPU sockets and threads
THREADS_PER_SOCKET=`cat /proc/cpuinfo | \
            grep "physical.*id.*:.*0" | wc -l`
MAX_THREAD_COUNT=`cat /proc/cpuinfo | \
            grep "physical.*id.*:.[[:digit:]]" | wc -l`
MAX_SOCKET_COUNT=$(($MAX_THREAD_COUNT / $THREADS_PER_SOCKET))

##########################
# Configuration Variables
##########################
MIN_ALPHA=5
MAX_ALPHA=95
BENCHMARK=${BFS}
RESULT_BASE="../results/"
TOTEM_EXE="../totem/totem"
MAX_GPU_COUNT=1

###############################
# Process command line options
###############################
while getopts 'a:b:g:hr:t:' options; do
  case $options in
    a)MIN_ALPHA="$OPTARG"
      ;;
    b)BENCHMARK="$OPTARG"
      ;;
    g)MAX_GPU_COUNT="$OPTARG"
      ;;
    h)usage; exit 0;
      ;;
    r)RESULT_BASE="$OPTARG"
      ;;
    t)TOTEM_EXE="$OPTARG"
      ;;
    ?)usage; exit -1;
      ;;
  esac
done
shift $(($OPTIND - 1))

if [ $# -ne 1 ]; then
    printf "\nError: Missing workload\n\n"
    usage
    exit -1
fi
if [ ! -f $1 ]; then
    printf "\nError: Workload $1 does not exist\n\n"
    exit -1
fi

##############################
# More configuration variables
##############################
# The workload to benchmark
WORKLOAD="$1"
WORKLOAD_NAME=`basename ${WORKLOAD} | awk -F. '{print $1}'`

# Results
RESULT_DIR=${RESULT_BASE}"/"${BENCHMARK_STR[$BENCHMARK]}"/"${WORKLOAD_NAME}
mkdir -p ${RESULT_DIR}

# Logs to report progress and failed runs
LOG=${RESULT_DIR}"/log"
LOG_FAILED_RUNS=${RESULT_DIR}"/logFailedRuns"

# Number of execution rounds (sources for traversal-based algorithms)
REPEAT_COUNT=${BENCHMARK_REPEAT[$BENCHMARK]}

# Configure OpenMP to bind OMP threads to specific hardware threads such that
# the first half is on socket one, and the other on socket two
# TODO(abdullah): detect the mapping automatically
export GOMP_CPU_AFFINITY="0-5 12-17 6-11 18-23"
export OMP_PROC_BIND=true

# Invokes totem executable for a specific set of parameters
function run() {
    local PLATFORM=$1;
    local SOCKET_COUNT=$2;
    local GPU_COUNT=$3;
    local PAR=$4;
    local ALPHA=$5;

    # Set the number of threads
    THREAD_COUNT=$(($SOCKET_COUNT*$THREADS_PER_SOCKET))
    if [ $PLATFORM -eq ${GPU} ]; then
        SOCKET_COUNT=0
        THREAD_COUNT=$MAX_THREAD_COUNT
    fi

    # Set the output file where the results will be dumped
    OUTPUT=${BENCHMARK_STR[$BENCHMARK]}_${SOCKET_COUNT}_${GPU_COUNT};
    OUTPUT=${OUTPUT}_${PAR_STR[$PAR]}_${ALPHA}_${WORKLOAD_NAME}.dat;
    DATE=`date`
    printf "${DATE}: ${OUTPUT} b${BENCHMARK} a${ALPHA} p${PLATFORM} " >> ${LOG};
    printf "i${PAR} g${GPU_COUNT} t${THREAD_COUNT} r${REPEAT_COUNT} " >> ${LOG};
    printf "${WORKLOAD}\n" >> ${LOG};
    ${TOTEM_EXE} -b${BENCHMARK} -a${ALPHA} -p${PLATFORM} -i${PAR} \
        -g${GPU_COUNT} -t${THREAD_COUNT} -r${REPEAT_COUNT} \
        ${WORKLOAD} &>> ${RESULT_DIR}/${OUTPUT}

    # Check the exit status, and log any problems
    exit_status=$?
    if [ $exit_status -ne 0 ]; then
        date &>> ${LOG_FAILED_RUNS}
        echo "${RESULT_DIR}/${OUTPUT}" &>> ${LOG_FAILED_RUNS}
        cat "${RESULT_DIR}/${OUTPUT}" &>> ${LOG_FAILED_RUNS}
        echo "" &>> ${LOG_FAILED_RUNS}
        rm "${RESULT_DIR}/${OUTPUT}"
    fi
}

## GPU Only, note that alpha has no effect when running only on GPU
alpha=0
gpu_count=1
run ${GPU} ${MAX_SOCKET_COUNT} ${gpu_count} ${PAR_RAN} ${alpha}
for gpu_count in $(seq 2 ${MAX_GPU_COUNT}); do
    for par_algo in ${PAR_RAN} ${PAR_HIGH} ${PAR_LOW}; do
        run ${GPU} ${MAX_SOCKET_COUNT} $gpu_count ${par_algo} ${alpha}
    done
done

## CPU Only, alpha and GPU count has no effect when running only on CPU 
alpha=0
gpu_count=0
for socket_count in $(seq 1 ${MAX_SOCKET_COUNT}); do
    run ${CPU} $socket_count ${gpu_count} ${PAR_RAN} ${alpha}
done

## Hybrid, iterate over all possible number of GPUs, CPUs and values of alpha
for gpu_count in $(seq 1 ${MAX_GPU_COUNT}); do
    for socket_count in $(seq 1 ${MAX_SOCKET_COUNT}); do
        for par_algo in ${PAR_LOW} ${PAR_HIGH} ${PAR_RAN}; do
            for alpha in $(seq ${MIN_ALPHA} 5 ${MAX_ALPHA}); do
                run ${HYBRID} ${socket_count} ${gpu_count} ${par_algo} $alpha
            done
       done
    done
done

echo "done!" >> ${LOG}
