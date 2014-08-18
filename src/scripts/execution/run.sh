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
#  * time_init:<Totem init time>\ttime_par:<Graph partitioning time>\t"
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
# comm: Communication phase (includes scatter/gather)
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
# Authors: Abdullah Gharaibeh
#          Scott Sallinen

###################
# Constants
###################
# The Benchmarks
BFS=0
PAGERANK=1
SSSP=2
BC=3
GRAPH500=4
BENCHMARK_STR=("BFS" "PAGERANK" "SSSP" "BC" "GRAPH500")
BENCHMARK_REPEAT=(64 10 64 20 10)

# Platforms
CPU=0
GPU=1
HYBRID=2
PLATFORM_STR=("CPU" "GPU" "HYBRID")

# Graph partitioning algorithms
PAR_RAN="0"
PAR_HIGH="1"
PAR_LOW="2"
PAR_STR=("RAND" "HIGH" "LOW")

# OpenMP thread scheduling algorithms
OMP_SCHED_STATIC="1"
OMP_SCHED_DYNAMIC="2"
OMP_SCHED_GUIDED="3"
OMP_SCHED_STR=("" "STATIC" "DYNAMIC" "GUIDED")

# Number of CPU sockets and threads
THREADS_PER_SOCKET=`cat /proc/cpuinfo | \
            grep "physical.*id.*:.*0" | wc -l`
MAX_THREAD_COUNT=`cat /proc/cpuinfo | \
            grep "physical.*id.*:.[[:digit:]]" | wc -l`
MAX_SOCKET_COUNT=$(($MAX_THREAD_COUNT / $THREADS_PER_SOCKET))

##########################
# Default Configuration
##########################
MIN_ALPHA=5
MAX_ALPHA=95
BENCHMARK=${BFS}
RESULT_BASE="../../results"
TOTEM_EXE="../../build/bin/benchmark"
MAX_GPU_COUNT=1
REPEAT_COUNT=
OMP_SCHED=${OMP_SCHED_GUIDED}
GPU_PAR_RAND=
COMPARE_MEMORY=()
COMPARE_SORTED=()


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
  echo "                      hybrid platforms (default ${MIN_ALPHA}%)"
  echo "  -b  <benchmark> BFS=${BFS}, PageRank=${PAGERANK}, SSSP=${SSSP}," \
       "BC=${BC}, GRAPH500=${GRAPH500}" \
       "(default ${BENCHMARK_STR[${BENCHMARK}]})"
  echo "  -d  <results base directory> (default ${RESULT_BASE})"
  echo "  -e  <totem executable> (default ${TOTEM_EXE})"
  echo "  -g  <max gpu count> maximum number of GPUs to use(default " \
       "${MAX_GPU_COUNT})"
  echo "  -m <type of memory> Enables allocating the vertex|edge array of the "
  echo "     GPU partitions as a memory mapped buffer on the host. Add "
  echo "     multiple -m flags to compare. If none are set, will default to "
  echo "     only device (0)."
  echo "  -o Enables random placement of vertices across GPU partitions in case"
  echo "     of multi-GPU setups (default FALSE)"
  echo "  -p Compare original and sorted versions of the graph. (default FALSE)"
  echo "  -q Sort the graph's vertices by degree. If enabled, will override "
  echo "     the -p flag and only run with the sorted option. (default FALSE)"
  echo "  -r  <repeat count> number of times an experiment is repeated"
  echo "                     (default BFS:${BENCHMARK_REPEAT[$BFS]}," \
       "PageRank:${BENCHMARK_REPEAT[$PAGERANK]})"
  echo "  -s  <OMP scheduling>" \
       "${OMP_SCHED_STR[${OMP_SCHED_STATIC}]}=${OMP_SCHED_STATIC}," \
       "${OMP_SCHED_STR[${OMP_SCHED_DYNAMIC}]}=${OMP_SCHED_DYNAMIC}," \
       "${OMP_SCHED_STR[${OMP_SCHED_GUIDED}]}=${OMP_SCHED_GUIDED}" \
       "(default ${OMP_SCHED_STR[${OMP_SCHED}]})"
  echo "  -x  <maximum alpha> maximum value of alpha (the percentage of edges "
  echo "                      in the CPU partition) to use for experiments on"
  echo "                      hybrid platforms (default ${MAX_ALPHA}%)"
  echo "  -h  Print this usage message and exit"
}


###############################
# Process command line options
###############################
while getopts 'a:b:d:e:g:hm:opqr:s:x:' options; do
  case $options in
    a)MIN_ALPHA="$OPTARG"
      ;;
    b)BENCHMARK="$OPTARG"
      ;;
    d)RESULT_BASE="$OPTARG"
      ;;
    e)TOTEM_EXE="$OPTARG"
      ;;
    g)MAX_GPU_COUNT="$OPTARG"
      ;;
    h)usage; exit 0;
      ;;
    m)COMPARE_MEMORY+=("$OPTARG")
      ;;
    o)GPU_PAR_RAND="-o"
      ;;
    p)COMPARE_SORTED=( 'true' 'false' )
      ;;
    q)COMPARE_SORTED=( 'true' )
      ;;
    r)REPEAT_COUNT="$OPTARG"
      ;;
    s)OMP_SCHED="$OPTARG"
      ;;
    x)MAX_ALPHA="$OPTARG"
      ;;
    ?)usage; exit -1;
      ;;
  esac
done
shift $(($OPTIND - 1))

# Add defaults if missing.
if [ ${#COMPARE_MEMORY[@]} -eq 0 ]; then
    COMPARE_MEMORY+=( 0 )
fi
if [ ${#COMPARE_SORTED[@]} -eq 0 ]; then
    COMPARE_SORTED+=( 'false' )
fi

# Check for errors.
if [ $# -ne 1 ]; then
    printf "Error: Missing workload\n"
    usage
    exit -1
fi
if [ ! -f $1 ]; then
    printf "Error: Workload \"$1\" does not exist\n"
    exit -1
fi
if [ ! -f $TOTEM_EXE ]; then
    printf "Error: The benchmark tool \"$TOTEM_EXE\" does not exist.\n"
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

# Get the default number of execution rounds if not already specified
# by command line
if [ ! $REPEAT_COUNT ]; then
    REPEAT_COUNT=${BENCHMARK_REPEAT[$BENCHMARK]}
fi

# Configure OpenMP to bind OMP threads to specific hardware threads such that
# the first half is on socket one, and the other on socket two
export GOMP_CPU_AFFINITY=`cat /proc/cpuinfo | grep "physical id" | \
    cut -d" " -f 3 | uniq -c | \
    awk 'BEGIN{cur = 0}{printf("%d %d-%d\n", $2, cur, (cur + $1 - 1)); \
    cur = cur + $1}' | sort -k1n -k2n | awk '{p = p " " $2}END{print p}'`
export OMP_PROC_BIND=true

# Invokes totem executable for a specific set of parameters
function run() {
    local PLATFORM=$1;
    local SOCKET_COUNT=$2;
    local GPU_COUNT=$3;
    local PAR=$4;
    local ALPHA=$5;
    local MEMORY=$6;
    local SORTED=$7;

    # Set the number of threads
    THREAD_COUNT=$(($SOCKET_COUNT*$THREADS_PER_SOCKET))
    if [ $PLATFORM -eq ${GPU} ]; then
        SOCKET_COUNT=0
        THREAD_COUNT=$MAX_THREAD_COUNT
    fi

    # Set the output file where the results will be dumped.
    OUTPUT=${BENCHMARK_STR[$BENCHMARK]}_${SOCKET_COUNT}_${GPU_COUNT}_${SORTED};
    OUTPUT+="_MEM"${MEMORY}_${PAR_STR[$PAR]}_${ALPHA}_${WORKLOAD_NAME}.dat;
    
    # Build the header for the log.
    DATE=`date`
    printf "${DATE}: ${OUTPUT} b${BENCHMARK} a${ALPHA} p${PLATFORM} " >> ${LOG};
    printf "i${PAR} g${GPU_COUNT} t${THREAD_COUNT} r${REPEAT_COUNT} " >> ${LOG};
    printf "s${OMP_SCHED} m${MEMORY} ${GPU_PAR_RAND} q${SORTED} "     >> ${LOG};
    printf "${WORKLOAD}\n" >> ${LOG};
    
    # Set up the configuration flags for the run.
    FLAGS="-b${BENCHMARK} -a${ALPHA} -p${PLATFORM} -i${PAR} \
           -g${GPU_COUNT} -t${THREAD_COUNT} -r${REPEAT_COUNT} -s${OMP_SCHED} \
           -m${MEMORY} ${GPU_PAR_RAND}"
    if [ "$SORTED" == true ]; then
      FLAGS+=" -q"
    fi
    
    # Start executing.
    ${TOTEM_EXE} ${FLAGS} ${WORKLOAD} &>> ${RESULT_DIR}/${OUTPUT}
        
    # Check the exit status, and log any problems
    exit_status=$?
    if [ ${exit_status} -ne 0 ]; then
        date &>> ${LOG_FAILED_RUNS}
        echo "${RESULT_DIR}/${OUTPUT}" &>> ${LOG_FAILED_RUNS}
        cat "${RESULT_DIR}/${OUTPUT}" &>> ${LOG_FAILED_RUNS}
        echo "" &>> ${LOG_FAILED_RUNS}
        rm "${RESULT_DIR}/${OUTPUT}"
    fi
}

## CPU Only, alpha and GPU count has no effect when running only on CPU
alpha=100
gpu_count=0
memory=0
for socket_count in $(seq 1 ${MAX_SOCKET_COUNT}); do
    for sorted in "${COMPARE_SORTED[@]}"; do
        # We can use "Random partitioning" avoid sorting the vertices. 
        # In addition, we can use "High" to sort the vertices in ascending
        # order - this decision over descending order is arbitrary and does
        # not seem to have an effect on the execution rate.
        if [ "$sorted" == false ]; then
            run ${CPU} ${socket_count} ${gpu_count} ${PAR_RAN} \
                ${alpha} ${memory} ${sorted}
        else
            run ${CPU} ${socket_count} ${gpu_count} ${PAR_HIGH} \
                ${alpha} ${memory} ${sorted}
        fi
    done
done

## Hybrid, iterate over all possible number of GPUs, CPUs and values of alpha
for gpu_count in $(seq 1 ${MAX_GPU_COUNT}); do
    for socket_count in $(seq 1 ${MAX_SOCKET_COUNT}); do
        for par_algo in ${PAR_LOW} ${PAR_HIGH} ${PAR_RAND}; do
            for alpha in $(seq ${MIN_ALPHA} 5 ${MAX_ALPHA}); do
                for memory in "${COMPARE_MEMORY[@]}"; do
                    for sorted in "${COMPARE_SORTED[@]}"; do
                        run ${HYBRID} ${socket_count} ${gpu_count} ${par_algo} \
                            ${alpha} ${memory} ${sorted}
                    done
                done
            done
        done
    done
done

## GPU Only, note that alpha has no effect when running only on GPU
if [ ${MAX_GPU_COUNT} -ge 1 ]; then
    alpha=0
    for gpu_count in $(seq 1 ${MAX_GPU_COUNT}); do
        for par_algo in ${PAR_RAN} ${PAR_HIGH} ${PAR_LOW}; do
            for memory in "${COMPARE_MEMORY[@]}"; do
                for sorted in "${COMPARE_SORTED[@]}"; do
                    run ${GPU} ${MAX_SOCKET_COUNT} ${gpu_count} ${par_algo} \
                        ${alpha} ${memory} ${sorted}
                done
            done
        done
    done
fi

echo "done!" >> ${LOG}
