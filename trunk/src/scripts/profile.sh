#!/bin/bash
## This script aggregates GPU performance counters. In addition to logging raw 
## numbers produced by the profiler, the script spells out user-friendly 
## derived statistics per kernel call. The script is largely agnostic to 
## totem, but has one display option that prints the vwarp and standard kernels
## side by side to enable easy comparison.
## The script prints out the following statistics (descriptions of counters
## presented thoughput the scripts are largely copied from CUDA Profiled User 
## Guide, please refer to the guide for detailed documentation):
##  1: kernel name
##  2: gputime: Execution time for the method on the GPU
##  3: occupancy: The ratio of the number of active warps per multiprocessor to 
##     the maximum number of active warps. This can be determined at compile 
##     time. It is the maximum occupancy that can be achieved by the kernel.
##  4: achieved occupancy (a-occ): the actual occupancy of the kernel based on
##     the number of warps executing per cycle on the multiprocessor.
##  5: global memory excess load (geld): the percentage of excess data that is 
##     fetched while making global memory load accesses
##  6: L2 load throughput (l2ld): The throughput achieved while reading from
##     L2 cache when a request for data residing in global memory is made by L1.
##  7: instruction/byte (IpB): The ratio of the total number of instructions
##     issued by the kernel and the total number of bytes accessed by the kernel
##     from global memory.
##  8: divergent branches (db): The percentage of branches that are causing 
##     divergence within a warp amongst all the branches present in the kernel.
##  9: control flow divergant (cfd): This gives the percentage of thread 
##     instructions that were not executed by all threads in the warp, hence 
##     causing divergence.
##
##  Created on: 2011-08-11
##  Author: Abdullah Gharaibeh

################################################################################
# Set environment variables needed by cudaprofile. Note that the profiler is not
# explicitly invoked. If the environment variable COMPUTE_PROFILE is set to 1, 
# applications that make cuda calls (e.g., a running totem executable) will be 
# profiled. The following variables controls the behavior of the profiler.
###############################################################################
# Enable profiling.
export COMPUTE_PROFILE=1 
# Generate profiler output in CVS format.
export COMPUTE_PROFILE_CSV=1 
# Profiler output file.
export COMPUTE_PROFILE_LOG="/local/data/profile.log" 
# Profiler configuration. Performance counters to be collected are specified in 
# this file one line per counter (check the user guide for a comprehensive list
# of counters names and descriptions).
export COMPUTE_PROFILE_CONFIG="/local/data/cudaprofiler.config" 


###########################
# Script's global variables
###########################
# The executable to be profiled. This can be changed via command line options.
EXE="../totem/totem"

# The executable's output log. This can be changed via command line options.
EXE_LOG="/dev/null"

# The directory where profile logs will be placed. This can be changed via
# command line options. Having this as an option enable users to organize 
# profile logs.
LOG_DIR="/local/data/log"

# The following are filters to filter the output displayed on the screen
FILTER_MASTER="grep _kernel" # A master filter applied to all displayed output.
                             # This can be changed via command line options
FILTER_ST_KERNELS="grep -v vwarp" # Filters vwarp kernels
FILTER_VW_KERNELS="grep vwarp" # Filters all standard (i.e., non-vwarp) kernels

# This is used as a prefix to log files' names. This enables users to organize
# logs under the same log directory (can be changed via command line options).
LOG_PREFIX=""

# If this is set to NO, then the script will not profile, rather it will look
# for existing log files under the log directory and just display the output
EXE_ENABLED="YES"

# The following are GPU specific parameters
# TODO (Abdullah): This should be queried dynamically from the GPU
GPU_SM_NUM=15 # Number of streaming multiprocessors the GPU has
GPU_WARP_WIDTH=32 # Warp width
GPU_MAX_WARP=48 # Maximum number of resident warps per multiprocessor

# A regular expression used in awk to identify kernel call records in profile 
# logs to do kernel-specific counters calculations. Note that, in addition to
# kernel calls records, the profiler output logs contain records for other cuda 
# calls (such as memory transfer calls), hence the need for such a filter to 
# identify kernel calls only.
# TODO (Abdullah): This expression is based only on observing the output of 
# the profile logs, and is not documented in CUDA documentations, hence it is
# something to keep an eye on.
KERNEL_REGEXP="^_Z[0-9]*"

# If this is set to YES, then the virtual warp and standard kernels are printed
# side by side. This is possible only if the executable runs both versions.
# This can be changed via command line options.
PRINT_SIDE_BY_SIDE="NO"

# File where the script's user-friendly output will be printed. This can be 
# changed via command line options.
OUTPUT_FILE="/dev/stdout"


###########################################
# Display usage message and exit the script
###########################################
function usage() {
  echo "This script aggregates GPU performance counters. The script is largely" 
  echo "agnostic to totem, but has one display option that prints the vwarp and"
  echo "standard kernels side by side to enable easy comparison."
  echo ""
  echo "Usage: $0 [options] <graph file>"
  echo "  -d  <log directory> Profile logs directory (default /local/data/log)"
  echo "  -e  <cuda executable> Executable to profile (default ../totem/totem)"
  echo "  -h  Print this usage message and exit"
  echo "  -l  <log file> Executable's output log file (default /dev/null)"
  echo "  -n  Disable profiling, and use existing logs to produce output"
  echo "  -o  <output file> Print output to a file (default stdout)"
  echo "  -p  <prefix> Prefix for log files names"
  echo "  -r  <regexp> A grep regexp to filter output (e.g., 'sum_.*' to"
  echo "               display sum_rank kernel calls for PageRank)"
  echo "  -w  Print virtual warp and standard kernels counters side by side"
  echo "      (totem specific option)"
  exit 1
}


###############################
# Process command line options
###############################
while getopts 'd:e:hl:no:p:r:w' options; do
  case $options in
    d)LOG_DIR="$OPTARG"
      ;;
    e)EXE="$OPTARG"
      ;;
    h)usage
      ;;
    l)EXE_LOG="$OPTARG"
      ;;
    n)EXE_ENABLED="NO"
      ;;
    o)OUTPUT_FILE="$OPTARG"
      ;;
    p)LOG_PREFIX="$OPTARG""_"
      ;;
    r)FILTER_MASTER="grep $OPTARG"
      ;;
    w)PRINT_SIDE_BY_SIDE="YES"
      ;;
    ?)usage
      ;;
  esac
done
shift $(($OPTIND - 1))

# One parameter is expected: the command line options string for the executable.
if [ $# -ne 1 -a "$EXE_ENABLED" = "YES" ]; then
  usage
fi

# Attach the executable's command line options with the executable command.
EXE="$EXE $1"

# Make sure the log folder exists, if not create it.
if [ ! -d $LOG_DIR ]; then 
  mkdir $LOG_DIR
fi

# The following are the log files for the profiler for various 
# performance counters.
# TODO(Abdullah): Add details
TIME_LOG="$LOG_DIR/$LOG_PREFIX""time.log"
L2LD_LOG="$LOG_DIR/$LOG_PREFIX""l2ld.log"
L2ST_LOG="$LOG_DIR/$LOG_PREFIX""l2st.log"
GLD_LOG="$LOG_DIR/$LOG_PREFIX""gld.log"
GST_LOG="$LOG_DIR/$LOG_PREFIX""gst.log"
GM_EXCESS_LOAD_LOG="$LOG_DIR/$LOG_PREFIX""geld.log"
GM_EXCESS_STORE_LOG="$LOG_DIR/$LOG_PREFIX""gest.log"
ACW_LOG="$LOG_DIR/$LOG_PREFIX""acw.log"
INST_ISSD_LOG="$LOG_DIR/$LOG_PREFIX""inst_issd.log"
INST_EXEC_LOG="$LOG_DIR/$LOG_PREFIX""inst_exec.log"
THRDS_INST_EXEC_LOG="$LOG_DIR/$LOG_PREFIX""thrds_inst_exec.log"
INST_PER_BYTE_LOG="$LOG_DIR/$LOG_PREFIX""ipb.log"
DB_LOG="$LOG_DIR/$LOG_PREFIX""db.log"
CONTROL_FLOW_DIV="$LOG_DIR/$LOG_PREFIX""cfd.log"
BANK_CONFLICTS="$LOG_DIR/$LOG_PREFIX""bc.log"


## The log_* functions profile various performance counters. This is done by
## specifying the counters to be profiled in the COMPUTE_PROFILE_CONFIG config
## file (one line per counter) and running the executable. The performance 
## counters names are available in the "Nvidia compute visual profiler user
## guide". The resulting profile log produced by the profiler is always at 
## COMPUTE_PROFILE_LOG.
## The log_* functions copy this file after each profile session to the log 
## folder under a name relevant to the performance counter profiled.
## The log file produced by cudaprofile will have a record for each kernel
## call. The record will have at least four fields: kernel name, gputime, 
## cputime and occupancy.
## In addition to the above mentioned four fields, the log will have a field
## for each performance counter specified in the COMPUTE_PROFILE_CONFIG 
## config file, hence the first performance counter specified in the config
## file will be at field 5, the second at 6 and so on.
## Note that due to limited hardware capability, only few counters (sometimes 
## only one) can be profiled at a time, therefore the executable will be run 
## several times to collect statistics for the required counters.
## Finally, the derive_* functions produce derived statistics from the
## performance counters recorded by the log_* functions.


###########################################################################
# Log default parameters with not performance counters. This is used to get
# an estimation of gputime and cputtime with minimum profiling overhead. It 
# also logs the maximum theoritical occupancy of the kernels. The resulting
# log file has four columns in the following order:   
# 1: kernel name
# 2: gputime (time spent executing the kernel on the GPU)
# 3: cputime (CPU time elapsed launching the kernel)
# 4: occupancy (percentage of maximum number of warps the kernel could have
#               available to execute at each SM. This depends on the amount
#               of SM local resources used by each thread in a thread block
#               such as registers and shared memory)
###########################################################################
function log_time() {
  local OUTPUT=$TIME_LOG
  printf "Profiling default parameters... "

  echo "" > $COMPUTE_PROFILE_CONFIG
  $EXE >> $EXE_LOG
  mv $COMPUTE_PROFILE_LOG $OUTPUT.RAW

  cat $TIME_LOG.RAW | awk -F, '{print $1, $2, $3, $4*100}' > $OUTPUT
  printf "done, log at $OUTPUT\n"
}

###########################################################################
# Log number of L2 read requests. This is a derived performance 
# counter. It is the sum of two performance counters. The resulting log file 
# has one column:
# 1: l2 load requests in bytes (l2ld)
###########################################################################
function log_l2_read_requests() {
  local OUTPUT=$L2LD_LOG
  printf "Profiling L2 read requests... "
  
  # Profile: get accumulated read sector queries from L1 to L2 cache for 
  # slices 0 and of all the L2 cache units.
  echo "l2_subp0_read_sector_queries" > $COMPUTE_PROFILE_CONFIG
  $EXE >> $EXE_LOG
  mv $COMPUTE_PROFILE_LOG $OUTPUT.SUBP0.RAW
  echo "l2_subp1_read_sector_queries" > $COMPUTE_PROFILE_CONFIG
  $EXE >> $EXE_LOG
  mv $COMPUTE_PROFILE_LOG $OUTPUT.SUBP1.RAW
  
  # The accumulated number from the counters is of 32 bytes granularity. Produce
  # a more usable final result by adding the counters and convert to bytes.
  paste -d"," $OUTPUT.SUBP1.RAW $OUTPUT.SUBP0.RAW | \
    awk -vkernel_regexp="$KERNEL_REGEXP" -F, \
    '{if (!match($1, kernel_regexp)) print "nan" 
      else print ($5 * 32) + ($10 * 32)}' > $OUTPUT
  printf "done, log at $OUTPUT\n"
}

###########################################################################
# Log number of L2 write requests in bytes. This is a derived performance 
# counter. It is the sum of two performance counters. The resulting log file 
# has one column:
# 1: l2 store requests in bytes (l2st)
###########################################################################
function log_l2_write_requests() {
  local OUTPUT=$L2ST_LOG
  printf "Profiling L2 write requests... "
  
  # Profile: get accumulated write sector queries from L1 to L2 cache for 
  # slices 0 and of all the L2 cache units.
  echo "l2_subp0_write_sector_queries" > $COMPUTE_PROFILE_CONFIG
  $EXE >> $EXE_LOG
  mv $COMPUTE_PROFILE_LOG $OUTPUT.SUBP0.RAW
  echo "l2_subp1_write_sector_queries" > $COMPUTE_PROFILE_CONFIG
  $EXE >> $EXE_LOG
  mv $COMPUTE_PROFILE_LOG $OUTPUT.SUBP1.RAW
  
  # The accumulated number from the counters is of 32 bytes granularity. Produce
  # a more usable final result by adding the counters and convert to bytes.
  paste -d"," $OUTPUT.SUBP1.RAW $OUTPUT.SUBP0.RAW | \
    awk -vkernel_regexp="$KERNEL_REGEXP" -F, \
    '{if (!match($1, kernel_regexp)) print "nan" 
      else print ($5 * 32) + ($10 * 32)}' > $OUTPUT
  printf "done, log at $OUTPUT\n"
}

###########################################################################
# Log number of global memory load requests in bytes. This is a derived 
# performance counter. It is the sum of five performance counters. The resulting
# log file has one column:
# 1: global memory load requests in bytes (gld)
###########################################################################
function log_global_memory_read_requests() {
  local OUTPUT=$GLD_LOG
  printf "Profiling global_memory_read_requests... "
  
  # Profile: number of 1, 2, 4, 8 and 16 byte load requests
  echo "gld_inst_8bit"   >  $COMPUTE_PROFILE_CONFIG
  echo "gld_inst_16bit"  >> $COMPUTE_PROFILE_CONFIG
  echo "gld_inst_32bit"  >> $COMPUTE_PROFILE_CONFIG
  echo "gld_inst_64bit"  >> $COMPUTE_PROFILE_CONFIG
  echo "gld_inst_128bit" >> $COMPUTE_PROFILE_CONFIG
  $EXE >> $EXE_LOG
  mv $COMPUTE_PROFILE_LOG $OUTPUT.RAW

  # Prodcue the final result by adding together all the registers above
  # 1: global loads requested (gld)
  awk -vkernel_regexp="$KERNEL_REGEXP" -F, \
    '{if (!match($1, kernel_regexp)) print "nan" 
      else { b8 = $5; b16 = $6; b32 = $7; b64 = $8; b128 = $9;
      tot = b8 + b16*2 + b32*4 + b64*8 + b128*16;
      print tot }}' $OUTPUT.RAW > $OUTPUT
  printf "done, log at $OUTPUT\n"
}

###########################################################################
# Log number of global memory store requests in bytes. This is a derived 
# performance counter. It is the sum of five performance counters. The resulting
# log file has one column:
# 1: global memory store requests in bytes (gst)
###########################################################################
function log_global_memory_write_requests() {
  local OUTPUT=$GST_LOG
  printf "Profiling global_memory_write_requests... "
  
  # Profile: number of 1, 2, 4, 8 and 16 byte load requests
  echo "gst_inst_8bit"   >  $COMPUTE_PROFILE_CONFIG
  echo "gst_inst_16bit"  >> $COMPUTE_PROFILE_CONFIG
  echo "gst_inst_32bit"  >> $COMPUTE_PROFILE_CONFIG
  echo "gst_inst_64bit"  >> $COMPUTE_PROFILE_CONFIG
  echo "gst_inst_128bit" >> $COMPUTE_PROFILE_CONFIG
  $EXE >> $EXE_LOG
  mv $COMPUTE_PROFILE_LOG $OUTPUT.RAW

  # Prodcue the final result by adding together all the registers above
  awk -vkernel_regexp="$KERNEL_REGEXP" -F, \
    '{if (!match($1, kernel_regexp)) print "nan" 
      else { b8 = $5; b16 = $6; b32 = $7; b64 = $8; b128 = $9;
      tot = b8 + b16*2 + b32*4 + b64*8 + b128*16;
      print tot }}' $OUTPUT.RAW > $OUTPUT
  printf "done, log at $OUTPUT\n"
}


###########################################################################
# Log number the number of instructions issued. This counter is incremented once
# per warp, hence it is multiplisd by the warp width to get the total number of
# instructions issued for all threads. The log file has one column:
# 1: total number of instructions issued (inst_issd)
###########################################################################
function log_instructions_issued() {
  local OUTPUT=$INST_ISSD_LOG
  printf "Profiling instructions_issued... "

  echo "inst_issued"   >  $COMPUTE_PROFILE_CONFIG
  $EXE >> $EXE_LOG
  mv $COMPUTE_PROFILE_LOG $OUTPUT.RAW

  awk -vkernel_regexp="$KERNEL_REGEXP" -vwarp_width="$GPU_WARP_WIDTH" -F, \
    '{if (!match($1, kernel_regexp)) print "nan" 
      else print $5 * warp_width}' $OUTPUT.RAW > $OUTPUT
  printf "done, log at $OUTPUT\n"
}

###########################################################################
# Log number the number of instructions executed. This counter is incremented 
# once per warp, hence it is multiplisd by the warp width to get the total 
# number of instructions issued for all threads. The log file has one column:
# 1: total number of instructions executed (inst_exec)
###########################################################################
function log_instructions_executed() {
  local OUTPUT=$INST_EXEC_LOG
  printf "Profiling instructions executed... "

  echo "inst_executed" > $COMPUTE_PROFILE_CONFIG
  $EXE >> $EXE_LOG
  mv $COMPUTE_PROFILE_LOG $OUTPUT.RAW

  awk -vkernel_regexp="$KERNEL_REGEXP" -vwarp_width="$GPU_WARP_WIDTH" -F, \
    '{if (!match($1, kernel_regexp)) print "nan" 
      else print $5 * warp_width}' $OUTPUT.RAW > $OUTPUT
  printf "done, log at $OUTPUT\n"
}

###########################################################################
# Log the number of instructions executed by all threads. This does not include 
# replays. For each instruction it increments by the number of threads in the 
# warp that execute the instruction. The log file has one column:
# 1: total number of threads instructions executed (thrds_inst_exec)
###########################################################################
function log_threads_instructions_executed() {
  local OUTPUT=$THRDS_INST_EXEC_LOG
  printf "Profiling number of instructions executed by all threads... "

  echo "thread_inst_executed_0"   >   $COMPUTE_PROFILE_CONFIG
  $EXE >> $EXE_LOG
  mv $COMPUTE_PROFILE_LOG $OUTPUT.0.RAW
  echo "thread_inst_executed_1"   >   $COMPUTE_PROFILE_CONFIG
  $EXE >> $EXE_LOG
  mv $COMPUTE_PROFILE_LOG $OUTPUT.1.RAW

  paste -d", " $OUTPUT.0.RAW $OUTPUT.1.RAW | \
    awk -vkernel_regexp="$KERNEL_REGEXP" -F, \
    '{if (!match($1, kernel_regexp)) print "nan" 
      else print $5 + $10 +$15 + $20}' > $OUTPUT
  printf "done, log at $OUTPUT\n"
}

###########################################################################
# Log achieved kernel occupancy. This performance counter is derived from two 
# performance couters: active cycles and active warps. The active cycles counter
#  records the number of cycles a multiprocessor has at least one active warp. 
# The active warps counter accumulates number of active warps per cycle. To this
# end, the achieved kernel occupancy is a ratio that is based on the number of 
# active warps per cycle on the SM. It is the ratio of active warps and active 
# cycles divided by the max number of warps that can execute on SM. This is 
# calculated as: (active warps/active cycles)/GPU_MAX_WARP
# The log file has two columns:
# 1: Average achieved occupancy (a-occ)
###########################################################################
function log_active_cycles_and_warps() {
  local OUTPUT=$ACW_LOG
  printf "Profiling active cycles and warps... "
  
  # Profile two performance counters
  # Active cycles - number of cycles a multiprocessor has at least one 
  # active warp
  echo "active_cycles" > $COMPUTE_PROFILE_CONFIG
  $EXE >> $EXE_LOG
  mv $COMPUTE_PROFILE_LOG $OUTPUT.CYCLES.RAW

  # Active warps - accumulated number of active warps per cycle. For every 
  # cycle it increments by the number of active warps in the cycle which can 
  # be in the range 0 to GPU_MAX_WARP. 
  echo "active_warps"  > $COMPUTE_PROFILE_CONFIG
  $EXE >> $EXE_LOG
  mv $COMPUTE_PROFILE_LOG $OUTPUT.WARPS.RAW
 
  paste -d"," $OUTPUT.CYCLES.RAW $OUTPUT.WARPS.RAW | \
    awk -vmax_warp=$GPU_MAX_WARP -vkernel_regexp="$KERNEL_REGEXP" -F, \
    '{if (!match($1, kernel_regexp)) print "nan"
      else print 100 * (($10/$5)/max_warp)}' > $OUTPUT

  printf "done, log at $OUTPUT\n"
}

###########################################################################
# Log the percentage of branches that are causing divergence within a warp 
# amongst all the branches present in the kernel. Note that this is a derived 
# counter from two performance counters: (i) branch, which is the total number
# of branches taken by the threads and (ii) divergenet_branch which is a 
# a counter incremented by one if at least one thread in a warp diverges (that 
# is, follows a different execution path). Divergent branches percentage is 
# calculated as (100*divergent branch)/(divergent branch + branch)
# The log file has one column:
# 1: percentage of divergent branches (db)
###########################################################################
function log_divergent_branches() {
  local OUTPUT=$DB_LOG
  printf "Profiling divergent branches... "

  echo "branch" >  $COMPUTE_PROFILE_CONFIG
  $EXE >> $EXE_LOG
  mv $COMPUTE_PROFILE_LOG $OUTPUT.BRANCH.RAW

  echo "divergent_branch" > $COMPUTE_PROFILE_CONFIG
  $EXE >> $EXE_LOG
  mv $COMPUTE_PROFILE_LOG $OUTPUT.DIVERGENT.RAW

  paste -d"," $OUTPUT.BRANCH.RAW $OUTPUT.DIVERGENT.RAW | \
    awk -vkernel_regexp="$KERNEL_REGEXP" -F, \
    '{if (!match($1, kernel_regexp)) print "nan" 
      else print 100 * $10 / ($10 + $5)}' > $OUTPUT
  printf "done, log at $OUTPUT\n"
}

###########################################################################
# This gives an indication of the number of bank conflicts caused per shared 
# memory instruction. This is calculated as: 
# 100 * (l1 shared bank conflict)/(shared load + shared store)
# The log file has one column:
# 1: Percentage of bank conflicts per shared memory instruction (bc)
###########################################################################
function log_bank_conflicts() {
  local OUTPUT=$BANK_CONFLICTS
  printf "Profiling bank conflicts... "

  echo "l1_shared_bank_conflict" >  $COMPUTE_PROFILE_CONFIG
  echo "shared_load" >> $COMPUTE_PROFILE_CONFIG
  echo "shared_store" >> $COMPUTE_PROFILE_CONFIG
  $EXE >> $EXE_LOG
  mv $COMPUTE_PROFILE_LOG $OUTPUT.SHARED.RAW

  awk -F, '{if (!match($1, kernel_regexp) || (($6+$7) == 0)) print "nan" 
            else print 100 * $5 / ($6 + $7)}' $OUTPUT.SHARED.RAW > $OUTPUT
  printf "done, log at $OUTPUT\n"
}

###########################################################################
# Derive the control flow divergence. This derived statistic gives the 
# percentage of thread instructions that were not executed by all threads in the
# warp, hence causing divergence
# 1: control flow divergence (cfd)
###########################################################################
function derive_control_flow_divergence() {
  local OUTPUT=$CONTROL_FLOW_DIV
  if [ ! -f $THRDS_INST_EXEC_LOG -o ! -f $INST_EXEC_LOG ]; then
    echo "Missing log files $THRDS_INST_EXEC_LOG or $INST_EXEC_LOG"; 
    exit 1
  fi

  paste -d" "  $INST_EXEC_LOG $THRDS_INST_EXEC_LOG | \
    awk '{if (!$2 || $2 == 0 || $2 == "nan") print "nan"; 
          else print ((100 * ($1 - $2)) / $1)}' > $OUTPUT
}

###########################################################################
# Derive the percentage of excess data that is fetched while making global
# memory load accesses. Ideally 0% excess loads will be achieved when kernel 
# requested global memory read throughput is equal to the L2 cache read 
# throughput i.e. the number of bytes requested by the kernel in terms of reads 
# are equal to the number of bytes actually fetched by the hardware during 
# kernel execution to service the kernel. If this statistic is high, it implies 
# that the access pattern for fetch is not coalesced, many extra bytes are 
# getting fetched while serving the threads of the kernel. It is calculated as: 
# 100 - (100 * requested global memory read bytes / l2 read bytes). The log file
# has one column:
# 1: global memory excess load (geld)
###########################################################################
function derive_gm_excess_load() {
  local OUTPUT=$GM_EXCESS_LOAD_LOG
  if [ ! -f $GLD_LOG -o ! -f $L2LD_LOG ]; then
    echo "Missing log files $GLD_LOG or $L2LD_LOG"; exit 1
  fi

  paste -d" "  $GLD_LOG $L2LD_LOG | \
    awk '{if (!$2 || $2 == 0 || $2 == "nan") print "nan"; 
          else print (100 - (100 * $1 / $2))}' > $OUTPUT
}

###########################################################################
# Derive the percentage of excess data that is accessed  while making global
# memory store transactions. (check derive_gm_excess_load for details).
# The log file has one column:
# 1: global memory excess store (gest)
###########################################################################
function derive_gm_excess_store() {
  local OUTPUT=$GM_EXCESS_STORE_LOG
  if [ ! -f $GST_LOG -o ! -f $L2ST_LOG ]; then
    echo "Missing log files $GST_LOG or $L2ST_LOG"; exit 1
  fi

  paste -d" "  $GST_LOG $L2ST_LOG | \
    awk '{if (!$2 || $2 == 0 || $2 == "nan") print "nan"; 
          else print (100 - (100 * $1 / $2))}' > $OUTPUT
}

###########################################################################
# Derive the ratio of the total number of instructions issued by the kernel 
# to the total number of bytes accessed from global memory. This is used
# to evaluate if the kernel is memory or compute bound. This log file has
# one column:
# 1: instructions per byte (IpB)
###########################################################################
function derive_inst_per_byte() {
  local OUTPUT=$INST_PER_BYTE_LOG
  if [ ! -f $INST_ISSD_LOG -o ! -f $L2LD_LOG -o ! -f $L2ST_LOG ]; then
    echo "Missing log file $INST_ISSD_LOG, $L2LD_LOG or $L2ST_LOG"; exit 1
  fi

  paste -d " " $INST_ISSD_LOG $L2LD_LOG $L2ST_LOG | \
    awk -vsm_num=$GPU_SM_NUM '{if ($2 + $3 > 0)
                                 print $1 * sm_num / ($2 + $3)
                               else print "nan"}' > $OUTPUT
}

###################################################################
# Print friendly the performance counters.
# OUTPUT (in/out): the output file (which can be /dev/stdout)
###################################################################
function print_friendly() {
  local OUTPUT=$1

  printf "==========================================================" > $OUTPUT
  printf "===============================\n" > $OUTPUT
  printf "gputime\tocc.\ta-occ.\tgeld\tgest\tL2ld\tinst/B\tdb\tcfd" >> $OUTPUT
  printf "\tbc\t\tkernel\n" >> $OUTPUT
  printf "ms\t%%\t%%\t%%\t%%\tGB/s\tratio\t%%\t%%\t%%\t\tname\n" >> $OUTPUT
  printf "==========================================================" >> $OUTPUT
  printf "===============================\n" >> $OUTPUT
  # 1: kernel name
  # 2: gputime
  # 3: cputime
  # 4: occupancy
  # 5: geld
  # 6: gest
  # 7: l2ld
  # 8: achieved occupancy
  # 9: instructions per byte
  # 10: divergant branches
  # 11: control flow divergence
  # 12: bank conflicts
  paste -d" " $TIME_LOG $GM_EXCESS_LOAD_LOG $GM_EXCESS_STORE_LOG $L2LD_LOG \
      $ACW_LOG $INST_PER_BYTE_LOG $DB_LOG $CONTROL_FLOW_DIV $BANK_CONFLICTS | \
    $FILTER_MASTER | \
    awk '{gsub(/^_Z[0-9]*/, "", $1); gsub(/kernel.*$/, "kernel", $1);
          printf "%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%s\t%s\n",
          $2/1000, $4, $8, $5, $6, $7/(1000*$2), $9, $10, $11, $12, $1}' >> $OUTPUT
}

########################################################################
# Print the standard and vwarp kernels side by side. Function parameters:
# OUTPUT: the file where the output will be printed (can be /dev/stdout) 
########################################################################
function print_friendly_standard_vwarp() {
  local OUTPUT=$1
  local TOTAL_LOG="$LOG_DIR/$LOG_PREFIX""TOTAL.log"
  print_friendly $TOTAL_LOG

  cat $TOTAL_LOG | $FILTER_MASTER | $FILTER_ST_KERNELS > $TOTAL_LOG.ST
  cat $TOTAL_LOG | $FILTER_MASTER | $FILTER_VW_KERNELS > $TOTAL_LOG.VW


  printf "=========================================================" > $OUTPUT
  printf "=========================================================" >> $OUTPUT
  printf "====================\n" >> $OUTPUT
  printf "gputime\tgeld\tgeld\tgest\tgest\tL2ld\tL2ld\ta-occ.\t" >> $OUTPUT
  printf "a-occ.\tIpB\tIpB\tdb\tdb\tcfd\tcfd\tkernel\n" >> $OUTPUT
  printf "st/vw\tst-%%\tvw-%%\tst-%%\tvw-%%" >> $OUTPUT
  printf "\tst-GB/s\tvw-GB/s\tst-%%\tvw-%%\tst\tvw\tst-%%\tvw-%%\t" >> $OUTPUT
  printf "st-%%\tvw-%%\tname\n" >> $OUTPUT
  printf "=========================================================" >> $OUTPUT
  printf "=========================================================" >> $OUTPUT
  printf "====================\n" >> $OUTPUT

  # 1: gputime
  # 2: theoritical occupancy
  # 3: achieved occupancy
  # 4: geld
  # 5: gest
  # 6: l2ld
  # 7: inst/B
  # 8: divergant branches
  # 9: control flow divergant
  # 10: bank conflict
  # 11: kernel name
  paste -d" " $TOTAL_LOG.ST $TOTAL_LOG.VW | \
    awk '{ gsub(/^_Z[0-9]*/, "", $1); gsub(/kernel.*$/, "kernel", $1);
           gputime=1; occ=2; a_occ=3; geld=4; gest=5; l2ld=6; ipb=7;
           db=8; cfd=9; bc=10; kname=11; count = 11;
           printf "%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t",
                  $gputime/$(gputime+count), $geld, $(geld+count), 
                  $gest, $(gest + count), $l2ld, $(l2ld + count);
           printf "%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%s\n",
                  $a_occ, $(a_occ+count), $ipb, $(ipb+count), 
                  $db, $(db+count), $cfd, $(cfd+count), $kname}' >> $OUTPUT

  rm $TOTAL_LOG.ST $TOTAL_LOG.VW $TOTAL_LOG
}

# Check if profiling is required
if [ "$EXE_ENABLED" == "YES" ]; then
  log_time
  log_l2_read_requests
  log_l2_write_requests
  log_global_memory_read_requests
  log_global_memory_write_requests
  log_active_cycles_and_warps
  log_instructions_issued
  log_instructions_executed
  log_threads_instructions_executed
  log_divergent_branches
  log_bank_conflicts
  derive_gm_excess_load
  derive_gm_excess_store
  derive_inst_per_byte
  derive_control_flow_divergence
fi

# Produce user-friendly output
if [ "$PRINT_SIDE_BY_SIDE" == "YES" ]; then
  print_friendly_standard_vwarp $OUTPUT_FILE
else
  print_friendly $OUTPUT_FILE
fi
