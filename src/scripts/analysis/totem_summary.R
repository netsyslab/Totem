## This script implements one main function called totem.summary. The
## function returns a data frame with averages and confidence intervals of the
## raw performance data produced by the Totem benchmark executed by run.sh.
## It takes as input a directory path to raw evaluation files, where each file
## will result in one row in the returned data frame.
##
## The returned data frame has the following columns (with no specific order):
## WORKLOAD, ALG, CPU_COUNT, GPU_COUNT, PAR, ALPHA, TOTAL, TOTAL_CI, EXEC, 
## EXEC_CI, INIT, INIT_CI, COMP, COMP_CI, COMM, COMM_CI, FINALIZE, FINALIZE_CI,
## GPUCOMP, GPUCOMP_CI, SCATTER, SCATTER_CI, GATHER, GATHER_CI, AGGR, AGGR_CI,
## TRV, TRV_CI, RATE, RATE_CI
##
## The function assumes that the raw data files in the directory have the
## following format: ALG_CPUs_GPUs_PAR_ALPHA_WORKLOAD.dat
##
## In the following we describe the format of the raw data files:
## The first line describes the experiment configuration:
##  * file:<graph file>\tbenchmark:<BFS|PageRank|etc>\tvertices:<number>\t
##  * edges:<number>\tpartitioning:<RAND|HIGH|LOW>\tplatform:<CPU|GPU|HYBRID>\t
##  * alpha:<percentage of edges on the CPU>\trepeat:<number of runs>\t
##  * gpu_count:<number>\tthread_count:<CPU threads>\tthread_bind:<TRUE|FALSE>\t
##  * time_init:<Totem init time>\ttime_par:<Graph partitoining time>\t"
##  * rmt_vertex:<% of remote vertices>\trmt_edge:<% of remote edges>\t
##  * beta:<% of remote edges after aggregation>
##
## Multiple lines, each line details a run's timing breakdown. Note that the 
## first line is a header:
##  * total\texec\tinit\tcomp\tcomm\tfinalize\tgpu_comp\tscatter\tgather\taggr\t
##  * trv_edges\texec_rate
##
## where
##
## total: Total run time, including output buffer allocation
## exec: Execution time without buffer allocation (comp + comm + aggr)
## init: Algorithm initialization (mainly buffer allocations)
## comp: Compute phase
## comm: Communication phase (inlcudes scatter/gather)
## finalize: Algorithm finalization (buffer deallocations)
## gpu_comp: GPU computation (included in comp)
## scatter: The scatter step in communication (push mode)
## gather: The gather step in communication (pull mode)
## aggr: Final result aggregation
## trv_edges: Number of traversed edges
## exec_rate: Billion Traversed Edges Per Second (trv_edges/exec)
##
## The following is an example:
##
## * file:/ssd/graphs/bter_18_64.tbin	benchmark:BFS	vertices:63742081      \
## * edges:1011271202	partitioning:LOW	platform:HYBRID	alpha:5	       \
## * repeat:64\thread_count:12	thread_bind:true	time_init:55829.16     \
## * time_par:7383.63	rmt_vertex:46	rmt_edge:5	beta:3
## * total	exec	init	comp	comm	finalize	gpu_comp       \
## * scatter	gather	aggr	trv_edges	exec_rate
## * 1540.31	1309.85	27.92	950.76	220.93	3.70	950.24	206.09	0.00   \
## *	138.15	994293064	0.7591
## * 1515.95	1285.39	28.40	935.01	212.11	3.49	934.57	197.27	0.00   \
## * 138.25	994293064	0.7735
##
##
## Date: 2013-02-23
## Author: Abdullah Gharaibeh

## Returns the 95% confidence interval of the values in vector x
totem.ci95 <- function(x) {  
  1.9723 * sd(x) / sqrt(length(x));
}

## Returns a list containing the configuration of an experiment passed as a
## string with the following format: ALG_CPUs_GPUs_PAR_ALPHA_WORKLOAD. The
## configuration parameters can be accessed via the returned list like this:
## config$ALG etc.
totem.config <- function(f) {
  config = unlist(strsplit(f, "_"));
  ## put the workload name back together (gets split if the workload name is
  ## of the form X_Y_Z)
  config[6] = paste(config[6:length(config)], collapse='_')
  names(config) = c("ALG", "CPU_COUNT", "GPU_COUNT", "PAR", "ALPHA",
                    "WORKLOAD");
  config = as.list(config[1:6]);
  config$CPU_COUNT = as.integer(config$CPU_COUNT);
  config$GPU_COUNT = as.integer(config$GPU_COUNT);
  config$ALPHA = as.integer(config$ALPHA);
  return(config);
}

## Returns a data frame with averages and confidence interval of all the files
## with .dat extension and put the result in one data frame. 
totem.summary <- function(dir) {  
  ## A raw data file name has the format ALG_CPUs_GPUs_PAR_ALPHA_WORKLOAD
  files = list.files(dir, pattern="*.dat");
  l = length(files);
  data.setup = data.frame(ALG = character(l), CPU_COUNT = numeric(l),
                          GPU_COUNT = numeric(l), PAR = character(l),
                          ALPHA = numeric(l), WORKLOAD = character(l),
                          stringsAsFactors = FALSE);
  data.avg   = data.frame(TOT = numeric(l), EXEC = numeric(l), 
                          INIT = numeric(l), COMP = numeric(l),
                          COMM = numeric(l), FINALIZE = numeric(l),
                          CPUCOMP = numeric(l),
                          GPUCOMP = numeric(l), SCATTER = numeric(l),
                          GATHER = numeric(l), AGGR = numeric(l),
                          TRV = numeric(l), RATE = numeric(l));
  data.ci = data.frame(TOT_CI = numeric(l), EXEC_CI = numeric(l),
                       INIT_CI = numeric(l), COMP_CI = numeric(l),
                       COMM_CI = numeric(l), FIN_CI = numeric(l),
                       CPUCOMP_CI = numeric(l),
                       GPUCOMP_CI = numeric(l), SCAT_CI = numeric(l),
                       GATH_CI = numeric(l), AGGR_CI = numeric(l),
                       TRV_CI = numeric(l), RATE_CI = numeric(l));
  index = 1;
  for (f in files) {
    ## read.table function throws an error and halts execution if the file being
    ## read is empty. tryCatch will catch the error, and continue to parse the 
    ## next file
    tryCatch({
      ## The following columns are read: total, exec, init, comp, comm,
      ## finalize, gpu_comp, scatter, gather, aggr, trv_edges, exec_rate
      raw = read.table(paste(dir, f, sep="/"), header = TRUE, skip = 1);

      ## Handle the case where the file contains only the header with no data
      if (length(raw$total) != 0) {
        ## Compute average and 95 confidence interval
        data.avg[index,] = as.vector(sapply(raw, mean));
        data.ci[index,] = as.vector(sapply(raw, totem.ci95));
        
        ## get this experiment's configuration
        data.setup[index, ] = totem.config(f);
        index = index + 1;
      }
    }, error = function(err){});
  }
  data = data.frame(data.setup, data.avg, data.ci);
  return(with(data, data[order(WORKLOAD, ALG, CPU_COUNT,
                               GPU_COUNT, PAR, ALPHA), ]));
}
