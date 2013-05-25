/**
 * This header define the virtual warp parameters. Virtual warp is a technique
 * to reduce divergence among threads within a warp. The idea is to have all the
 * threads that belong to a warp work as a unit. To this end, instead of
 * dividing the work among threads, the work is divided among warps. A warp goes
 * through phases of SISD and SIMD in complete lock-step as if they were all a
 * single thread. The technique divides the work into batches, where each warp
 * is responsible for one batch of work. Typically, the threads of a warp
 * collaborate to fetch their assigned batch data to shared memory, and together
 * process the batch. 
 * The technique is presented in [Hong11] S. Hong, S. Kim, T. Oguntebi and 
 * K.Olukotun "Accelerating CUDA Graph Algorithms at Maximum Warp, PPoPP11.
 *
 *  Created on: 2013-05-02
 *  Author: Abdullah Gharaibeh
 */

#ifndef TOTEM_VWARP_CUH
#define TOTEM_VWARP_CUH

/**
 * the number of threads a warp consists of
 */
const int VWARP_SHORT_WARP_WIDTH = 1;
const int VWARP_MEDIUM_WARP_WIDTH = 32;
const int VWARP_LONG_WARP_WIDTH = 64;
const int VWARP_DEFAULT_WARP_WIDTH = 32;

/**
 * the size of the batch of work assigned to each virtual warp
 */
const int VWARP_SMALL_BATCH_SIZE = 4;
const int VWARP_MEDIUM_BATCH_SIZE = 32;
const int VWARP_LARGE_BATCH_SIZE = 64;
const int VWARP_DEFAULT_BATCH_SIZE = 64;

/**
 * The index of a thread in the virtual warp which it belongs to
 */
inline __device__
int vwarp_thread_index(const int vwarp_width) { 
  return THREAD_BLOCK_INDEX % vwarp_width;
}

/**
 * The warp index in the thread-block it belongs it
 */
inline __device__
int vwarp_warp_index(const int vwarp_width) {
  return THREAD_BLOCK_INDEX / vwarp_width;
}

/**
 * The index of the first vertex in the batch of vertices to be processed by 
 * a virtual warp
 */
inline __device__
vid_t vwarp_warp_start_vertex(const int vwarp_width, const int vwarp_batch) {
  return vwarp_warp_index(vwarp_width) * vwarp_batch;
}

/**
 * The number of virtual warp in a thread-block
 */
#define VWARP_WARPS_PER_BLOCK(_vwarp_width)     \
  (MAX_THREADS_PER_BLOCK / (_vwarp_width))

/**
 * The maximum size of work that can be assigned to a thread-block
 */
#define VWARP_BLOCK_MAX_BATCH_SIZE(_vwarp_width, _vwarp_batch)  \
  (VWARP_WARPS_PER_BLOCK(_vwarp_width) * _vwarp_batch)

/**
 * The id of the first vertex in the batch of vertices to be processed by a
 * thread-block
 */
inline __device__
vid_t vwarp_block_start_vertex(const int vwarp_width, const int vwarp_batch) {
  return BLOCK_GLOBAL_INDEX * VWARP_WARPS_PER_BLOCK(vwarp_width) * vwarp_batch;
}

/**
 * The amount of work assigned to a specific thread-block
 */
inline __device__
vid_t vwarp_block_batch_size(const vid_t vertex_count, const int vwarp_width, 
                             const int vwarp_batch) {
  vid_t start_vertex = vwarp_block_start_vertex(vwarp_width, vwarp_batch);
  vid_t last_vertex = start_vertex + 
    VWARP_BLOCK_MAX_BATCH_SIZE(vwarp_width, vwarp_batch);
  return (last_vertex > vertex_count ? (vertex_count - start_vertex) :
          VWARP_BLOCK_MAX_BATCH_SIZE(vwarp_width, vwarp_batch));
}

/*
 * The size of the batch of work assigned to a specific virtual warp
 */
inline __device__
vid_t vwarp_warp_batch_size(const vid_t vertex_count, const int vwarp_width,
                            const int vwarp_batch) {
  vid_t block_batch_size = vwarp_block_batch_size(vertex_count, vwarp_width, 
                                                  vwarp_batch);
  vid_t start_vertex = vwarp_warp_start_vertex(vwarp_width, vwarp_batch); 
  vid_t last_vertex = start_vertex + vwarp_batch;
  return (last_vertex >= block_batch_size ? (block_batch_size - start_vertex) :
          vwarp_batch);
}

/**
 * The number of batches of vertices a workload has
 */
inline __device__ __host__
vid_t vwarp_batch_count(const int vertex_count, const int vwarp_batch) {
  return ((vertex_count / vwarp_batch) + 
          (vertex_count % vwarp_batch == 0 ? 0 : 1));
}

/**
 * The number of threads needed to process a given number of vertices 
 */
inline __device__ __host__
vid_t vwarp_thread_count(const vid_t vertex_count, const int vwarp_width, 
                         const int vwarp_batch) {
  return vwarp_width * vwarp_batch_count(vertex_count, vwarp_batch);
}

/**
 * The number of threads needed to process a given number of vertices. This
 * version of the macro uses the default configuration parameters 
 */
inline __device__ __host__
vid_t vwarp_default_thread_count(const vid_t vertex_count) {
  return VWARP_DEFAULT_WARP_WIDTH * 
    vwarp_batch_count(vertex_count, VWARP_DEFAULT_BATCH_SIZE);
}

/**
 * The length of the state to be allocated by warp-based implementations that 
 * uses default configuration parameters
 */
inline __device__ __host__
vid_t vwarp_default_state_length(const vid_t vertex_count) {
  return VWARP_DEFAULT_BATCH_SIZE * 
    vwarp_batch_count(vertex_count, VWARP_DEFAULT_BATCH_SIZE);
}

/**
 * This threshold determines when to switch from coarse to fine grained
 * parallelism in traversal-based algorithms. Once the number of active
 * vertices (i.e., the number of vertices in the frontier) is larger than
 * this threshold, vwarp-based kernel could be configured to use a longer
 * warp width to better utilize parallelism across the neighbours. This 
 * threshold has been determined experimentally and it is proportional to
 * the number of hardware threads available in the GPU.
 */
const vid_t VWARP_ACTIVE_VERTICES_THRESHOLD = 3000;

/**
 * A SIMD version of memcpy for the virtual warp technique. The assumption is
 * that the threads of a warp invoke this function to copy their batch of work
 * from global memory (src) to shared memory (dst). In each iteration of the for
 * loop, each thread copies one element. For example, thread 0 in the warp
 * copies elements 0, VWARP_WARP_WIDTH, (2 * VWARP_WARP_WIDTH) etc., while
 * thread 1 in the warp copies elements 1, (1 + VWARP_WARP_WIDTH),
 * (1 + 2 * VWARP_WARP_WIDTH) and so on. Finally, this function is called only
 * from within a kernel.
 * @param[in] dst destination buffer (typically shared memory buffer)
 * @param[in] src the source buffer (typically global memory buffer)
 * @param[in] size number of elements to copy
 * @param[in] thread_offset_in_warp thread index within its warp
 */
template<typename T>
inline __device__ 
void vwarp_memcpy(T* dst, T* src, uint32_t size, uint32_t thread_offset,
                  uint32_t threads_per_block = VWARP_DEFAULT_WARP_WIDTH) {
  for(int i = thread_offset; i < size; i += threads_per_block) dst[i] = src[i];
}

#endif  // TOTEM_VWARP_CUH
