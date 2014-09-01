/**
 *  Defines CUDA utility funtions
 *
 *  Created on: 2011-10-07
 *  Author: Greg Redekop
 */

// totem includes
#include "totem_util.h"
#include "totem_partition.h"

/**
 * Ensure the device supports the minimum CUDA architecture requirements
 * @return generic success or failure
 */
error_t check_cuda_version() {
  // Check that the device(s) actually support the right CUDA version.
  int num_devices = 0;
  cudaGetDeviceCount(&num_devices);
  if (num_devices == 0) {
    fprintf(stdout, "No CUDA-supported devices found\n");
    return FAILURE;
  }
  cudaDeviceProp device;
  // TODO(abdullah): Check for other CUDA devices.
  cudaGetDeviceProperties(&device, 0);
  if (device.major < REQ_CUDAVERSION_MAJOR ||
      (device.minor < REQ_CUDAVERSION_MINOR &&
       device.major == REQ_CUDAVERSION_MAJOR)) {
    fprintf(stdout, "Detected CUDA version %d.%d "
            "is less than required version %d.%d\n",
            device.major, device.minor, REQ_CUDAVERSION_MAJOR,
            REQ_CUDAVERSION_MINOR);
    return FAILURE;
  }
  return SUCCESS;
}

int get_gpu_count() {
  int gpu_count = 0;
  cudaGetDeviceCount(&gpu_count);
  return min((MAX_PARTITION_COUNT - 1), gpu_count);
}

int compare_ids_asc(const void* a, const void* b) {
  vid_t v1 = *(reinterpret_cast<const vid_t*>(a));
  vid_t v2 = *(reinterpret_cast<const vid_t*>(b));
  return v1 - v2;
}

int compare_ids_dsc(const void* a, const void* b) {
  vid_t v1 = *(reinterpret_cast<const vid_t*>(a));
  vid_t v2 = *(reinterpret_cast<const vid_t*>(b));
  return v2 - v1;
}

bool compare_ids_tbb(const vid_t& v1, const vid_t& v2) {
  return (v1 < v2);
}

int get_mssb(uint32_t word) {
  if (word == 0) return -1;
  int mssb = 0;
  while ((word >>= 1) != 0) mssb++;
  return mssb;
}
