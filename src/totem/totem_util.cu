/**
 *  Defines CUDA utility funtions
 *
 *  Created on: 2011-10-07
 *  Author: Greg Redekop
 */

// totem includes
#include "totem_util.h"


/**
 * Ensure the device supports the minimum CUDA architecture requirements
 * @return generic success or failure
 */
error_t check_cuda_version() {

  /* check that the device(s) actually support the right CUDA version */
  int num_devices = 0;
  cudaGetDeviceCount(&num_devices);
  if (num_devices == 0) {
    fprintf(stdout, "No CUDA-supported devices found\n");
    return FAILURE;
  }
  cudaDeviceProp device;
  /* TODO: Check for other CUDA devices */
  cudaGetDeviceProperties(&device, 0);
  if(device.major < REQ_CUDAVERSION_MAJOR ||
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

