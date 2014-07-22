#ifndef _GCL_H_
#define _GCL_H_

#include <iostream>
#ifdef _GCL_MPI_
#include <mpi.h>
#endif

#include "../common/host_device.h"


#ifdef GCL_TRACE
#include "high-level/stats_collector.h"
#endif


#include "../common/boollist.h"
#include "low-level/gcl_arch.h"


#ifdef _GCL_GPU_

// workaround that uses host buffering to avoid bad sends for messages larger than 512 kB on Cray systems
//#define HOSTWORKAROUND

#define _USE_DATATYPES_


inline bool checkCudaStatus( cudaError_t status ) {
  if( status != cudaSuccess ) {
    std::cout << cudaGetErrorString( status ) << std::endl;
    return false;
  } else return true;
}
#endif

#ifdef _GCL_GPU_
#ifdef GCL_MULTI_STREAMS
#ifdef GCL_USE_3
#pragma message "Using 3 streams for packing and unpaking in GCL"
    extern cudaStream_t ZL_stream ;
    extern cudaStream_t& ZU_stream ;
    extern cudaStream_t YL_stream ;
    extern cudaStream_t& YU_stream ;
    extern cudaStream_t XL_stream ;
    extern cudaStream_t& XU_stream ;
#else
#pragma message "Using 6 streams for packing and unpaking in GCL"
    extern cudaStream_t ZL_stream ;
    extern cudaStream_t ZU_stream ;
    extern cudaStream_t YL_stream ;
    extern cudaStream_t YU_stream ;
    extern cudaStream_t XL_stream ;
    extern cudaStream_t XU_stream ;
#endif
#else
#define ZL_stream 0
#define ZU_stream 0
#define YL_stream 0
#define YU_stream 0
#define XL_stream 0
#define XU_stream 0
#endif
#endif

namespace gridtools {

enum packing_version {version_mpi_pack=0, version_datatype, version_manual};

#ifdef _GCL_MPI_
  extern MPI_Comm GCL_WORLD;
#else
  extern int GCL_WORLD;
#endif
  extern int PID;
  extern int PROCS;

  void GCL_Init(int argc, char** argv);

  void GCL_Init();

  void GCL_Finalize();

} // namespace gridtools


#endif
