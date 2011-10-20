#ifndef CUDAUTILS_H
#define CUDAUTILS_H

#include "stdio.h"

#include "gmx_fatal.h"

#include "cuda.h"

/*
   CUDA library and hardware related defines
 */


#define GRID_MAX_DIM        65535
#define WARP_SIZE           32


/* TODO error checking needs to be rewritten. We have 2 types of error checks needed 
   based on where they occur in the code: 
   - non performance-critical: these errors are unsafe to be ignored and must be 
     _always_ checked for, e.g. initializations
   - performance critical: handling errors might hurt performance so care need to be taken
     when/if we should check for them at all, e.g. in cu_upload_X. However, we should be 
     able to turn the check for these errors on!

  Probably we'll need two sets of the macros below... 
 
 */
#define CHECK_CUDA_ERRORS

#ifdef CHECK_CUDA_ERRORS

#define CU_RET_ERR(status, msg) \
    do { \
        if (status != cudaSuccess) \
        { \
            gmx_fatal(FARGS, "%s: %s\n", msg, cudaGetErrorString(status)); \
        } \
    } while (0)

#define CU_CHECK_PREV_ERR() \
    do { \
        cudaError_t _CU_CHECK_PREV_ERR_status = cudaGetLastError(); \
        if (_CU_CHECK_PREV_ERR_status != cudaSuccess) { \
            gmx_warning("Just caught a previously occured CUDA error (%s), will try to continue.", cudaGetErrorString(_CU_CHECK_PREV_ERR_status)); \
        } \
    } while (0)

#define CU_LAUNCH_ERR(msg) \
    do { \
        cudaError_t _CU_LAUNCH_ERR_status = cudaGetLastError(); \
        if (_CU_LAUNCH_ERR_status != cudaSuccess) { \
            gmx_fatal(FARGS, "Error while launching kernel %s: %s\n", msg, cudaGetErrorString(_CU_LAUNCH_ERR_status)); \
        } \
    } while (0)

#define CU_LAUNCH_ERR_SYNC(msg) \
    do { \
        cudaError_t _CU_SYNC_LAUNCH_ERR_status = cudaThreadSynchronize(); \
        if (_CU_SYNC_LAUNCH_ERR_status != cudaSuccess) { \
            gmx_fatal(FARGS, "Error while launching kernel %s: %s\n", msg, cudaGetErrorString(_CU_SYNC_LAUNCH_ERR_status)); \
        } \
    } while (0)

#else

#define CU_RET_ERR(status, msg) do { } while (0)
#define CU_CHECK_PREV_ERR()     do { } while (0)
#define CU_LAUNCH_ERR(msg)      do { } while (0)
#define CU_LAUNCH_ERR_SYNC(msg) do { } while (0)

#endif /* CHECK_CUDA_ERRORS */ 

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cu_dev_info  cu_dev_info_t;
struct cu_dev_info
{
    int dev_id;                 /* id of the CUDA device in use */
    cudaDeviceProp dev_prop;    /* CUDA device properties */
};

int download_cudata(void * /*h_dest*/, void * /*d_src*/, size_t /*bytes*/);
int download_cudata_async(void * /*h_dest*/, void * /*d_src*/, size_t /*bytes*/, cudaStream_t /*stream = 0*/);
int download_cudata_alloc(void ** /*h_dest*/, void * /*d_src*/, size_t /*bytes*/);

int upload_cudata(void * /*d_dest*/, void * /*h_src*/, size_t /*bytes*/);
int upload_cudata_async(void * /*d_dest*/, void * /*h_src*/, size_t /*bytes*/, cudaStream_t /*stream = 0*/);
int upload_cudata_alloc(void ** /*d_dest*/, void * /*h_src*/, size_t /*bytes*/);

int cu_blockwait_event(cudaEvent_t /*stop*/, cudaEvent_t /*start*/, float * /*time*/);

void cu_unbind_texture(const char * /*tex_name*/);

float cu_event_elapsed(cudaEvent_t /*start*/, cudaEvent_t /*stop*/);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
template <typename T>
size_t cu_bind_texture(const char * /*tex_name*/, const T * /*d_ptr*/, int /*size*/);
#endif

#endif /* CUDAUTILS_H */