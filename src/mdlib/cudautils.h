#ifndef CUDAUTILS_H
#define CUDAUTILS_H

#include "stdio.h"

#include "gmx_fatal.h"

#include "cuda.h"

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

#define CU_LAUNCH_ERR(msg) \
    do { \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            gmx_fatal(FARGS, "Error while launching kernel %s: %s\n", msg, cudaGetErrorString(status)); \
        } \
    } while (0)

#else

#define CU_RET_ERR(status, s) do { } while (0)

#define CU_LAUNCH_ERR(status, s) do { } while (0)

#endif /* CHECK_CUDA_ERRORS */ 




#define GRID_MAX_DIM        65535


#ifdef __cplusplus
extern "C" {
#endif
int download_cudata(void * /*h_dest*/, void * /*d_src*/, size_t /*bytes*/);

int download_cudata_alloc(void ** /*h_dest*/, void * /*d_src*/, size_t /*bytes*/);

int upload_cudata(void * /*d_dest*/, void * /*h_src*/, size_t /*bytes*/);

int upload_cudata_alloc(void ** /*d_dest*/, void * /*h_src*/, size_t /*bytes*/);

void * pmalloc(size_t /*bytes*/, FILE * /*fplog*/);

void pfree(void * /*h_ptr*/, FILE * /*fplog*/);

#ifdef __cplusplus
}
#endif


#endif
