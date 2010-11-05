#include <stdlib.h>

#include "gmx_fatal.h"
#include "smalloc.h"

#include "cuda.h"
#include "cudautils.h"

/*** General CUDA data operations ***/
/* TODO: create a cusmalloc module that implements similar things as smalloc */

int download_cudata(void * h_dest, void * d_src, size_t bytes)
{
    cudaError_t stat;
    
    if (h_dest == 0 || d_src == 0 || bytes <= 0)
        return -1;
    
    stat = cudaMemcpy(h_dest, d_src, bytes, cudaMemcpyDeviceToHost);
    CU_RET_ERR(stat, "DtoH cudaMemcpy failed");

    return 0;
}

int download_cudata_alloc(void ** h_dest, void * d_src, size_t bytes)
{ 
    if (h_dest == 0 || d_src == 0 || bytes <= 0)
        return -1;

    smalloc(*h_dest, bytes);

    return download_cudata(*h_dest, d_src, bytes);
}

int upload_cudata(void * d_dest, void * h_src, size_t bytes)
{   
    cudaError_t stat;

    if (d_dest == 0 || h_src == 0 || bytes <= 0)
        return -1;

    stat = cudaMemcpy(d_dest, h_src, bytes, cudaMemcpyHostToDevice);
    CU_RET_ERR(stat, "HtoD cudaMemcpy failed");

    return 0;
}

int upload_cudata_alloc(void ** d_dest, void * h_src, size_t bytes)
{
    cudaError_t stat;
    
    if (d_dest == 0 || h_src == 0 || bytes <= 0)
        return -1;

    stat = cudaMalloc(d_dest, bytes);
    CU_RET_ERR(stat, "cudaMalloc failed in upload_cudata_alloc");

    return upload_cudata(*d_dest, h_src, bytes);
}
