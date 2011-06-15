#include <stdlib.h>

#include "gmx_fatal.h"
#include "smalloc.h"

#include "cuda.h"
#include "cudautils.h"

/*** General CUDA data operations ***/
/* TODO: create a cusmalloc module that implements similar things as smalloc */

int _download_cudata_generic(void * h_dest, void * d_src, size_t bytes, 
                             gmx_bool async = FALSE, cudaStream_t stream = 0)
{
    cudaError_t stat;
    
    if (h_dest == 0 || d_src == 0 || bytes <= 0)
        return -1;

    if (async)
    {
        stat = cudaMemcpyAsync(h_dest, d_src, bytes, cudaMemcpyDeviceToHost, stream);
        CU_RET_ERR(stat, "DtoH cudaMemcpyAsync failed");

    }
    else
    {
        stat = cudaMemcpy(h_dest, d_src, bytes, cudaMemcpyDeviceToHost);
        CU_RET_ERR(stat, "DtoH cudaMemcpy failed");
    }

    return 0;
}

int download_cudata(void * h_dest, void * d_src, size_t bytes)
{
    return _download_cudata_generic(h_dest, d_src, bytes, FALSE);
}

int download_cudata_async(void * h_dest, void * d_src, size_t bytes, cudaStream_t stream = 0)
{
    return _download_cudata_generic(h_dest, d_src, bytes, TRUE, stream);
}

int download_cudata_alloc(void ** h_dest, void * d_src, size_t bytes)
{ 
    if (h_dest == 0 || d_src == 0 || bytes <= 0)
        return -1;

    smalloc(*h_dest, bytes);

    return download_cudata(*h_dest, d_src, bytes);
}


int _upload_cudata_generic(void * d_dest, void * h_src, size_t bytes, 
                                 gmx_bool async = FALSE, cudaStream_t stream = 0)
{
    cudaError_t stat;

    if (d_dest == 0 || h_src == 0 || bytes <= 0)
        return -1;

    if (async)
    {
        stat = cudaMemcpyAsync(d_dest, h_src, bytes, cudaMemcpyHostToDevice, stream);
        CU_RET_ERR(stat, "HtoD cudaMemcpyAsync failed");
    }
    else
    {
        stat = cudaMemcpy(d_dest, h_src, bytes, cudaMemcpyHostToDevice);
        CU_RET_ERR(stat, "HtoD cudaMemcpy failed");
    }

    return 0;
}

int upload_cudata(void * d_dest, void * h_src, size_t bytes)
{   
    return _upload_cudata_generic(d_dest, h_src, bytes, FALSE);
}

int upload_cudata_async(void * d_dest, void * h_src, size_t bytes, cudaStream_t stream = 0)
{   
    return _upload_cudata_generic(d_dest, h_src, bytes, TRUE, stream);
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

int cu_blockwait_event(cudaEvent_t stop, cudaEvent_t start, float *time)
{
    cudaError_t s;

    s = cudaEventSynchronize(stop);
    CU_RET_ERR(s, "cudaEventSynchronize failed in cu_blockwait_event");

    s = cudaEventElapsedTime(time, start, stop);
    CU_RET_ERR(s, "cudaEventElapsedTime failed in cu_blockwait_event");

    return 0;
}

/* Binds texture with name tex_name to the GPU global memory (of size elements) 
   pointed by d_ptr.
   Returns the offset that needs to be used when fetching from the texture.
 */
template <typename T>
size_t cu_bind_texture(const char *tex_name, const T *d_ptr, int size)
{
    cudaError_t             stat;
    cudaChannelFormatDesc   cd;
    const textureReference  *tex;
    char                    str[100];

    size_t offset;

    stat = cudaGetTextureReference(&tex, tex_name);
    sprintf(str, "cudaGetTextureReference on %s failed", tex_name);
    CU_RET_ERR(stat, str);
    cd = cudaCreateChannelDesc<T>();

    stat = cudaBindTexture(&offset, tex, d_ptr, &cd, size*sizeof(*d_ptr));
    sprintf(str, "cudaBindTexture on %s failed ", tex_name);
    CU_RET_ERR(stat, str);

    return offset;
}

/* Instantiate cu_bind_texture with float */
template size_t cu_bind_texture<float>(const char *, const float *, int);

/*! Unbinds texture with name tex_name. */
void cu_unbind_texture(const char *tex_name)
{
    cudaError_t             stat;
    const textureReference  *tex;
    char                    str[100];

    stat = cudaGetTextureReference(&tex, tex_name);
    sprintf(str, "cudaGetTextureReference on %s failed", tex_name);
    CU_RET_ERR(stat, str);
    stat = cudaUnbindTexture(tex);
    sprintf(str, "cudaUnbindTexture on %s failed ", tex_name);
    CU_RET_ERR(stat, str);
}


