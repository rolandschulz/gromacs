#include "cudatype_utils.h"

#define CELL_SIZE_2         (CELL_SIZE * CELL_SIZE)
#define STRIDE_DIM          (CELL_SIZE_2)
#define STRIDE_SI           (3*STRIDE_DIM)
#define GPU_FACEL           (138.935485f)

/* texture reference bound to the cudata.nbfp array */
texture<float, 1, cudaReadModeElementType> tex_nbfp;

/* texture reference bound to the cudata.coulomb_tab array */
texture<float, 1, cudaReadModeElementType> tex_coulomb_tab;

inline __device__ void reduce_force_i_generic_strided(float *fbuf, float4 *fout,
        int tidxi, int tidxj, int aidx)
{
    if (tidxj == 0)
    {
        float4 f = make_float4(0.0f);
        for (int j = tidxi * CELL_SIZE; j < (tidxi + 1) * CELL_SIZE; j++)
        {
            f.x += fbuf[                 j];
            f.y += fbuf[    STRIDE_DIM + j];
            f.z += fbuf[2 * STRIDE_DIM + j];
        }

        atomicAdd(&fout[aidx].x, f.x);
        atomicAdd(&fout[aidx].y, f.y);
        atomicAdd(&fout[aidx].z, f.z);
    }
}

inline __device__ void reduce_force_j_generic_strided(float *fbuf, float4 *fout,
        int tidxi, int tidxj, int aidx)
{
    if (tidxi == 0)
    {
        float4 f = make_float4(0.0f);
        for (int j = tidxj; j < CELL_SIZE_2; j += CELL_SIZE)
        {
            f.x += fbuf[                 j];
            f.y += fbuf[    STRIDE_DIM + j];
            f.z += fbuf[2 * STRIDE_DIM + j];
        }

        atomicAdd(&fout[aidx].x, f.x);
        atomicAdd(&fout[aidx].y, f.y);
        atomicAdd(&fout[aidx].z, f.z);
    }
}

inline __device__ float coulomb(float q1, 
                                float q2,
                                float r2, 
                                float inv_r, 
                                float inv_r2, 
                                float beta,
                                float erfc_tab_scale)
{
    float x      = r2 * inv_r * beta;
    float x2     = x * x; 
    // float inv_x2 = inv_r2 / (beta * beta); 
    float res    =
        q1 * q2 * (erfc(x) * inv_r + beta * exp(-x2)) * inv_r2;
    return res;
}

/* source: OpenMM */
static __device__ float interpolate_coulomb_force_r(float r, float scale)
{  
    float   normalized = scale * r;
    int     index = (int) normalized;
    float   fract2 = normalized - index;
    float   fract1 = 1.0f - fract2;

    return  fract1 * tex1Dfetch(tex_coulomb_tab, index) 
            + fract2 * tex1Dfetch(tex_coulomb_tab, index + 1);
}

/*********************************************************************************/
/* Old stuff  */
#if 0
/* 8x8 */
__device__ void reduce_force8_strided(float *fbuf, float4 *fout,
        int tidxi, int tidxj, int ai)
{
    /* 64 -> 32: 8x4 threads */
    if (tidxj < CELL_SIZE/2)
    {
        fbuf[                 tidxi * CELL_SIZE + tidxj] += fbuf[                 tidxi * CELL_SIZE + tidxj + CELL_SIZE/2];
        fbuf[    STRIDE_DIM + tidxi * CELL_SIZE + tidxj] += fbuf[    STRIDE_DIM + tidxi * CELL_SIZE + tidxj + CELL_SIZE/2];
        fbuf[2 * STRIDE_DIM + tidxi * CELL_SIZE + tidxj] += fbuf[2 * STRIDE_DIM + tidxi * CELL_SIZE + tidxj + CELL_SIZE/2];
    }
    /* 32 -> 16: 8x2 threads */
    if (tidxj < CELL_SIZE/4)
    {
        fbuf[                 tidxi * CELL_SIZE + tidxj] += fbuf[                 tidxi * CELL_SIZE + tidxj + CELL_SIZE/4];
        fbuf[    STRIDE_DIM + tidxi * CELL_SIZE + tidxj] += fbuf[    STRIDE_DIM + tidxi * CELL_SIZE + tidxj + CELL_SIZE/4];
        fbuf[2 * STRIDE_DIM + tidxi * CELL_SIZE + tidxj] += fbuf[2 * STRIDE_DIM + tidxi * CELL_SIZE + tidxj + CELL_SIZE/4];
    }
    /* 16 ->  8: 8 threads */
    if (tidxj < CELL_SIZE/8)
    {
        atomicAdd(&fout[ai].x, fbuf[                 tidxi * CELL_SIZE + tidxj] + fbuf[                 tidxi * CELL_SIZE + tidxj + CELL_SIZE/8]);
        atomicAdd(&fout[ai].y, fbuf[    STRIDE_DIM + tidxi * CELL_SIZE + tidxj] + fbuf[    STRIDE_DIM + tidxi * CELL_SIZE + tidxj + CELL_SIZE/8]);
        atomicAdd(&fout[ai].z, fbuf[2 * STRIDE_DIM + tidxi * CELL_SIZE + tidxj] + fbuf[2 * STRIDE_DIM + tidxi * CELL_SIZE + tidxj + CELL_SIZE/8]);
    }
}

inline __device__ void reduce_force_pow2_strided(float *fbuf, float4 *fout,
        int tidxi, int tidxj, int aidx)
{
    int i = CELL_SIZE/2;

    /* Reduce the initial CELL_SIZE values for each i atom to half
       every step by using CELL_SIZE * i threads. */
    # pragma unroll 5
    while (i > 1)
    {
        if (tidxj < i)
        {

            fbuf[                 tidxi * CELL_SIZE + tidxj] += fbuf[                 tidxi * CELL_SIZE + tidxj + i];
            fbuf[    STRIDE_DIM + tidxi * CELL_SIZE + tidxj] += fbuf[    STRIDE_DIM + tidxi * CELL_SIZE + tidxj + i];
            fbuf[2 * STRIDE_DIM + tidxi * CELL_SIZE + tidxj] += fbuf[2 * STRIDE_DIM + tidxi * CELL_SIZE + tidxj + i];
        }
        i >>= 1;
    }

    /* i == 1, last reduction step, writing to global mem */
    if (tidxj == 0)
    {
        atomicAdd(&fout[aidx].x,
            fbuf[                 tidxi * CELL_SIZE + tidxj] + fbuf[                 tidxi * CELL_SIZE + tidxj + i]);
        atomicAdd(&fout[aidx].y,
            fbuf[    STRIDE_DIM + tidxi * CELL_SIZE + tidxj] + fbuf[    STRIDE_DIM + tidxi * CELL_SIZE + tidxj + i]);
        atomicAdd(&fout[aidx].z,
            fbuf[2 * STRIDE_DIM + tidxi * CELL_SIZE + tidxj] + fbuf[2 * STRIDE_DIM + tidxi * CELL_SIZE + tidxj + i]);
    }
}

inline __device__ void reduce_force_strided(float *forcebuf, float4 *f,
        int tidxi, int tidxj, int ai)
{
    if ((CELL_SIZE & (CELL_SIZE - 1)))
    {
        // reduce_force_generic_strided(forcebuf, f, tidxi, tidxj, ai);
    }

    else
    {
        // reduce_force8_strided(forcebuf, f, tidxi, tidxj, ai);
        // reduce_force_pow2_strided(forcebuf, f, tidxi, tidxj, ai);
    }
}
#endif 
