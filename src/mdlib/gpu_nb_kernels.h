#include "cudatype_utils.h"

/* 32x32 */
__device__ void reduce_force32(float4 *fbuf, float4 *fout,
                             int tidxx, int tidxy, int ai) 
{
    /* 1024 -> 512: 32x16 threads */
    if (tidxy < CELL_SIZE/2)
    {
        fbuf[tidxx * CELL_SIZE + tidxy] += 
            fbuf[tidxx * CELL_SIZE + tidxy + CELL_SIZE/2];
    }
    /* 512 -> 256: 32x8 threads */
    if (tidxy < CELL_SIZE/4)
    {
        fbuf[tidxx * CELL_SIZE + tidxy] += 
            fbuf[tidxx * CELL_SIZE + tidxy + CELL_SIZE/4];
    }
    /* 256 -> 128: 32x4 threads */
    if (tidxy < CELL_SIZE/8)
    {
        fbuf[tidxx * CELL_SIZE + tidxy] += 
            fbuf[tidxx * CELL_SIZE + tidxy + CELL_SIZE/8];
    }
    /* 128 -> 64: 32x2 threads */
    if (tidxy < CELL_SIZE/16)
    {
        fbuf[tidxx * CELL_SIZE + tidxy] += 
            fbuf[tidxx * CELL_SIZE + tidxy + CELL_SIZE/16];
    }
    /* 64 -> 32: 32 threads */
    if (tidxy < CELL_SIZE/32) // tidxy == 0
    {
        fout[ai] = fbuf[tidxx * CELL_SIZE + tidxy] + 
            fbuf[tidxx * CELL_SIZE + tidxy + CELL_SIZE/32];
    }
}

/* 16x16 */
__device__ void reduce_force16(float4 *fbuf, float4 *fout,
                             int tidxx, int tidxy, int ai) 
{
    /* 256 -> 128: 16x8 threads */
    if (tidxy < CELL_SIZE/2)
    {
        fbuf[tidxx * CELL_SIZE + tidxy] += 
            fbuf[tidxx * CELL_SIZE + tidxy + CELL_SIZE/2];
    }
    /* 128 -> 64: 16x4 threads */
    if (tidxy < CELL_SIZE/4)
    {
        fbuf[tidxx * CELL_SIZE + tidxy] += 
            fbuf[tidxx * CELL_SIZE + tidxy + CELL_SIZE/4];
    }
    /* 64 -> 32: 16x2 threads */
    if (tidxy < CELL_SIZE/8)
    {
        fbuf[tidxx * CELL_SIZE + tidxy] += 
            fbuf[tidxx * CELL_SIZE + tidxy + CELL_SIZE/8];
    }
    /* 32 -> 16: 16 threads */   
    if (tidxy < CELL_SIZE/16)
    {
        fout[ai] = fbuf[tidxx * CELL_SIZE + tidxy] + 
            fbuf[tidxx * CELL_SIZE + tidxy + CELL_SIZE/16];
    }
}

/* 8x8 */
__device__ void reduce_force8(float4 *fbuf, float4 *fout,
                             int tidxx, int tidxy, int ai) 
{
    /* 64 -> 32: 8x4 threads */
    if (tidxy < CELL_SIZE/2)
    {
        fbuf[tidxx * CELL_SIZE + tidxy] += 
            fbuf[tidxx * CELL_SIZE + tidxy + CELL_SIZE/2];
    }
    /* 32 -> 16: 8x2 threads */
    if (tidxy < CELL_SIZE/4)
    {
        fbuf[tidxx * CELL_SIZE + tidxy] += 
            fbuf[tidxx * CELL_SIZE + tidxy + CELL_SIZE/4];
    }   
    /* 16 ->  8: 8 threads */   
    if (tidxy < CELL_SIZE/8)
    {
        fout[ai] = fbuf[tidxx * CELL_SIZE + tidxy] + 
            fbuf[tidxx * CELL_SIZE + tidxy + CELL_SIZE/8];
    }
}

__device__ void reduce_force_pow2(float4 *fbuf, float4 *fout,
                             int tidxx, int tidxy, int ai)
{
    int i = CELL_SIZE/2;
    
    /* Reduce the initial CELL_SIZE values for each i atom to half 
       every step by using CELL_SIZE * i threads. */
    while (i > 1)
    {
        if (tidxy < i)
        {

            fbuf[tidxx * CELL_SIZE + tidxy] += 
                fbuf[tidxx * CELL_SIZE + tidxy + i];
        }
        i >>= 1;
    }

    /* i == 1, last reduction step, writing to global mem */
    if (tidxy == 0)
    {
        fout[ai] = fbuf[tidxx * CELL_SIZE + tidxy] +
                fbuf[tidxx * CELL_SIZE + tidxy + i];
    }
}

/* FIXME for some @#$@#%! reason it does not work if in a separate function.
   Maybe it has something to do with the volatile business... */
__device__ void reduce_force_generic(float4 *fbuf, float4 *fout,
                             int tidxx, int tidxy, int ai)
{
    if (tidxy == 0)
    {      
        float4 f = make_float4(0.0f);
        # pragma unroll 24
        for (int j = tidxx * CELL_SIZE; j < (tidxx + 1) * CELL_SIZE; j++)
        {
            f.x += fbuf[j].x;
            f.y += fbuf[j].y;
            f.z += fbuf[j].z;
            // fbuf += forcebuf[i];
        } 
        fout[ai] = f;
    }
}

inline __device__ void reduce_force(float4 *fbuf, float4 *fout,
                             int tidxx, int tidxy, int ai) 
{
    if (CELL_SIZE & (CELL_SIZE - 1))
    {
        reduce_force_pow2(fbuf, fout, tidxx, tidxy, ai);
    }
    else 
    {
        reduce_force_generic(fbuf, fout, tidxx, tidxy, ai);
    }
}
/*  Launch parameterts: 
        - #blocks   = #neighbor lists, blockId = neigbor_listId
        - #threads  = CELL_SIZE^2
        - shmem     = CELL_SIZE^2 * sizeof(float4)
        - registers = 47

    Each thread calculates an i force-component taking one pair of i-j atoms.
*/
__global__ void k_calc_nb(const gmx_nbs_jlist_t *nblist, 
                          int ntypes, 
                          const float4 *xq, 
                          const int *atom_types, 
                          const int *cjlist, 
                          const float *nbfp,
                          float4 *f,
                          int cpg)
{
    unsigned int tidxx  = threadIdx.x; // local
    unsigned int tidxy  = threadIdx.y; 
    unsigned int tidx   = tidxx * blockDim.y + tidxy;
    unsigned int bidx   = blockIdx.x;

    int ci, cj, 
        ai, aj, 
        cij_start, cij_end, 
        typei, typej,
        j; //, ntypes;
    float /* qi, qj, */
          inv_r, inv_r2, inv_r6, 
          c6, c12, 
          dVdr;
    float4 f4tmp, fbuf;
    float3 xi, xj, rv;
    gmx_nbs_jlist_t nbl;

    extern __shared__ float4 forcebuf[];  // force buffer  

    for (int i = 0; i < cpg; i++)
    {
    //     nbl         = nblist[bidx];
    nbl         = nblist[cpg * bidx + i];
    ci          = nbl.ci; /* i cell index = current block index */
    ai          = ci * CELL_SIZE + tidxx;  /* i atom index */
    cij_start   = nbl.jind_start; /* first ...*/
    cij_end     = nbl.jind_end;  /* and last index of j cells */

    // load i atom data into registers
    f4tmp   = xq[ai];
    xi      = make_float3(f4tmp.x, f4tmp.y, f4tmp.z);
    /* qi      = f4tmp.w; */
    typei   = atom_types[ai];

    fbuf = make_float4(0.0f);
    // loop over i-s neighboring cells
    for (j = cij_start ; j < cij_end; j++)
    {        
        cj      = cjlist[j];
        aj      = cj * CELL_SIZE + tidxy;

        if (aj != ai)
        {
            // all threads load an atom from j cell into shmem!
            f4tmp   = xq[aj];
            xj      = make_float3(f4tmp.x, f4tmp.y, f4tmp.z);
            /* qj      = f4tmp.w; */ 
            typej   = atom_types[aj];
        
            /* TODO more uncool stuff here */
            c6          = nbfp[2 * (ntypes * typei + typej)]; // LJ C6 
                          // tex1Dfetch(texnbfp, 2 * (ntypes * typei + typej));
                          // 1; 
            c12         = nbfp[2 * (ntypes * typei + typej) + 1]; // LJ C12
                          // tex1Dfetch(texnbfp, 2 * (ntypes * typei + typej) + 1);                       
                          // 1;
            rv          = xi - xj;
            inv_r       = 1.0f / norm(rv);
            inv_r2      = inv_r * inv_r;
            inv_r6      = inv_r2 * inv_r2 * inv_r2;
            dVdr        = inv_r6 * (12.0 * c12 * inv_r6 - 6.0 * c6) * inv_r2;

            // accumulate forces            
            fbuf.x += rv.x * dVdr;
            fbuf.y += rv.y * dVdr;
            fbuf.z += rv.z * dVdr; 
        }
    }        
    forcebuf[tidx] = fbuf;
    __syncthreads();
    
    /* reduce forces */
#if 0 // CELL_SIZE != 8 && CELL_SIZE != 16 && CELL_SIZE != 32
    // XXX lame reduction 
    if (tidxy == 0)
    {      
        fbuf = make_float4(0.0f);
        # pragma unroll 24
        for (int j = tidxx * CELL_SIZE; j < (tidxx + 1) * CELL_SIZE; j++)
        {
            fbuf.x += forcebuf[j].x;
            fbuf.y += forcebuf[j].y;
            fbuf.z += forcebuf[j].z;
        }            
        f[ai] = fbuf; 
    }
#else
    reduce_force(forcebuf, f, tidxx, tidxy, ai); 
#endif 
    } /* for i */
}
    
#if 0
    if (tidxy == 0)
    {
        printf("Thread %5d(%2d, %2d) in block %d working on ci=%d, ai=%d; xi=(%5.1f, %5.1f, %5.1f ); cij=%4d-%4d \n",
                tidx, tidxx, tidxy, bidx, ci, ai, xi.x, xi.y, xi.z, cij_start, cij_end);
    }
#endif
   
__global__ void k_calc_nb1(struct cudata devData)
{
    t_cudata devD = &devData;    
    unsigned int tidxx  = threadIdx.x; // local
    unsigned int tidxy  = threadIdx.y; 
    unsigned int tidx   = tidxx * blockDim.y + tidxy;
    //unsigned int tpb    = gridDim.x * gridDim.y;
    unsigned int bidx   = blockIdx.x;

    int ci, cj, 
        ai, aj, 
        cij_start, cij_end, 
        typei, typej,
        j, ntypes;
    float /* qi, qj, */
          inv_r, inv_r2, inv_r6, 
          c6, c12, 
          dVdr;
    float4 f4tmp, fbuf;
    float3 xi, xj, rv;

    extern __shared__ float4 forcebuf[];  // force buffer  
    fbuf = make_float4(0.0f);

    ci          = devD->nblist[bidx].ci; /* i cell index = current block index */
    ai          = ci * CELL_SIZE + tidxx;  /* i atom index */
    cij_start   = devD->nblist[bidx].jind_start; /* first ...*/
    cij_end     = devD->nblist[bidx].jind_end;  /* and last index of j cells */
    ntypes      = devD->ntypes;

    // load i atom data into registers
    f4tmp   = devD->xq[ai];
    xi      = make_float3(f4tmp.x, f4tmp.y, f4tmp.z);
    /* qi      = f4tmp.w; */
    typei   = devD->atom_types[ai];
 
#if 0
    if (tidxy == 0)
    {
        printf("Thread %5d(%2d, %2d) in block %d working on ci=%d, ai=%d; xi=(%5.1f, %5.1f, %5.1f ); cij=%4d-%4d \n",
                tidx, tidxx, tidxy, bidx, ci, ai, xi.x, xi.y, xi.z, cij_start, cij_end);
    }
#endif
   
    // loop over i-s neighboring cells
    for (j = cij_start ; j < cij_end; j++)
    {        
        cj      = devD->cj[j]; // TODO quite uncool operation
        aj      = cj * CELL_SIZE + tidxy;

        if (aj != ai)
        {
            // all threads load an atom from j cell into shmem!
            f4tmp   = devD->xq[aj];
            xj      = make_float3(f4tmp.x, f4tmp.y, f4tmp.z);
            /* qj      = f4tmp.w; */ 
            typej   = devD->atom_types[aj];
        
            /* TODO more uncool stuff here */
            c6          =  devD->nbfp[2 * (ntypes * typei + typej)]; // LJ C6 
            c12         =  devD->nbfp[2 * (ntypes * typei + typej) + 1]; // LJ C12
            rv          = xi - xj;
            inv_r       = 1.0f / norm(rv);
            inv_r2      = inv_r * inv_r;
            inv_r6      = inv_r2 * inv_r2 * inv_r2;
            dVdr        = inv_r6 * (12.0 * c12 * inv_r6 - 6.0 * c6) * inv_r2;
#if 0
        if (tidxy == 0 /*&& norm(rv) < 10.01*/)
        {
            printf("%d %d | %g %g %g\n", ai, aj, rv.x,rv.y, rv.z);
        }
#endif 
            // accumulate forces            
            fbuf.x += rv.x * dVdr;
            fbuf.y += rv.y * dVdr;
            fbuf.z += rv.z * dVdr; 
        }
    }        

    forcebuf[tidx] = fbuf;
    __syncthreads();

    /* reduce forces */
#if 0
    // XXX lame reduction 
    if (tidxy == 0)
    {      
        fbuf = make_float4(0.0f);
        # pragma unroll 24
        for (j = tidxx * CELL_SIZE; j < (tidxx + 1) * CELL_SIZE; j++)
        // for (int i = tidxx * CELL_SIZE; i < (tidxx ) * CELL_SIZE + 1; i++)
        {
            fbuf.x += forcebuf[j].x;
            fbuf.y += forcebuf[j].y;
            fbuf.z += forcebuf[j].z;
            // fbuf += forcebuf[i];
        }            
        devD->f[ai] = fbuf; 
    }
#else
    reduce_force(forcebuf, devD->f, tidxx, tidxy, ai); 
#endif
}

__global__ void k_calc_bboxes()
{

}
