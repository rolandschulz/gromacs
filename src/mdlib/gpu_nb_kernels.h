#include "cudatype_utils.h"


__device__ void reduce_force(volatile float4 *fbuf, float4 *fout,
                             int tidxx, int tidxy, int ai) 
{

}

/*  Launch parameterts: 
        - #bloks    = 
        - #threads  = #pairs of atoms in cell i and j = |Ci| x |Cj| 
        - shmem     
            - EXT - for i atoms 
            - for all j atoms 
        - registers  =
    Each thread calculated one  pair of atoms from neigbouring i-j cells.
*/
__global__ void k_calc_nb(struct cudata devData)
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
        nbidx, ntypes;
    float qi, qj, 
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
    qi      = f4tmp.w;
    typei   = devD->atom_types[ai];
 
#if 0
    if (tidxy == 0)
    {
        printf("Thread %5d(%2d, %2d) in block %d working on ci=%d, ai=%d; xi=(%5.1f, %5.1f, %5.1f ); cij=%4d-%4d \n",
                tidx, tidxx, tidxy, bidx, ci, ai, xi.x, xi.y, xi.z, cij_start, cij_end);
    }
#endif
   
    // loop over i-s neighboring cells
    // If not, cell_nblist_idx should be loaded into constant mem
    for (nbidx = cij_start ; nbidx < cij_end; nbidx++)
    {        
        cj      = devD->cj[nbidx]; // TODO quite uncool operation
        aj      = cj * CELL_SIZE + tidxy;

        if (aj != ai)
        {

            // all threads load an atom from j cell into shmem!
            f4tmp   = devD->xq[aj];
            xj      = make_float3(f4tmp.x, f4tmp.y, f4tmp.z);
            qj      = f4tmp.w;
            typej   = devD->atom_types[aj];
        
            /* TODO more uncool stuff here */
            c6          =  // 1; 
                devD->nbfp[2 * (ntypes * typei + typej)]; // LJ C6 
            c12         =  // 1; 
                 devD->nbfp[2 * (ntypes * typei + typej) + 1]; // LJ C12
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
            /*
            float ff = norm(rv);
            fbuf.x = fmin(fbuf.x, ff); // rv.x * dVdr;
            fbuf.y = fmin(fbuf.y, fabs(rv.y));
            fbuf.z = fmin(fbuf.z, fabs(rv.z));           
            */
            
        }

    }        

    forcebuf[tidx] = fbuf;
    __syncthreads();

    /* reduce forces */
#if 1
    // XXX lame reduction 
    if (tidxy == 0)
    {      
        fbuf = make_float4(0.0f);
        # pragma unroll 32
        for (int i = tidxx * CELL_SIZE; i < (tidxx + 1) * CELL_SIZE; i++)
        // for (int i = tidxx * CELL_SIZE; i < (tidxx ) * CELL_SIZE + 1; i++)
        {

            fbuf.x += forcebuf[i].x;
            fbuf.y += forcebuf[i].y;
            fbuf.z += forcebuf[i].z;

           // fbuf += forcebuf[i];
        }            
        devD->f[ai] = fbuf; 
    }
#endif 
 
# if 0
/* 1024 -> 512: 32x16 threads */
if (tidxy < CELL_SIZE/2)
{        
    forcebuf[tidxx * CELL_SIZE + tidxy] += 
        forcebuf[tidxx * CELL_SIZE + tidxy + CELL_SIZE/2];
}
/* 512 -> 256: 32x8 threads */
if (tidxy < CELL_SIZE/4)
{        
    forcebuf[tidxx * CELL_SIZE + tidxy] += 
        forcebuf[tidxx * CELL_SIZE + tidxy + CELL_SIZE/4];
}
/* 256 -> 128: 32x4 threads */
if (tidxy < CELL_SIZE/8)
{        
    forcebuf[tidxx * CELL_SIZE + tidxy] += 
        forcebuf[tidxx * CELL_SIZE + tidxy + CELL_SIZE/8];
}
/* 128 -> 64: 32x2 threads */
if (tidxy < CELL_SIZE/16)
{        
    forcebuf[tidxx * CELL_SIZE + tidxy] += 
        forcebuf[tidxx * CELL_SIZE + tidxy + CELL_SIZE/16];
}
/* 64 -> 32: 32 threads */
if (tidxy < CELL_SIZE/32) // tidxy == 0
{
    devD->f[ai] = forcebuf[tidxx * CELL_SIZE + tidxy] + 
        forcebuf[tidxx * CELL_SIZE + tidxy + CELL_SIZE/32];
}
#endif 

  

}

__global__ void k_calc_bboxes()
{

}
