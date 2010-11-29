#include "cudatype_utils.h"

#define CELL_SIZE_2     (CELL_SIZE * CELL_SIZE)
#define STRIDE_DIM      (CELL_SIZE_2)
#define STRIDE_SI       (3*STRIDE_DIM)

/* 8x8 */
__device__ void reduce_force8_strided(float *fbuf, float4 *fout,
        int tidxx, int tidxy, int ai)
{
    /* 64 -> 32: 8x4 threads */
    if (tidxy < CELL_SIZE/2)
    {
        fbuf[                 tidxx * CELL_SIZE + tidxy] += fbuf[                 tidxx * CELL_SIZE + tidxy + CELL_SIZE/2];
        fbuf[    STRIDE_DIM + tidxx * CELL_SIZE + tidxy] += fbuf[    STRIDE_DIM + tidxx * CELL_SIZE + tidxy + CELL_SIZE/2];
        fbuf[2 * STRIDE_DIM + tidxx * CELL_SIZE + tidxy] += fbuf[2 * STRIDE_DIM + tidxx * CELL_SIZE + tidxy + CELL_SIZE/2];
    }
    /* 32 -> 16: 8x2 threads */
    if (tidxy < CELL_SIZE/4)
    {
        fbuf[                 tidxx * CELL_SIZE + tidxy] += fbuf[                 tidxx * CELL_SIZE + tidxy + CELL_SIZE/4];
        fbuf[    STRIDE_DIM + tidxx * CELL_SIZE + tidxy] += fbuf[    STRIDE_DIM + tidxx * CELL_SIZE + tidxy + CELL_SIZE/4];
        fbuf[2 * STRIDE_DIM + tidxx * CELL_SIZE + tidxy] += fbuf[2 * STRIDE_DIM + tidxx * CELL_SIZE + tidxy + CELL_SIZE/4];
    }
    /* 16 ->  8: 8 threads */
    if (tidxy < CELL_SIZE/8)
    {
        atomicAdd(&fout[ai].x, fbuf[                 tidxx * CELL_SIZE + tidxy] + fbuf[                 tidxx * CELL_SIZE + tidxy + CELL_SIZE/8]);
        atomicAdd(&fout[ai].y, fbuf[    STRIDE_DIM + tidxx * CELL_SIZE + tidxy] + fbuf[    STRIDE_DIM + tidxx * CELL_SIZE + tidxy + CELL_SIZE/8]);
        atomicAdd(&fout[ai].z, fbuf[2 * STRIDE_DIM + tidxx * CELL_SIZE + tidxy] + fbuf[2 * STRIDE_DIM + tidxx * CELL_SIZE + tidxy + CELL_SIZE/8]);
    }
}

inline __device__ void reduce_force_pow2_strided(float *fbuf, float4 *fout,
        int tidxx, int tidxy, int aidx)
{
    int i = CELL_SIZE/2;

    /* Reduce the initial CELL_SIZE values for each i atom to half
       every step by using CELL_SIZE * i threads. */
    # pragma unroll 5
    while (i > 1)
    {
        if (tidxy < i)
        {

            fbuf[                 tidxx * CELL_SIZE + tidxy] += fbuf[                 tidxx * CELL_SIZE + tidxy + i];
            fbuf[    STRIDE_DIM + tidxx * CELL_SIZE + tidxy] += fbuf[    STRIDE_DIM + tidxx * CELL_SIZE + tidxy + i];
            fbuf[2 * STRIDE_DIM + tidxx * CELL_SIZE + tidxy] += fbuf[2 * STRIDE_DIM + tidxx * CELL_SIZE + tidxy + i];
        }
        i >>= 1;
    }

    /* i == 1, last reduction step, writing to global mem */
    if (tidxy == 0)
    {
        atomicAdd(&fout[aidx].x,
            fbuf[                 tidxx * CELL_SIZE + tidxy] + fbuf[                 tidxx * CELL_SIZE + tidxy + i]);
        atomicAdd(&fout[aidx].y,
            fbuf[    STRIDE_DIM + tidxx * CELL_SIZE + tidxy] + fbuf[    STRIDE_DIM + tidxx * CELL_SIZE + tidxy + i]);
        atomicAdd(&fout[aidx].z,
            fbuf[2 * STRIDE_DIM + tidxx * CELL_SIZE + tidxy] + fbuf[2 * STRIDE_DIM + tidxx * CELL_SIZE + tidxy + i]);
    }
}

inline __device__ void reduce_force_strided(float *forcebuf, float4 *f,
        int tidxx, int tidxy, int ai)
{
    if ((CELL_SIZE & (CELL_SIZE - 1)))
    {
        // reduce_force_generic_strided(forcebuf, f, tidxx, tidxy, ai);
    }

    else
    {
        // reduce_force8_strided(forcebuf, f, tidxx, tidxy, ai);
        // reduce_force_pow2_strided(forcebuf, f, tidxx, tidxy, ai);
    }
}



inline __device__ void reduce_force_i_generic_strided(float *fbuf, float4 *fout,
        int tidxx, int tidxy, int aidx)
{
    if (tidxy == 0)
    {
        float4 f = make_float4(0.0f);
        for (int j = tidxx * CELL_SIZE; j < (tidxx + 1) * CELL_SIZE; j++)
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
        int tidxx, int tidxy, int aidx)
{
    if (tidxx == 0)
    {
        float4 f = make_float4(0.0f);
        for (int j = tidxy; j < CELL_SIZE_2; j += CELL_SIZE)
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


/*  Launch parameterts:
    - #blocks   = #neighbor lists, blockId = neigbor_listId
    - #threads  = CELL_SIZE^2
    - shmem     = (1 + NSUBCELL) * CELL_SIZE^2 * sizeof(float4)
    - registers = 40/44

    Each thread calculates an i force-component taking one pair of i-j atoms.
 */
__global__ void k_calc_nb_1(const gmx_nbl_ci_t *nbl_ci,
                            const gmx_nbl_sj_t *nbl_sj,
                            const gmx_nbl_si_t *nbl_si,
                            const int *atom_types,
                            int ntypes,
                            const float4 *xq,
                            const float *nbfp,
                            const float3 *shiftvec,
                            float4 *f)
{
    unsigned int tidxx  = threadIdx.y;
    unsigned int tidxy  = threadIdx.x;
    unsigned int tidx   = tidxx * blockDim.y + tidxy;
    unsigned int bidx   = blockIdx.x;

    int ci, si, sj, si_offset,
        ai, aj,
        cij_start, cij_end,
        si_start, si_end,
        typei, typej,
        i, j; //, ntypes;
    float qi, qj_f,
          r2, inv_r, inv_r2, inv_r6,
          c6, c12,
          dVdr;
    float4  f4tmp;
    float3  xi, xj, rv;
    float3  shift;
    float3  f_ij, fsj_buf;
    gmx_nbl_ci_t    nb_ci;
    unsigned int excl_bit;

    extern __shared__ float forcebuf[];  // force buffer

    nb_ci       = nbl_ci[bidx];
    ci          = nb_ci.ci; /* i cell index = current block index */
    cij_start   = nb_ci.sj_ind_start; /* first ...*/
    cij_end     = nb_ci.sj_ind_end;  /* and last index of j cells */

    shift       = shiftvec[nb_ci.shift];

    for(si_offset = 0; si_offset < NSUBCELL; si_offset++)
    {
        forcebuf[                 (1 + si_offset) * STRIDE_SI + tidx] = 0.0;
        forcebuf[    STRIDE_DIM + (1 + si_offset) * STRIDE_SI + tidx] = 0.0;
        forcebuf[2 * STRIDE_DIM + (1 + si_offset) * STRIDE_SI + tidx] = 0.0;
    }

    // loop over i-s neighboring cells
    for (j = cij_start ; j < cij_end; j++)
    {
        sj          = nbl_sj[j].sj; /* TODO int4? */
        si_start    = nbl_sj[j].si_ind;
        si_end      = nbl_sj[j + 1].si_ind;
        aj          = sj * CELL_SIZE + tidxy;

        /* load j atom data into registers */
        f4tmp   = xq[aj];
        xj      = make_float3(f4tmp.x, f4tmp.y, f4tmp.z);
        qj_f    = GPU_FACEL * f4tmp.w;
        typej   = atom_types[aj];
        xj      -= shift;

        fsj_buf = make_float3(0.0);

        for (i = si_start; i < si_end; i++)
        {
            si = nbl_si[i].si;
            si_offset = si - ci * NSUBCELL;
            ai  = si * CELL_SIZE + tidxx;  /* i atom index */

            excl_bit = (nbl_si[i].excl >> tidx) & 1;

            // if (aj != ai)
            {
                // all threads load an atom from i cell into shmem!
                f4tmp   = xq[ai];
                xi      = make_float3(f4tmp.x, f4tmp.y, f4tmp.z);
                qi      = f4tmp.w;
                typei   = atom_types[ai];

                c6          = nbfp[2 * (ntypes * typei + typej)]; // LJ C6
                c12         = nbfp[2 * (ntypes * typei + typej) + 1]; // LJ C12
                rv          = xi - xj;
                r2          = norm2(rv);
                inv_r       = 1.0f / sqrt(r2);
                inv_r2      = inv_r * inv_r;
                inv_r6      = inv_r2 * inv_r2 * inv_r2;

                dVdr        = qi * qj_f * inv_r2 * inv_r;
                // qi_f * qj * (erfc(r2 * inv_r) * inv_r + exp(-r2)) * inv_r2;
                dVdr        += inv_r6 * (12.0 * c12 * inv_r6 - 6.0 * c6) * inv_r2;

                f_ij = excl_bit * rv * dVdr;

                // accumulate j forces
                fsj_buf -= f_ij;

                // accumulate i forces
                // forcebuf[tidx + (1 + si_offset) * CELL_SIZE * CELL_SIZE] += f_ij;
                forcebuf[                 (1 + si_offset) * STRIDE_SI + tidx] += f_ij.x;
                forcebuf[    STRIDE_DIM + (1 + si_offset) * STRIDE_SI + tidx] += f_ij.y;
                forcebuf[2 * STRIDE_DIM + (1 + si_offset) * STRIDE_SI + tidx] += f_ij.z;
            }
        }
        /* store j forces in shmem */
        // forcebuf[tidx] = fsj_buf;
        forcebuf[                 tidx] = fsj_buf.x;
        forcebuf[    STRIDE_DIM + tidx] = fsj_buf.y;
        forcebuf[2 * STRIDE_DIM + tidx] = fsj_buf.z;
        __syncthreads();

        /* reduce j forces */
        reduce_force_j_generic_strided(forcebuf, f, tidxx, tidxy, aj);
    }
    __syncthreads();

    /* reduce i forces */
    for(si_offset = 0; si_offset < NSUBCELL; si_offset++)
    {
        ai  = (ci * NSUBCELL + si_offset) * CELL_SIZE + tidxx;  /* i atom index */
        reduce_force_i_generic_strided(forcebuf + (1 + si_offset) * STRIDE_SI, f, tidxx, tidxy, ai);
    }
}


/*  Launch parameterts:
    - #blocks   = #neighbor lists, blockId = neigbor_listId
    - #threads  = CELL_SIZE^2
    - shmem     = (1 + NSUBCELL) * CELL_SIZE^2 * sizeof(float4)
    - registers = 40/44

    Each thread calculates an i force-component taking one pair of i-j atoms.
 */
__global__ void k_calc_nb_2(const gmx_nbl_ci_t *nbl_ci,
                            const gmx_nbl_sj_t *nbl_sj,
                            const gmx_nbl_si_t *nbl_si,
                            const int *atom_types,
                            int ntypes,
                            const float4 *xq,
                            const float *nbfp,
                            const float3 *shiftvec,
                            float4 *f)
{
    unsigned int tidxx  = threadIdx.y;
    unsigned int tidxy  = threadIdx.x;
    unsigned int tidx   = tidxx * blockDim.y + tidxy;
    unsigned int bidx   = blockIdx.x;

    int ci, si, sj, si_offset,
        ai, aj,
        cij_start, cij_end,
        si_start, si_end,
        typei, typej,
        i, j; //, ntypes;
    float qi, qj_f,
          r2, inv_r, inv_r2, inv_r6,
          c6, c12,
          dVdr;
    float4 f4tmp;
    float3 xi, xj, rv;
    float3 shift;
    float3 f_ij, fsj_buf;
    gmx_nbl_ci_t nb_ci;
    unsigned int excl_bit;

    extern __shared__ float forcebuf[];  // force buffer
    float3 fsi_buf[NSUBCELL];

    nb_ci       = nbl_ci[bidx];
    ci          = nb_ci.ci; /* i cell index = current block index */
    cij_start   = nb_ci.sj_ind_start; /* first ...*/
    cij_end     = nb_ci.sj_ind_end;  /* and last index of j cells */

    shift       = shiftvec[nb_ci.shift];

    for(si_offset = 0; si_offset < NSUBCELL; si_offset++)
    {
        fsi_buf[si_offset] = make_float3(0.0);
    }

    // loop over i-s neighboring cells
    for (j = cij_start ; j < cij_end; j++)
    {
        sj          = nbl_sj[j].sj; /* TODO int4? */
        si_start    = nbl_sj[j].si_ind;
        si_end      = nbl_sj[j + 1].si_ind;
        aj          = sj * CELL_SIZE + tidxy;

        /* load j atom data into registers */
        f4tmp   = xq[aj];
        xj      = make_float3(f4tmp.x, f4tmp.y, f4tmp.z);
        qj_f    = GPU_FACEL * f4tmp.w;
        typej   = atom_types[aj];
        xj      -= shift;

        fsj_buf = make_float3(0.0);

        for (i = si_start; i < si_end; i++)
        {
            si = nbl_si[i].si;
            si_offset = si - ci * NSUBCELL;
            ai  = si * CELL_SIZE + tidxx;  /* i atom index */

            excl_bit = (nbl_si[i].excl >> tidx) & 1;

            // if (aj > ai)
            {
                // all threads load an atom from i cell into shmem!
                f4tmp   = xq[ai];
                xi      = make_float3(f4tmp.x, f4tmp.y, f4tmp.z);
                qi      = f4tmp.w;
                typei   = atom_types[ai];

                c6          = nbfp[2 * (ntypes * typei + typej)]; // LJ C6
                c12         = nbfp[2 * (ntypes * typei + typej) + 1]; // LJ C12
                rv          = xi - xj;
                r2          = norm2(rv);
                inv_r       = 1.0f / sqrt(r2);
                inv_r2      = inv_r * inv_r;
                inv_r6      = inv_r2 * inv_r2 * inv_r2;

                dVdr        = qi * qj_f * inv_r2 * inv_r;
                // qi_f * qj * (erfc(r2 * inv_r) * inv_r + exp(-r2)) * inv_r2;
                dVdr        += inv_r6 * (12.0 * c12 * inv_r6 - 6.0 * c6) * inv_r2;

                f_ij = excl_bit * rv * dVdr;

                // accumulate j forces
                fsj_buf -= f_ij;

                // accumulate i forces
                fsi_buf[si_offset] += f_ij;
            }
        }
        /* store j forces in shmem */
        forcebuf[                 tidx] = fsj_buf.x;
        forcebuf[    STRIDE_DIM + tidx] = fsj_buf.y;
        forcebuf[2 * STRIDE_DIM + tidx] = fsj_buf.z;
        __syncthreads();

        /* reduce j forces */
        reduce_force_j_generic_strided(forcebuf, f, tidxx, tidxy, aj);
    }
    __syncthreads();

    /* reduce i forces */
    for (si_offset = 0; si_offset < NSUBCELL; si_offset++)
    {
        ai  = (ci * NSUBCELL + si_offset) * CELL_SIZE + tidxx;
        forcebuf[                 tidx] = fsi_buf[si_offset].x;
        forcebuf[    STRIDE_DIM + tidx] = fsi_buf[si_offset].y;
        forcebuf[2 * STRIDE_DIM + tidx] = fsi_buf[si_offset].z;
        reduce_force_i_generic_strided(forcebuf, f, tidxx, tidxy, ai);

        __syncthreads();
    }
}
