/*  Launch parameterts:
    - #blocks   = #neighbor lists, blockId = neigbor_listId
    - #threads  = CELL_SIZE^2
    - shmem     = (1 + NSUBCELL) * CELL_SIZE^2 * 3 * sizeof(float)
    - registers = 44

    Each thread calculates an i force-component taking one pair of i-j atoms.
 */
__global__ void FUNCTION_NAME(k_calc_nb, forces_1)(
                            const gmx_nbl_ci_t *nbl_ci,
                            const gmx_nbl_sj_t *nbl_sj,
                            const gmx_nbl_si_t *nbl_si,
                            const int *atom_types,
                            int ntypes,
                            const float4 *xq,
                            const float *nbfp,
                            const float3 *shift_vec,
                            float two_krf,
                            float cutoff_sq,
                            float erfc_tab_scale,
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
        i, j; 
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

    shift       = shift_vec[nb_ci.shift];

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

            // all threads load an atom from i cell into shmem!
            f4tmp   = xq[ai];
            xi      = make_float3(f4tmp.x, f4tmp.y, f4tmp.z);
            qi      = f4tmp.w;
            typei   = atom_types[ai];

            c6          = nbfp[2 * (ntypes * typei + typej)]; // LJ C6
                          // 1e-3;
            c12         = nbfp[2 * (ntypes * typei + typej) + 1]; // LJ C12
                          // 1e-3;

            rv          = xi - xj;
            r2          = norm2(rv);

            if (r2 < cutoff_sq * excl_bit)
            {
                inv_r       = 1.0f / sqrt(r2);
                inv_r2      = inv_r * inv_r;
                inv_r6      = inv_r2 * inv_r2 * inv_r2;

                dVdr        = inv_r6 * (12.0 * c12 * inv_r6 - 6.0 * c6) * inv_r2;

#ifdef EL_CUTOFF
                dVdr        += qi * qj_f * inv_r2 * inv_r;  
#endif
#ifdef EL_RF
                dVdr        += qi * qj_f * (inv_r2 * inv_r - two_krf); 
#endif
#ifdef EL_EWALD
                dVdr        += qi * qj_f * interpolate_coulomb_force_r(r2 * inv_r, erfc_tab_scale);
#endif

                f_ij = rv * dVdr;

                // accumulate j forces
                fsj_buf -= f_ij;

                // accumulate i forces
                forcebuf[                 (1 + si_offset) * STRIDE_SI + tidx] += f_ij.x;
                forcebuf[    STRIDE_DIM + (1 + si_offset) * STRIDE_SI + tidx] += f_ij.y;
                forcebuf[2 * STRIDE_DIM + (1 + si_offset) * STRIDE_SI + tidx] += f_ij.z;
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
    for(si_offset = 0; si_offset < NSUBCELL; si_offset++)
    {
        ai  = (ci * NSUBCELL + si_offset) * CELL_SIZE + tidxx;  /* i atom index */
        reduce_force_i_generic_strided(forcebuf + (1 + si_offset) * STRIDE_SI, f, tidxx, tidxy, ai);
    }
    __syncthreads();
}


/*  Launch parameterts:
    - #blocks   = #neighbor lists, blockId = neigbor_listId
    - #threads  = CELL_SIZE^2
    - shmem     = CELL_SIZE^2 * 3 * sizeof(float)
    - registers = 45
    - local mem = 4 bytes !!! 

    Each thread calculates an i force-component taking one pair of i-j atoms.
 */
__global__ void FUNCTION_NAME(k_calc_nb, forces_2)(
                            const gmx_nbl_ci_t *nbl_ci,
                            const gmx_nbl_sj_t *nbl_sj,
                            const gmx_nbl_si_t *nbl_si,
                            const int *atom_types,
                            int ntypes,
                            const float4 *xq,
                            const float *nbfp,
                            const float3 *shift_vec,
                            float two_krf,
                            float cutoff_sq,       
                            float erfc_tab_scale,
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
        i, j; 
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

    shift       = shift_vec[nb_ci.shift];

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

            // all threads load an atom from i cell into shmem!
            f4tmp   = xq[ai];
            xi      = make_float3(f4tmp.x, f4tmp.y, f4tmp.z);
            qi      = f4tmp.w;
            typei   = atom_types[ai];

            c6  = nbfp[2 * (ntypes * typei + typej)]; // LJ C6
                    // 1e-3;
            c12 = nbfp[2 * (ntypes * typei + typej) + 1]; // LJ C12
                    // 1e-3;

            rv          = xi - xj;
            r2          = norm2(rv);

            if (r2 < cutoff_sq * excl_bit)
            {
                inv_r       = 1.0f / sqrt(r2);
                inv_r2      = inv_r * inv_r;
                inv_r6      = inv_r2 * inv_r2 * inv_r2;

                dVdr        = inv_r6 * (12.0 * c12 * inv_r6 - 6.0 * c6) * inv_r2;

#ifdef EL_CUTOFF
                dVdr        += qi * qj_f * inv_r2 * inv_r;  
#endif
#ifdef EL_RF
                dVdr        += qi * qj_f * (inv_r2 * inv_r - two_krf); 
#endif
#ifdef EL_EWALD
                dVdr        += qi * qj_f * interpolate_coulomb_force_r(r2 * inv_r, erfc_tab_scale);
#endif

                f_ij = rv * dVdr;

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
    __syncthreads();
}

#undef EL_CUTOFF
#undef EL_RF
#undef EL_EWALD
