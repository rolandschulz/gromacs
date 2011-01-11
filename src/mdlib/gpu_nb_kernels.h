/*  Launch parameterts:
    - #blocks   = #neighbor lists, blockId = neigbor_listId
    - #threads  = CELL_SIZE^2
    - shmem     = (1 + NSUBCELL) * CELL_SIZE^2 * 3 * sizeof(float)
    - registers = 44

    Each thread calculates an i force-component taking one pair of i-j atoms.
 */
/*
TODO:
  - fix GPU_FACEL
  - improve energy reduction!
 */

#ifdef CALC_ENERGIES                           
__global__ void FUNCTION_NAME(k_calc_nb, forces_energies_1)
#else
__global__ void FUNCTION_NAME(k_calc_nb, forces_1)
#endif
                           (const gmx_nbl_ci_t *nbl_ci,
                            const gmx_nbl_sj_t *nbl_sj,
                            const gmx_nbl_si_t *nbl_si,
                            const int *atom_types,
                            int ntypes,
                            const float4 *xq,
                            const float *nbfp,
                            const float3 *shift_vec,
                            float two_k_rf,
                            float cutoff_sq,
                            float coulomb_tab_scale,
#ifdef CALC_ENERGIES
                            float beta,
                            float c_rf,
                            float *e_lj,
                            float *e_el,
#endif
                            float4 *f)
{    
    unsigned int tidxi  = threadIdx.x;
    unsigned int tidxj  = threadIdx.y;
    unsigned int tidx   = threadIdx.y * blockDim.x + threadIdx.x;
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
#ifdef CALC_ENERGIES
          E_lj, E_el,
#endif                       
          F_invr;
    float4  f4tmp;
    float3  xi, xj, rv;
    float3  shift;
    float3  f_ij, fsj_buf;
    gmx_nbl_ci_t nb_ci;
    unsigned int excl_bit;

    extern __shared__ float forcebuf[]; /* force buffer */

    nb_ci       = nbl_ci[bidx];         /* cell index */
    ci          = nb_ci.ci;             /* i cell index = current block index */
    cij_start   = nb_ci.sj_ind_start;   /* first ...*/
    cij_end     = nb_ci.sj_ind_end;     /* and last index of j cells */

    shift       = shift_vec[nb_ci.shift];

    for(si_offset = 0; si_offset < NSUBCELL; si_offset++)
    {
        forcebuf[                 (1 + si_offset) * STRIDE_SI + tidx] = 0.0f;
        forcebuf[    STRIDE_DIM + (1 + si_offset) * STRIDE_SI + tidx] = 0.0f;
        forcebuf[2 * STRIDE_DIM + (1 + si_offset) * STRIDE_SI + tidx] = 0.0f;
    }

#ifdef CALC_ENERGIES
    E_lj = 0.0f;
    E_el = 0.0f;
#endif

    /* loop over the j sub-cells = seen by any of the atoms in the current cell */
    for (j = cij_start ; j < cij_end; j++)
    {
        sj          = nbl_sj[j].sj; /* TODO int4? */
        si_start    = nbl_sj[j].si_ind;
        si_end      = nbl_sj[j + 1].si_ind;
        aj          = sj * CELL_SIZE + tidxj;

        /* load j atom data into registers */
        f4tmp   = xq[aj];
        xj      = make_float3(f4tmp.x, f4tmp.y, f4tmp.z);
        qj_f    = GPU_FACEL * f4tmp.w;
        typej   = atom_types[aj];
        xj      -= shift;

        fsj_buf = make_float3(0.0f);

        /* loop over i sub-cells in ci */
        for (i = si_start; i < si_end; i++)
        {
            si          = nbl_si[i].si;
            si_offset   = si - ci * NSUBCELL;       /* i force buffer offset */ 
            ai          = si * CELL_SIZE + tidxi;  /* i atom index */

            excl_bit = (nbl_si[i].excl >> tidx) & 1;

            /* all threads load an atom from i cell si into shmem! */
            f4tmp   = xq[ai];
            xi      = make_float3(f4tmp.x, f4tmp.y, f4tmp.z);
            qi      = f4tmp.w;
            typei   = atom_types[ai];

            /* LJ C6 and C12 */
            c6      = tex1Dfetch(tex_nbfp, 2 * (ntypes * typei + typej));
            c12     = tex1Dfetch(tex_nbfp, 2 * (ntypes * typei + typej) + 1);

            rv      = xi - xj;
            r2      = norm2(rv);

            /* cutoff & exclusion check */
            if (r2 < cutoff_sq * excl_bit)
            {
                inv_r       = 1.0f / sqrt(r2);
                inv_r2      = inv_r * inv_r;
                inv_r6      = inv_r2 * inv_r2 * inv_r2;

                F_invr      = inv_r6 * (12.0f * c12 * inv_r6 - 6.0f * c6) * inv_r2;

#ifdef CALC_ENERGIES
                E_lj        += inv_r6 * (c12 * inv_r6 - c6);
#endif

#ifdef EL_CUTOFF
                F_invr      += qi * qj_f * inv_r2 * inv_r;
#endif
#ifdef EL_RF
                F_invr      += qi * qj_f * (inv_r2 * inv_r - two_k_rf);
#endif
#ifdef EL_EWALD
                F_invr      += qi * qj_f * interpolate_coulomb_force_r(r2 * inv_r, coulomb_tab_scale);
#endif
                
#ifdef CALC_ENERGIES
#ifdef EL_CUTOFF
                E_el        += qi * qj_f * inv_r;           
#endif
#ifdef EL_RF
                E_el        += qi * qj_f * (inv_r + 0.5f * two_k_rf * r2 - c_rf);
#endif
#ifdef EL_EWALD
                E_el        += qi * qj_f * inv_r * erfc(inv_r * beta);
#endif
#endif
                f_ij    = rv * F_invr;

                /* accumulate j forces in registers */
                fsj_buf -= f_ij;

                /* accumulate i forces in shmem */
                forcebuf[                 (1 + si_offset) * STRIDE_SI + tidx] += f_ij.x;
                forcebuf[    STRIDE_DIM + (1 + si_offset) * STRIDE_SI + tidx] += f_ij.y;
                forcebuf[2 * STRIDE_DIM + (1 + si_offset) * STRIDE_SI + tidx] += f_ij.z;
            }
        }

        /* store j forces in shmem */
        forcebuf[                 tidx] = fsj_buf.x;
        forcebuf[    STRIDE_DIM + tidx] = fsj_buf.y;
        forcebuf[2 * STRIDE_DIM + tidx] = fsj_buf.z;

        /* reduce j forces */
        reduce_force_j_generic_strided(forcebuf, f, tidxi, tidxj, aj);
    }    
    __syncthreads();

    /* reduce i forces */
    for(si_offset = 0; si_offset < NSUBCELL; si_offset++)
    {
        ai  = (ci * NSUBCELL + si_offset) * CELL_SIZE + tidxi;  /* i atom index */
        reduce_force_i_strided(forcebuf + (1 + si_offset) * STRIDE_SI, f, tidxi, tidxj, ai);
    }

#ifdef CALC_ENERGIES
    /* add each thread's local energy to the global value */
    atomicAdd(e_lj, E_lj);
    atomicAdd(e_el, E_el);
#endif
}


/*  Launch parameterts:
    - #blocks   = #neighbor lists, blockId = neigbor_listId
    - #threads  = CELL_SIZE^2
    - shmem     = CELL_SIZE^2 * 3 * sizeof(float)
    - registers = 45
    - local mem = 4 bytes !!! 

    Each thread calculates an i force-component taking one pair of i-j atoms.
 */
#ifdef CALC_ENERGIES                           
__global__ void FUNCTION_NAME(k_calc_nb, forces_energies_2)
#else
__global__ void FUNCTION_NAME(k_calc_nb, forces_2)
#endif
                           (const gmx_nbl_ci_t *nbl_ci,
                            const gmx_nbl_sj_t *nbl_sj,
                            const gmx_nbl_si_t *nbl_si,
                            const int *atom_types,
                            int ntypes,
                            const float4 *xq,
                            const float *nbfp,
                            const float3 *shift_vec,
                            float two_k_rf,
                            float cutoff_sq,
                            float coulomb_tab_scale,
#ifdef CALC_ENERGIES
                            float beta,
                            float c_rf,
                            float *e_lj,
                            float *e_el,
#endif                          
                            float4 *f)
{
    unsigned int tidxi  = threadIdx.x;
    unsigned int tidxj  = threadIdx.y;
    unsigned int tidx   = threadIdx.y * blockDim.x + threadIdx.x;
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
#ifdef CALC_ENERGIES
          E_lj, E_el,
#endif             
          F_invr;
    float4 f4tmp;
    float3 xi, xj, rv;
    float3 shift;
    float3 f_ij, fsj_buf;
    gmx_nbl_ci_t nb_ci;
    unsigned int excl_bit;

    extern __shared__ float forcebuf[];  /* j force buffer */
    float3 fsi_buf[NSUBCELL];            /* i force buffer */

    nb_ci       = nbl_ci[bidx];         /* cell index */
    ci          = nb_ci.ci;             /* i cell index = current block index */
    cij_start   = nb_ci.sj_ind_start;   /* first ...*/
    cij_end     = nb_ci.sj_ind_end;     /* and last index of j cells */

    shift       = shift_vec[nb_ci.shift];

    for(si_offset = 0; si_offset < NSUBCELL; si_offset++)
    {
        fsi_buf[si_offset] = make_float3(0.0f);
    }

#ifdef CALC_ENERGIES
    E_lj = 0.0f;
    E_el = 0.0f;
#endif

    /* loop over the j sub-cells = seen by any of the atoms in the current cell */   
    for (j = cij_start ; j < cij_end; j++)
    {
        sj          = nbl_sj[j].sj; /* TODO int4? */
        si_start    = nbl_sj[j].si_ind;
        si_end      = nbl_sj[j + 1].si_ind;
        aj          = sj * CELL_SIZE + tidxj;

        /* load j atom data into registers */
        f4tmp   = xq[aj];
        xj      = make_float3(f4tmp.x, f4tmp.y, f4tmp.z);
        qj_f    = GPU_FACEL * f4tmp.w;
        typej   = atom_types[aj];
        xj      -= shift;

        fsj_buf = make_float3(0.0f);

        /* loop over i sub-cells in ci */
        for (i = si_start; i < si_end; i++)
        {
            si          = nbl_si[i].si;
            si_offset   = si - ci * NSUBCELL;       /* i force buffer offset */     
            ai          = si * CELL_SIZE + tidxi;  /* i atom index */

            excl_bit = (nbl_si[i].excl >> tidx) & 1;

            /* all threads load an atom from i cell si into shmem! */            
            f4tmp   = xq[ai];
            xi      = make_float3(f4tmp.x, f4tmp.y, f4tmp.z);
            qi      = f4tmp.w;
            typei   = atom_types[ai];

            /* LJ C6 and C12 */
            c6  = tex1Dfetch(tex_nbfp, 2 * (ntypes * typei + typej));
            c12 = tex1Dfetch(tex_nbfp, 2 * (ntypes * typei + typej) + 1);

            rv  = xi - xj;
            r2  = norm2(rv);

            /* cutoff & exclusion check */
            if (r2 < cutoff_sq * excl_bit)
            {
                inv_r       = 1.0f / sqrt(r2);
                inv_r2      = inv_r * inv_r;
                inv_r6      = inv_r2 * inv_r2 * inv_r2;

                F_invr      = inv_r6 * (12.0f * c12 * inv_r6 - 6.0f * c6) * inv_r2;

#ifdef CALC_ENERGIES
                E_lj        += inv_r6 * (c12 * inv_r6 - c6);
#endif

#ifdef EL_CUTOFF
                F_invr      += qi * qj_f * inv_r2 * inv_r;  
#endif
#ifdef EL_RF
                F_invr      += qi * qj_f * (inv_r2 * inv_r - two_k_rf); 
#endif
#ifdef EL_EWALD
                F_invr      += qi * qj_f * interpolate_coulomb_force_r(r2 * inv_r, coulomb_tab_scale);
#endif
                
#ifdef CALC_ENERGIES
#ifdef EL_CUTOFF
                E_el        += qi * qj_f * inv_r;
#endif
#ifdef EL_RF
                E_el        += qi * qj_f * (inv_r + 0.5f * two_k_rf * r2 - c_rf);
#endif
#ifdef EL_EWALD
                E_el        += qi * qj_f * inv_r * erfc(inv_r * beta);
#endif
#endif
                f_ij    = rv * F_invr;

                /* accumulate j forces in registers */
                fsj_buf -= f_ij;

                /* accumulate i forces in registers */
                fsi_buf[si_offset] += f_ij;
            }
        }

        /* store j forces in shmem */
        forcebuf[                 tidx] = fsj_buf.x;
        forcebuf[    STRIDE_DIM + tidx] = fsj_buf.y;
        forcebuf[2 * STRIDE_DIM + tidx] = fsj_buf.z;

        /* reduce j forces */
        reduce_force_j_generic_strided(forcebuf, f, tidxi, tidxj, aj);
    }

    /* reduce i forces */
    for (si_offset = 0; si_offset < NSUBCELL; si_offset++)
    {
        ai  = (ci * NSUBCELL + si_offset) * CELL_SIZE + tidxi;
        forcebuf[                 tidx] = fsi_buf[si_offset].x;
        forcebuf[    STRIDE_DIM + tidx] = fsi_buf[si_offset].y;
        forcebuf[2 * STRIDE_DIM + tidx] = fsi_buf[si_offset].z;
        __syncthreads();
        reduce_force_i_strided(forcebuf, f, tidxi, tidxj, ai);        
        __syncthreads();
    }

#ifdef CALC_ENERGIES
    /* add each thread's local energy to the global value */
    atomicAdd(e_lj, E_lj);
    atomicAdd(e_el, E_el);
#endif
}

#undef FUNCTION_NAME
#undef EL_CUTOFF
#undef EL_RF
#undef EL_EWALD
