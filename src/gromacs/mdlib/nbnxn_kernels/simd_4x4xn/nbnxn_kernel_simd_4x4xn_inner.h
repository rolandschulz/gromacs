/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2012,2013,2014, by the GROMACS development team, led by
 * Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
 * and including many others, as listed in the AUTHORS file in the
 * top-level source directory and at http://www.gromacs.org.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * http://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at http://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org.
 */

/* This is the innermost loop contents for the 4 x N atom SIMD kernel.
 * This flavor of the kernel duplicates the data for N j-particles in
 * 2xN wide SIMD registers to do operate on 2 i-particles at once.
 * This leads to 4/2=2 sets of most instructions. Therefore we call
 * this kernel 2x(N+N) = 2xnn
 *
 * This 2xnn kernel is basically the 4xn equivalent with half the registers
 * and instructions removed.
 *
 * An alternative would be to load to different cluster of N j-particles
 * into SIMD registers, giving a 4x(N+N) kernel. This doubles the amount
 * of instructions, which could lead to better scheduling. But we actually
 * observed worse scheduling for the AVX-256 4x8 normal analytical PME
 * kernel, which has a lower pair throughput than 2x(4+4) with gcc 4.7.
 * It could be worth trying this option, but it takes some more effort.
 * This 2xnn kernel is basically the 4xn equivalent with
 */


/* When calculating RF or Ewald interactions we calculate the electrostatic/LJ
 * forces on excluded atom pairs here in the non-bonded loops.
 * But when energies and/or virial is required we calculate them
 * separately to as then it is easier to separate the energy and virial
 * contributions.
 */
#if defined CHECK_EXCLS && (defined CALC_COULOMB || defined LJ_EWALD_GEOM)
#define EXCL_FORCES
#endif

/* Without exclusions and energies we only need to mask the cut-off,
 * this can be faster with blendv.
 */
#if !(defined CHECK_EXCLS || defined CALC_ENERGIES || defined LJ_EWALD_GEOM) && defined GMX_SIMD_HAVE_BLENDV
/* With RF and tabulated Coulomb we replace cmp+and with sub+blendv.
 * With gcc this is slower, except for RF on Sandy Bridge.
 * Tested with gcc 4.6.2, 4.6.3 and 4.7.1.
 */
#if (defined CALC_COUL_RF || defined CALC_COUL_TAB) && (!defined __GNUC__ || (defined CALC_COUL_RF && defined GMX_SIMD_X86_AVX_256_OR_HIGHER))
#define NBNXN_CUTOFF_USE_BLENDV
#endif
/* With analytical Ewald we replace cmp+and+and with sub+blendv+blendv.
 * This is only faster with icc on Sandy Bridge (PS kernel slower than gcc 4.7).
 * Tested with icc 13.
 */
#if defined CALC_COUL_EWALD && defined __INTEL_COMPILER && defined GMX_SIMD_X86_AVX_256_OR_HIGHER
#define NBNXN_CUTOFF_USE_BLENDV
#endif
#endif

{
    int        cj, aj, ajx, ajy, ajz;

#ifdef ENERGY_GROUPS
    /* Energy group indices for two atoms packed into one int */
    int        egp_jj[UNROLLJ/2];
#endif

#ifdef CHECK_EXCLS
    /* Interaction (non-exclusion) mask of all 1's or 0's */
    gmx_simd_bool_t  interact_S0;
#endif

    gmx_simd_real_t  jx_S, jy_S, jz_S;
    gmx_simd_real_t  dx_S0, dy_S0, dz_S0;
    gmx_simd_real_t  tx_S0, ty_S0, tz_S0;
    gmx_simd_real_t  rsq_S0, rinv_S0, rinvsq_S0;
#ifndef NBNXN_CUTOFF_USE_BLENDV
    /* wco: within cut-off, mask of all 1's or 0's */
    gmx_simd_bool_t  wco_S0;
#endif
#ifdef VDW_CUTOFF_CHECK
    gmx_simd_bool_t  wco_vdw_S0;
#endif

#if (defined CALC_COULOMB && defined CALC_COUL_TAB) || defined LJ_FORCE_SWITCH || defined LJ_POT_SWITCH
    gmx_simd_real_t r_S0;
#endif

#if defined LJ_FORCE_SWITCH || defined LJ_POT_SWITCH
    gmx_simd_real_t  rsw_S0, rsw2_S0, rsw2_r_S0;
#endif

#ifdef CALC_COULOMB
#ifdef CHECK_EXCLS
    /* 1/r masked with the interaction mask */
    gmx_simd_real_t  rinv_ex_S0;
#endif
    gmx_simd_real_t  jq_S;
    gmx_simd_real_t  qq_S0;
#ifdef CALC_COUL_TAB
    /* The force (PME mesh force) we need to subtract from 1/r^2 */
    gmx_simd_real_t  fsub_S0;
#endif
#ifdef CALC_COUL_EWALD
    gmx_simd_real_t  brsq_S0, ewcorr_S0;
#endif

    /* frcoul = (1/r - fsub)*r */
    gmx_simd_real_t  frcoul_S0;
#ifdef CALC_COUL_TAB
    /* For tables: r, rs=r/sp, rf=floor(rs), frac=rs-rf */
    gmx_simd_real_t         rs_S0, rf_S0, frac_S0;
    /* Table index: rs truncated to an int */
    gmx_simd_int32_t        ti_S0;
    /* Linear force table values */
    gmx_simd_real_t         ctab0_S0, ctab1_S0;
#ifdef CALC_ENERGIES
    /* Quadratic energy table value */
    gmx_simd_real_t  ctabv_S0;
#endif
#endif
#if defined CALC_ENERGIES && (defined CALC_COUL_EWALD || defined CALC_COUL_TAB)
    /* The potential (PME mesh) we need to subtract from 1/r */
    gmx_simd_real_t  vc_sub_S0;
#endif
#ifdef CALC_ENERGIES
    /* Electrostatic potential */
    gmx_simd_real_t  vcoul_S0;
#endif
#endif
    /* The force times 1/r */
    gmx_simd_real_t  fscal_S0;

#ifdef CALC_LJ
#ifdef LJ_COMB_LB
    /* LJ sigma_j/2 and sqrt(epsilon_j) */
    gmx_simd_real_t  hsig_j_S, seps_j_S;
    /* LJ sigma_ij and epsilon_ij */
    gmx_simd_real_t  sig_S0, eps_S0;
#ifdef CALC_ENERGIES
    gmx_simd_real_t  sig2_S0, sig6_S0;
#endif /* LJ_COMB_LB */
#endif /* CALC_LJ */

#ifdef LJ_COMB_GEOM
    gmx_simd_real_t  c6s_j_S, c12s_j_S;
#endif

#if defined LJ_COMB_GEOM || defined LJ_COMB_LB || defined LJ_EWALD_GEOM
    /* Index for loading LJ parameters, complicated when interleaving */
    int         aj2;
#endif

#ifndef FIX_LJ_C
    /* LJ C6 and C12 parameters, used with geometric comb. rule */
    gmx_simd_real_t  c6_S0, c12_S0;
#endif

    /* Intermediate variables for LJ calculation */
#ifndef LJ_COMB_LB
    gmx_simd_real_t  rinvsix_S0;
#endif
#ifdef LJ_COMB_LB
    gmx_simd_real_t  sir_S0, sir2_S0, sir6_S0;
#endif

    gmx_simd_real_t  FrLJ6_S0, FrLJ12_S0, frLJ_S0;
#if defined CALC_ENERGIES || defined LJ_POT_SWITCH
    gmx_simd_real_t  VLJ6_S0, VLJ12_S0, VLJ_S0;
#endif
#endif /* CALC_LJ */

    gmx_mm_qpr fjx_S, fjy_S, fjz_S;

    /* j-cluster index */
    cj            = l_cj[cjind].cj;

    /* Atom indices (of the first atom in the cluster) */
    aj            = cj*UNROLLJ;
#if defined CALC_LJ && (defined LJ_COMB_GEOM || defined LJ_COMB_LB || defined LJ_EWALD_GEOM)
    aj2           = aj*2;
#endif
    ajx           = aj*DIM;
    ajy           = ajx + STRIDE;
    ajz           = ajy + STRIDE;

#ifdef CHECK_EXCLS
    gmx_load_simd_4x4xn_interactions(l_cj[cjind].excl,
                                     filter_S0, &interact_S0);
#endif /* CHECK_EXCLS */

    /* load j atom coordinates */
    gmx_bcastq_pr(&jx_S, x+ajx);
    gmx_bcastq_pr(&jy_S, x+ajy);
    gmx_bcastq_pr(&jz_S, x+ajz);

    /* Calculate distance */
    dx_S0       = gmx_simd_sub_r(ix_S0, jx_S);
    dy_S0       = gmx_simd_sub_r(iy_S0, jy_S);
    dz_S0       = gmx_simd_sub_r(iz_S0, jz_S);

    /* rsq = dx*dx+dy*dy+dz*dz */
    rsq_S0      = gmx_simd_calc_rsq_r(dx_S0, dy_S0, dz_S0);

#ifndef NBNXN_CUTOFF_USE_BLENDV
    wco_S0      = gmx_simd_cmplt_r(rsq_S0, rc2_S);
#endif

#ifdef CHECK_EXCLS
#ifdef EXCL_FORCES
    /* Only remove the (sub-)diagonal to avoid double counting */
#if UNROLLJ == UNROLLI
    if (cj == ci_sh)
    {
        wco_S0  = gmx_simd_and_b(wco_S0, diagonal_mask_S0);
    }
#else
#error "only UNROLLJ == UNROLLI currently supported in 2xnn kernels"
#endif
#else /* EXCL_FORCES */
      /* No exclusion forces: remove all excluded atom pairs from the list */
    wco_S0      = gmx_simd_and_b(wco_S0, interact_S0);
#endif
#endif

#ifdef COUNT_PAIRS
    {
        int  i, j;
        real tmpa[2*GMX_SIMD_REAL_WIDTH], *tmp;
        tmp = gmx_simd_align_r(tmpa);
        gmx_simd_store_r(tmp, gmx_simd_sub_r(rc2_S, rsq_S0));
        for (j = 0; j < UNROLLI*UNROLLJ; j++)
        {
            if (tmp[j] >= 0)
            {
                npair++;
            }
        }
    }
#endif

#ifdef CHECK_EXCLS
    /* For excluded pairs add a small number to avoid r^-6 = NaN */
    rsq_S0      = gmx_simd_add_r(rsq_S0, gmx_simd_blendnotzero_r(avoid_sing_S, interact_S0));
#endif

    /* Calculate 1/r */
    rinv_S0     = gmx_simd_invsqrt_r(rsq_S0);

#ifdef CALC_COULOMB
    /* Load parameters for j atom */
    gmx_bcastq_pr(&jq_S, q+aj);
    qq_S0       = gmx_simd_mul_r(iq_S0, jq_S);
#endif

#ifdef CALC_LJ

#if !defined LJ_COMB_GEOM && !defined LJ_COMB_LB && !defined FIX_LJ_C
#ifndef LJ_TABLE_GATHER
    load_lj_pair_params4(nbfp0, nbfp1, nbfp2, nbfp3, type, aj, &c6_S0, &c12_S0);
#else
    gmx_simd_int32_t nbfp_j;
    gmx_bcastq_epi(&nbfp_j, type+aj);
    nbfp_j = gmx_simd_add_i(nbfp_j, nbfp);
    gmx_gather_pr(nbfp_ptr,   nbfp_j, &c6_S0, nbfp_stride);
    gmx_gather_pr(nbfp_ptr+1, nbfp_j, &c12_S0, nbfp_stride);
#endif
#endif /* not defined any LJ rule */

#ifdef LJ_COMB_GEOM
    gmx_bcastq_pr(&c6s_j_S,  ljc+aj2+0);
    gmx_bcastq_pr(&c12s_j_S, ljc+aj2+STRIDE);
    c6_S0       = gmx_simd_mul_r(c6s_S0, c6s_j_S );
    c12_S0      = gmx_simd_mul_r(c12s_S0, c12s_j_S);
#endif /* LJ_COMB_GEOM */

#ifdef LJ_COMB_LB
    gmx_bcastq_pr(&hsig_j_S, ljc+aj2+0);
    gmx_bcastq_pr(&seps_j_S, ljc+aj2+STRIDE);

    sig_S0      = gmx_simd_add_r(hsig_i_S0, hsig_j_S);
    eps_S0      = gmx_simd_mul_r(seps_i_S0, seps_j_S);
#endif /* LJ_COMB_LB */

#endif /* CALC_LJ */

#ifndef NBNXN_CUTOFF_USE_BLENDV
    rinv_S0     = gmx_simd_blendzero_r(rinv_S0, wco_S0);
#else
    /* This needs to be modified: It makes assumptions about the internal storage
     * of the SIMD representation, in particular that the blendv instruction always
     * selects based on the sign bit. If the performance is really critical, it
     * should be turned into a function that is platform-specific.
     */
    /* We only need to mask for the cut-off: blendv is faster */
    rinv_S0     = gmx_simd_blendv_r(rinv_S0, zero_S, gmx_simd_sub_r(rc2_S, rsq_S0));
#endif

    rinvsq_S0   = gmx_simd_mul_r(rinv_S0, rinv_S0);

#ifdef CALC_COULOMB
    /* Note that here we calculate force*r, not the usual force/r.
     * This allows avoiding masking the reaction-field contribution,
     * as frcoul is later multiplied by rinvsq which has been
     * masked with the cut-off check.
     */

#ifdef EXCL_FORCES
    /* Only add 1/r for non-excluded atom pairs */
    rinv_ex_S0  = gmx_simd_blendzero_r(rinv_S0, interact_S0);
#else
    /* No exclusion forces, we always need 1/r */
#define     rinv_ex_S0    rinv_S0
#endif

#ifdef CALC_COUL_RF
    /* Electrostatic interactions */
    frcoul_S0   = gmx_simd_mul_r(qq_S0, gmx_simd_fmadd_r(rsq_S0, mrc_3_S, rinv_ex_S0));

#ifdef CALC_ENERGIES
    vcoul_S0    = gmx_simd_mul_r(qq_S0, gmx_simd_add_r(rinv_ex_S0, gmx_simd_add_r(gmx_simd_mul_r(rsq_S0, hrc_3_S), moh_rc_S)));
#endif
#endif

#ifdef CALC_COUL_EWALD
    /* We need to mask (or limit) rsq for the cut-off,
     * as large distances can cause an overflow in gmx_pmecorrF/V.
     */
#ifndef NBNXN_CUTOFF_USE_BLENDV
    brsq_S0     = gmx_simd_mul_r(beta2_S, gmx_simd_blendzero_r(rsq_S0, wco_S0));
#else
    /* Strangely, putting mul on a separate line is slower (icc 13) */
    brsq_S0     = gmx_simd_mul_r(beta2_S, gmx_simd_blendv_r(rsq_S0, zero_S, gmx_simd_sub_r(rc2_S, rsq_S0)));
#endif
    ewcorr_S0   = gmx_simd_mul_r(gmx_simd_pmecorrF_r(brsq_S0), beta_S);
    frcoul_S0   = gmx_simd_mul_r(qq_S0, gmx_simd_fmadd_r(ewcorr_S0, brsq_S0, rinv_ex_S0));

#ifdef CALC_ENERGIES
    vc_sub_S0   = gmx_simd_mul_r(gmx_simd_pmecorrV_r(brsq_S0), beta_S);
#endif

#endif /* CALC_COUL_EWALD */

#ifdef CALC_COUL_TAB
    /* Electrostatic interactions */
    r_S0        = gmx_simd_mul_r(rsq_S0, rinv_S0);
    /* Convert r to scaled table units */
    rs_S0       = gmx_simd_mul_r(r_S0, invtsp_S);
    /* Truncate scaled r to an int */
    ti_S0       = gmx_simd_cvtt_r2i(rs_S0);
#ifdef GMX_SIMD_HAVE_TRUNC
    rf_S0       = gmx_simd_trunc_r(rs_S0);
#else
    rf_S0       = gmx_simd_cvt_i2r(ti_S0);
#endif
    frac_S0     = gmx_simd_sub_r(rs_S0, rf_S0);

    /* Load and interpolate table forces and possibly energies.
     * Force and energy can be combined in one table, stride 4: FDV0
     * or in two separate tables with stride 1: F and V
     * Currently single precision uses FDV0, double F and V.
     */
#ifndef CALC_ENERGIES
    load_table_f(tab_coul_F, ti_S0, ti0, &ctab0_S0, &ctab1_S0);
#else
#ifdef TAB_FDV0
    load_table_f_v(tab_coul_F, ti_S0, ti0, &ctab0_S0, &ctab1_S0, &ctabv_S0);
#else
    load_table_f_v(tab_coul_F, tab_coul_V, ti_S0, ti0, &ctab0_S0, &ctab1_S0, &ctabv_S0);
#endif
#endif
    fsub_S0     = gmx_simd_add_r(ctab0_S0, gmx_simd_mul_r(frac_S0, ctab1_S0));
    frcoul_S0   = gmx_simd_mul_r(qq_S0, gmx_simd_sub_r(rinv_ex_S0, gmx_simd_mul_r(fsub_S0, r_S0)));

#ifdef CALC_ENERGIES
    vc_sub_S0   = gmx_simd_add_r(ctabv_S0, gmx_simd_mul_r(gmx_simd_mul_r(mhalfsp_S, frac_S0), gmx_simd_add_r(ctab0_S0, fsub_S0)));
#endif
#endif /* CALC_COUL_TAB */

#if defined CALC_ENERGIES && (defined CALC_COUL_EWALD || defined CALC_COUL_TAB)
#ifndef NO_SHIFT_EWALD
    /* Add Ewald potential shift to vc_sub for convenience */
#ifdef CHECK_EXCLS
    vc_sub_S0   = gmx_simd_add_r(vc_sub_S0, gmx_simd_blendzero_r(sh_ewald_S, interact_S0));
#else
    vc_sub_S0   = gmx_simd_add_r(vc_sub_S0, sh_ewald_S);
#endif
#endif

    vcoul_S0    = gmx_simd_mul_r(qq_S0, gmx_simd_sub_r(rinv_ex_S0, vc_sub_S0));
#endif

#ifdef CALC_ENERGIES
    /* Mask energy for cut-off and diagonal */
    vcoul_S0    = gmx_simd_blendzero_r(vcoul_S0, wco_S0);
#endif

#endif /* CALC_COULOMB */

#ifdef CALC_LJ
    /* Lennard-Jones interaction */

#ifdef VDW_CUTOFF_CHECK
    wco_vdw_S0  = gmx_simd_cmplt_r(rsq_S0, rcvdw2_S);
#else
    /* Same cut-off for Coulomb and VdW, reuse the registers */
#define     wco_vdw_S0    wco_S0
#endif

#ifndef LJ_COMB_LB
    rinvsix_S0  = gmx_simd_mul_r(rinvsq_S0, gmx_simd_mul_r(rinvsq_S0, rinvsq_S0));
#ifdef EXCL_FORCES
    rinvsix_S0  = gmx_simd_blendzero_r(rinvsix_S0, interact_S0);
#endif

#if defined LJ_CUT || defined LJ_POT_SWITCH
    /* We have plain LJ or LJ-PME with simple C6/6 C12/12 coefficients */
    FrLJ6_S0    = gmx_simd_mul_r(c6_S0, rinvsix_S0);
    FrLJ12_S0   = gmx_simd_mul_r(c12_S0, gmx_simd_mul_r(rinvsix_S0, rinvsix_S0));
#endif

#if defined LJ_FORCE_SWITCH || defined LJ_POT_SWITCH
    /* We switch the LJ force */
    r_S0        = gmx_simd_mul_r(rsq_S0, rinv_S0);
    rsw_S0      = gmx_simd_max_r(gmx_simd_sub_r(r_S0, rswitch_S), zero_S);
    rsw2_S0     = gmx_simd_mul_r(rsw_S0, rsw_S0);
    rsw2_r_S0   = gmx_simd_mul_r(rsw2_S0, r_S0);
#endif

#ifdef LJ_FORCE_SWITCH

#define add_fr_switch(fr, rsw, rsw2_r, c2, c3) gmx_simd_fmadd_r(gmx_simd_fmadd_r(c3, rsw, c2), rsw2_r, fr)

    FrLJ6_S0    = gmx_simd_mul_r(c6_S0, add_fr_switch(rinvsix_S0, rsw_S0, rsw2_r_S0, p6_fc2_S, p6_fc3_S));
    FrLJ12_S0   = gmx_simd_mul_r(c12_S0, add_fr_switch(gmx_simd_mul_r(rinvsix_S0, rinvsix_S0), rsw_S0, rsw2_r_S0, p12_fc2_S, p12_fc3_S));
#undef add_fr_switch
#endif /* LJ_FORCE_SWITCH */

#endif /* not LJ_COMB_LB */

#ifdef LJ_COMB_LB
    sir_S0      = gmx_simd_mul_r(sig_S0, rinv_S0);
    sir2_S0     = gmx_simd_mul_r(sir_S0, sir_S0);
    sir6_S0     = gmx_simd_mul_r(sir2_S0, gmx_simd_mul_r(sir2_S0, sir2_S0));
#ifdef EXCL_FORCES
    sir6_S0     = gmx_simd_blendzero_r(sir6_S0, interact_S0);
#endif
#ifdef VDW_CUTOFF_CHECK
    sir6_S0     = gmx_simd_blendzero_r(sir6_S0, wco_vdw_S0);
#endif
    FrLJ6_S0    = gmx_simd_mul_r(eps_S0, sir6_S0);
    FrLJ12_S0   = gmx_simd_mul_r(FrLJ6_S0, sir6_S0);
#if defined CALC_ENERGIES
    /* We need C6 and C12 to calculate the LJ potential shift */
    sig2_S0     = gmx_simd_mul_r(sig_S0, sig_S0);
    sig6_S0     = gmx_simd_mul_r(sig2_S0, gmx_simd_mul_r(sig2_S0, sig2_S0));
    c6_S0       = gmx_simd_mul_r(eps_S0, sig6_S0);
    c12_S0      = gmx_simd_mul_r(c6_S0, sig6_S0);
#endif
#endif /* LJ_COMB_LB */

    /* Determine the total scalar LJ force*r */
    frLJ_S0     = gmx_simd_sub_r(FrLJ12_S0, FrLJ6_S0);

#if (defined LJ_CUT || defined LJ_FORCE_SWITCH) && defined CALC_ENERGIES

#ifdef LJ_CUT
    /* Calculate the LJ energies, with constant potential shift */
    VLJ6_S0     = gmx_simd_mul_r(sixth_S, gmx_simd_fmadd_r(c6_S0, p6_cpot_S, FrLJ6_S0));
    VLJ12_S0    = gmx_simd_mul_r(twelveth_S, gmx_simd_fmadd_r(c12_S0, p12_cpot_S, FrLJ12_S0));
#endif /* LJ_CUT */

#ifdef LJ_FORCE_SWITCH
#define v_fswitch_pr(rsw, rsw2, c0, c3, c4) gmx_simd_fmadd_r(gmx_simd_fmadd_r(c4, rsw, c3), gmx_simd_mul_r(rsw2, rsw), c0)

    VLJ6_S0     = gmx_simd_mul_r(c6_S0, gmx_simd_fmadd_r(sixth_S, rinvsix_S0, v_fswitch_pr(rsw_S0, rsw2_S0, p6_6cpot_S, p6_vc3_S, p6_vc4_S)));
    VLJ12_S0    = gmx_simd_mul_r(c12_S0, gmx_simd_fmadd_r(twelveth_S, gmx_simd_mul_r(rinvsix_S0, rinvsix_S0), v_fswitch_pr(rsw_S0, rsw2_S0, p12_12cpot_S, p12_vc3_S, p12_vc4_S)));
#undef v_fswitch_pr
#endif /* LJ_FORCE_SWITCH */

    /* Add up the repulsion and dispersion */
    VLJ_S0      = gmx_simd_sub_r(VLJ12_S0, VLJ6_S0);

#endif /* (LJ_CUT || LJ_FORCE_SWITCH) && CALC_ENERGIES */

#ifdef LJ_POT_SWITCH
    /* We always need the potential, since it is needed for the force */
    VLJ_S0 = gmx_simd_fnmadd_r(sixth_S, FrLJ6_S0, gmx_simd_mul_r(twelveth_S, FrLJ12_S0));

    {
        gmx_simd_real_t sw_S0, dsw_S0;

#define switch_pr(rsw, rsw2, c3, c4, c5) gmx_simd_fmadd_r(gmx_simd_fmadd_r(gmx_simd_fmadd_r(c5, rsw, c4), rsw, c3), gmx_simd_mul_r(rsw2, rsw), one_S)
#define dswitch_pr(rsw, rsw2, c2, c3, c4) gmx_simd_mul_r(gmx_simd_fmadd_r(gmx_simd_fmadd_r(c4, rsw, c3), rsw, c2), rsw2)

        sw_S0  = switch_pr(rsw_S0, rsw2_S0, swV3_S, swV4_S, swV5_S);
        dsw_S0 = dswitch_pr(rsw_S0, rsw2_S0, swF2_S, swF3_S, swF4_S);
        frLJ_S0 = gmx_simd_fnmadd_r(gmx_simd_mul_r(dsw_S0, VLJ_S0), r_S0, gmx_simd_mul_r(sw_S0, frLJ_S0));
#ifdef CALC_ENERGIES
        VLJ_S0  = gmx_simd_mul_r(sw_S0, VLJ_S0);
#endif

#undef switch_pr
#undef dswitch_pr
    }
#endif /* LJ_POT_SWITCH */

#if defined CALC_ENERGIES && defined CHECK_EXCLS
    /* The potential shift should be removed for excluded pairs */
    VLJ_S0      = gmx_simd_blendzero_r(VLJ_S0, interact_S0);
#endif

#ifdef LJ_EWALD_GEOM
    {
        gmx_simd_real_t c6s_j_S;
        gmx_simd_real_t c6grid_S0, rinvsix_nm_S0, cr2_S0, expmcr2_S0, poly_S0;
#ifdef CALC_ENERGIES
        gmx_simd_real_t sh_mask_S0;
#endif

        /* Determine C6 for the grid using the geometric combination rule */
        gmx_bcastq_pr(&c6s_j_S,  ljc+aj2+0);
        c6grid_S0       = gmx_simd_mul_r(c6s_S0, c6s_j_S);

#ifdef CHECK_EXCLS
        /* Recalculate rinvsix without exclusion mask (compiler might optimize) */
        rinvsix_nm_S0 = gmx_simd_mul_r(rinvsq_S0, gmx_simd_mul_r(rinvsq_S0, rinvsq_S0));
#else
        /* We didn't use a mask, so we can copy */
        rinvsix_nm_S0 = rinvsix_S0;
#endif

        /* Mask for the cut-off to avoid overflow of cr2^2 */
        cr2_S0        = gmx_simd_mul_r(lje_c2_S, gmx_simd_blendzero_r(rsq_S0, wco_vdw_S0));
        expmcr2_S0    = gmx_simd_exp_r(gmx_simd_mul_r(mone_S, cr2_S0));

        /* 1 + cr2 + 1/2*cr2^2 */
        poly_S0       = gmx_simd_fmadd_r(gmx_simd_fmadd_r(half_S, cr2_S0, one_S), cr2_S0, one_S);

        /* We calculate LJ F*r = (6*C6)*(r^-6 - F_mesh/6), we use:
         * r^-6*cexp*(1 + cr2 + cr2^2/2 + cr2^3/6) = cexp*(r^-6*poly + c^6/6)
         */
        frLJ_S0       = gmx_simd_fmadd_r(c6grid_S0, gmx_simd_fnmadd_r(expmcr2_S0, gmx_simd_fmadd_r(rinvsix_nm_S0, poly_S0, lje_c6_6_S), rinvsix_nm_S0), frLJ_S0);

#ifdef CALC_ENERGIES
#ifdef CHECK_EXCLS
        sh_mask_S0    = gmx_simd_blendzero_r(lje_vc_S, interact_S0);
#else
        sh_mask_S0    = lje_vc_S;
#endif

        VLJ_S0        = gmx_simd_fmadd_r(gmx_simd_mul_r(sixth_S, c6grid_S0), gmx_simd_fmadd_r(rinvsix_nm_S0, gmx_simd_fnmadd_r(expmcr2_S0, poly_S0, one_S), sh_mask_S0), VLJ_S0);
#endif /* CALC_ENERGIES */
    }
#endif /* LJ_EWALD_GEOM */

#if defined VDW_CUTOFF_CHECK
    /* frLJ is multiplied later by rinvsq, which is masked for the Coulomb
     * cut-off, but if the VdW cut-off is shorter, we need to mask with that.
     */
    frLJ_S0     = gmx_simd_blendzero_r(frLJ_S0, wco_vdw_S0);
#endif

#ifdef CALC_ENERGIES
    /* The potential shift should be removed for pairs beyond cut-off */
    VLJ_S0      = gmx_simd_blendzero_r(VLJ_S0, wco_vdw_S0);
#endif

#endif /* CALC_LJ */

#ifdef CALC_ENERGIES
#ifdef ENERGY_GROUPS
    /* Extract the group pair index per j pair.
     * Energy groups are stored per i-cluster, so things get
     * complicated when the i- and j-cluster size don't match.
     */
    {
        int egps_j;
#if UNROLLJ == 2
        egps_j    = nbat->energrp[cj>>1];
        egp_jj[0] = ((egps_j >> ((cj & 1)*egps_jshift)) & egps_jmask)*egps_jstride;
#else
        /* We assume UNROLLI <= UNROLLJ */
        int jdi;
        for (jdi = 0; jdi < UNROLLJ/UNROLLI; jdi++)
        {
            int jj;
            egps_j = nbat->energrp[cj*(UNROLLJ/UNROLLI)+jdi];
            for (jj = 0; jj < (UNROLLI/2); jj++)
            {
                egp_jj[jdi*(UNROLLI/2)+jj] = ((egps_j >> (jj*egps_jshift)) & egps_jmask)*egps_jstride;
            }
        }
#endif
    }
#endif

#ifdef CALC_COULOMB
#ifndef ENERGY_GROUPS
    vctot_S      = gmx_simd_add_r(vctot_S, vcoul_S0);
#else
    add_ener_grp_quarters(vcoul_S0, vctp, egp_jj, FALSE);
#endif
#endif

#ifdef CALC_LJ
#ifndef ENERGY_GROUPS
    Vvdwtot_S    = gmx_simd_add_r(Vvdwtot_S, VLJ_S0);
#else
    add_ener_grp_quarters(VLJ_S0, vvdwtp, egp_jj, half_LJ); //this can be done nicely with hardware gather/scatter (becuase often they are the same grp and complicated tempory 1-2 buffer is not needed then). Then no reduce_group_energies is needed. Masking those not needed (e.g. half_lj/beyond cut-off) can reduce load/store further.
#endif
#endif /* CALC_LJ */
#endif /* CALC_ENERGIES */

#ifdef CALC_LJ
#ifdef CALC_COULOMB
    fscal_S0    = gmx_simd_mul_r(rinvsq_S0, gmx_simd_add_r(frcoul_S0, frLJ_S0));
#else
    fscal_S0    = gmx_simd_mul_r(rinvsq_S0, frLJ_S0);
#endif
#else
    fscal_S0    = gmx_simd_mul_r(rinvsq_S0, frcoul_S0);
#endif /* CALC_LJ */

    /* Calculate temporary vectorial force */
    tx_S0       = gmx_simd_mul_r(fscal_S0, dx_S0);
    ty_S0       = gmx_simd_mul_r(fscal_S0, dy_S0);
    tz_S0       = gmx_simd_mul_r(fscal_S0, dz_S0);

    /* Increment i atom force */
    fix_S0      = gmx_simd_add_r(fix_S0, tx_S0);
    fiy_S0      = gmx_simd_add_r(fiy_S0, ty_S0);
    fiz_S0      = gmx_simd_add_r(fiz_S0, tz_S0);

    /* Decrement j atom force */
    gmx_load_qpr(&fjx_S, f+ajx);
    gmx_load_qpr(&fjy_S, f+ajy);
    gmx_load_qpr(&fjz_S, f+ajz);
    gmx_store_qpr(f+ajx, gmx_sub_qpr(fjx_S, gmx_sum4_qpr(tx_S0)));
    gmx_store_qpr(f+ajy, gmx_sub_qpr(fjy_S, gmx_sum4_qpr(ty_S0)));
    gmx_store_qpr(f+ajz, gmx_sub_qpr(fjz_S, gmx_sum4_qpr(tz_S0)));
}

#undef  rinv_ex_S0

#undef  wco_vdw_S0

#undef  NBNXN_CUTOFF_USE_BLENDV

#undef  EXCL_FORCES
