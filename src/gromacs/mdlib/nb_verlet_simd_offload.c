/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team.
 * Copyright (c) 2013,2014, by the GROMACS development team, led by
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
#include <stdlib.h>
#include <immintrin.h>
#include "nbnxn_internal.h"
#include "nbnxn_atomdata.h"
#include "nb_verlet.h"
#include "nbnxn_kernels/simd_2xnn/nbnxn_kernel_simd_2xnn.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/utility/smalloc.h"
#include "nb_verlet_simd_offload.h"
#include "timing.h"
#include "packdata.h"

gmx_bool bUseOffloadedKernel = FALSE;
gmx_offload gmx_bool bRefreshNbl = TRUE;
rvec *force_buffer = NULL;
size_t fb_size = 0;
gmx_bool bOffloadDummyPtrInit = FALSE;
void *offload_dummy_ptr = NULL;

// Reuse pointer (neither allocate nor free)
#define OFFREUSEPTR(PTR) PTR:length(0) alloc_if(0) free_if(0)

#define OFFGETPTR(TYPE, PTRVAR, PTRVAL, LENVAR, LENVAL) \
  TYPE *PTRVAR = PTRVAL; size_t LENVAR = LENVAL * sizeof(TYPE); \
  if (!bOffloadDummyPtrInit) \
  { \
    smalloc(offload_dummy_ptr, 1); \
    bOffloadDummyPtrInit = TRUE; \
  } \
  if (LENVAL < 1) {PTRVAR = (TYPE *)offload_dummy_ptr; LENVAR = 0;}

#define OFFEXTPTR(PTR, LEN) PTR:length(LEN) alloc_if(0) free_if(0)

// "Mirror" malloc with corresponding renew and free. Memory is allocated on both
// host and coprocessor, and the two are linked to support offloading operations.
void *mmalloc(size_t s)
{
	void *p;
	smalloc(p,s);
	char *c = (char *)p;
#pragma offload_transfer target(mic:0) in(c:length(s) alloc_if(1) free_if(0))
	return p;
}

void *mrenew(void *p, size_t s)
{
	char *oldc = (char *)p;
	srenew(p,s);
	char *newc = (char *)p;
#pragma offload_transfer target(mic:0) in(oldc:length(0) alloc_if(0) free_if(1))
#pragma offload_transfer target(mic:0) in(newc:length(s) alloc_if(1) free_if(0))
	return p;
}

void mfree(void *p)
{
	sfree(p);
	char *c = (char *)p;
#pragma offload_transfer target(mic:0) in(c:length(0) alloc_if(0) free_if(1))
}

// TODO: move so that forward declaration isn't needed
gmx_offload void nbnxn_atomdata_init_simple_exclusion_masks(nbnxn_atomdata_t *nbat);

void nbnxn_kernel_simd_2xnn_offload(t_forcerec *fr,
                                    interaction_const_t *ic,
                                    gmx_enerdata_t *enerd,
                                    int flags, int ilocality,
                                    int clearF,
                                    t_nrnb *nrnb,
                                    gmx_wallcycle_t wcycle)
{
	code_timer *ct = create_code_timer();

	int i;
	nonbonded_verlet_group_t  *nbvg = &fr->nbv->grp[ilocality];
	gmx_offload static nbnxn_pairlist_set_t *nbl_lists;
	nbl_lists = &nbvg->nbl_lists;

	static int nbl_buffer_size = 0;
	static int ci_buffer_size  = 0;
	static int sci_buffer_size = 0;
	static int cj_buffer_size  = 0;
	static int cj4_buffer_size = 0;
	gmx_offload static nbnxn_pairlist_t *nbl_buffer;
	gmx_offload static nbnxn_ci_t  *ci_buffer;
	gmx_offload static nbnxn_sci_t *sci_buffer;
	gmx_offload static nbnxn_cj_t  *cj_buffer;
	gmx_offload static nbnxn_cj4_t *cj4_buffer;

	gmx_offload static gmx_bool firstRefresh = TRUE;
	reset_timer(ct);
	if (bRefreshNbl)
	{
		dprintf(2, "Refresh nbl\n");
		if (nbl_buffer_size > 0) mfree(nbl_buffer);
		if (ci_buffer_size > 0)  mfree(ci_buffer);
		if (sci_buffer_size > 0) mfree(sci_buffer);
		if (cj_buffer_size > 0)  mfree(cj_buffer);
		if (cj4_buffer_size > 0) mfree(cj4_buffer);

		nbl_buffer_size = nbvg->nbl_lists.nnbl;
		ci_buffer_size = 0;
		sci_buffer_size = 0;
		cj_buffer_size = 0;
		cj4_buffer_size = 0;
		nbnxn_pairlist_t **nbl = nbl_lists->nbl;
		for (i=0; i<nbl_buffer_size; i++)
		{
			ci_buffer_size   += nbl[i]->nci;
			sci_buffer_size  += nbl[i]->nsci;
			cj_buffer_size   += nbl[i]->ncj;
			cj4_buffer_size  += nbl[i]->ncj4;
		}
		// dprintf(2, "Buffer sizes %d %d %d %d\n", ci_buffer_size, sci_buffer_size, cj_buffer_size, cj4_buffer_size);
		nbl_buffer = mmalloc(sizeof(nbnxn_pairlist_t)*nbl_buffer_size);
		if (ci_buffer_size > 0)  ci_buffer = mmalloc(sizeof(nbnxn_ci_t)*ci_buffer_size);
		if (sci_buffer_size > 0) sci_buffer = mmalloc(sizeof(nbnxn_sci_t)*sci_buffer_size);
		if (cj_buffer_size > 0)  cj_buffer = mmalloc(sizeof(nbnxn_cj_t)*cj_buffer_size);
		if (cj4_buffer_size > 0) cj4_buffer = mmalloc(sizeof(nbnxn_cj4_t)*cj4_buffer_size);

		// dprintf(2, "Malloced buffers are %p %p %p %p\n", ci_buffer, sci_buffer, cj_buffer, cj4_buffer);
		int ci_offset  = 0;
		int sci_offset = 0;
		int cj_offset  = 0;
		int cj4_offset = 0;
		for (i=0; i<nbvg->nbl_lists.nnbl; i++)
		{
			memcpy(nbl_buffer + i, nbl[i], sizeof(nbnxn_pairlist_t));
			// TODO: memcpy does nothing if size is 0, right?
			memcpy(ci_buffer + ci_offset, nbl[i]->ci, nbl[i]->nci * sizeof(nbnxn_ci_t));
			ci_offset += nbl[i]->nci;
			memcpy(sci_buffer + sci_offset, nbl[i]->sci, nbl[i]->nsci * sizeof(nbnxn_sci_t));
			sci_offset += nbl[i]->nsci;
			memcpy(cj_buffer + cj_offset, nbl[i]->cj, nbl[i]->ncj * sizeof(nbnxn_cj_t));
			cj_offset += nbl[i]->ncj;
			memcpy(cj4_buffer + cj4_offset, nbl[i]->cj4, nbl[i]->ncj4 * sizeof(nbnxn_cj4_t));
			cj4_offset += nbl[i]->ncj4;
		}
	}
	OFFGETPTR(nbnxn_pairlist_set_t, nbl_lists_p, nbl_lists, nbl_lists_p_len, 1);
	OFFGETPTR(nbnxn_pairlist_t, nblb, nbl_buffer, nblb_len, nbl_buffer_size);
	OFFGETPTR(nbnxn_ci_t, cib, ci_buffer, cib_len, ci_buffer_size);
	OFFGETPTR(nbnxn_sci_t, scib, sci_buffer, scib_len, sci_buffer_size);
	OFFGETPTR(nbnxn_cj_t, cjb, cj_buffer, cjb_len, cj_buffer_size);
	OFFGETPTR(nbnxn_cj4_t, cj4b, cj4_buffer, cj4b_len, cj4_buffer_size);
	// Avoid transferring unnecessarily.
	if (!bRefreshNbl)
	{
		nbl_lists_p_len = 0;
		nblb_len = 0;
		cib_len = 0;
		scib_len = 0;
		cjb_len = 0;
		cj4b_len = 0;
	}

	OFFGETPTR(nbnxn_atomdata_t, nbat, nbvg->nbat, nbat_len, 1)
	OFFGETPTR(real, nbfp_p, nbat->nbfp, nbfp_p_len, nbat->ntype*nbat->ntype*2)
	OFFGETPTR(real, nbfp_comb_p, nbat->nbfp_comb, nbfp_comb_p_len, nbat->ntype*2)
	OFFGETPTR(real, nbfp_s4_p, nbat->nbfp_s4, nbfp_s4_p_len, nbat->ntype*nbat->ntype*4)
	OFFGETPTR(int, type_p, nbat->type, type_p_len, nbat->natoms)
	OFFGETPTR(real, lj_comb_p, nbat->lj_comb, lj_comb_p_len, nbat->natoms*2)
	OFFGETPTR(real, q_p, nbat->q, q_p_len, nbat->natoms)
	OFFGETPTR(int, energrp_p, nbat->energrp, energrp_p_len, ((nbat->nenergrp>1) ? (nbat->natoms/nbat->na_c):0))
	OFFGETPTR(rvec, shift_vec_p, nbat->shift_vec, shift_vec_p_len, SHIFTS)
	OFFGETPTR(real, x_p, nbat->x, x_p_len, nbat->natoms * nbat->xstride)
	OFFGETPTR(gmx_bitmask_t, flag_p, nbat->buffer_flags.flag, flag_p_len, nbat->buffer_flags.flag_nalloc)

	reset_timer(ct);
	if (nbat->comb_rule == ljcrNONE)
	{
		nbfp_comb_p = offload_dummy_ptr;
		nbfp_comb_p_len = 0;
	}

	OFFGETPTR(interaction_const_t, ic_buffer, ic, ic_buffer_len, 1);
	OFFGETPTR(rvec, shift_vec, (fr->shift_vec), shift_vec_len, SHIFTS);
	OFFGETPTR(real, fshift, (fr->fshift[0]), fshift_len, DIM*SHIFTS);
	OFFGETPTR(real, Vc, (enerd->grpp.ener[egCOULSR]), Vc_len, enerd->grpp.nener);
	OFFGETPTR(real, Vvdw, (fr->bBHAM ? enerd->grpp.ener[egBHAMSR] : enerd->grpp.ener[egLJSR]), Vvdw_len, enerd->grpp.nener);
	int                   ewald_excl = nbvg->ewald_excl;

	// Data needed for force and shift reductions
	OFFGETPTR(struct nbnxn_search, nbs, fr->nbv->nbs, nbs_len, 1);
	OFFGETPTR(int, cell_p, fr->nbv->nbs->cell, cell_p_len, fr->nbv->nbs->cell_nalloc);
	OFFGETPTR(rvec, f_p, force_buffer, f_p_len, fb_size);
	// dprintf(2, "Are great pointers %p %p %p %p %p %p %p %p %p %p %p %p %p %p %p %p %p %p %p %p %p %p %p\n", cib, scib, cjb, cj4b, nbat, nbfp_p, nbfp_comb_p, nbfp_s4_p, type_p, lj_comb_p, q_p, energrp_p,
			                  // shift_vec_p, x_p, flag_p, ic_buffer, shift_vec, fshift, Vc, Vvdw, nbs, cell_p, f_p);
	 // dprintf(2, "Are great lengths %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n", cib_len, scib_len, cjb_len, cj4b_len, nbat_len, nbfp_p_len, nbfp_comb_p_len, nbfp_s4_p_len, type_p_len,
        //     lj_comb_p_len, q_p_len, energrp_p_len, shift_vec_p_len, x_p_len, flag_p_len, ic_buffer_len, shift_vec_len, fshift_len, Vc_len, Vvdw_len,
			//    nbs_len, cell_p_len, f_p_len);
	// dprintf(2, "Fshifty things %p %d %d\n", fshift, fshift_len, SHIFTS);
	void *transfer_in_buffers[25] = {nbl_lists_p, nblb, cib, scib, cjb, cj4b, nbat, nbfp_p, nbfp_comb_p, nbfp_s4_p, type_p, lj_comb_p, q_p, energrp_p,
			                  shift_vec_p, x_p, flag_p, ic_buffer, shift_vec, fshift, Vc, Vvdw, nbs, cell_p, f_p};
	size_t transfer_in_sizes[25] = {nbl_lists_p_len, nblb_len, cib_len, scib_len, cjb_len, cj4b_len, nbat_len, nbfp_p_len, nbfp_comb_p_len, nbfp_s4_p_len, type_p_len,
			               lj_comb_p_len, q_p_len, energrp_p_len, shift_vec_p_len, x_p_len, flag_p_len, ic_buffer_len, shift_vec_len, fshift_len, Vc_len, Vvdw_len,
						   nbs_len, cell_p_len, f_p_len};
	static char *transfer_in_packet = NULL;
	size_t packet_in_size = compute_required_size(transfer_in_sizes, 25);
	if (transfer_in_packet == NULL)
	{
		transfer_in_packet = mmalloc(packet_in_size);
	}
	else
	{
		transfer_in_packet = mrenew(transfer_in_packet, packet_in_size);
	}
	packdata(transfer_in_packet, transfer_in_buffers, transfer_in_sizes, 25);

	size_t transfer_out_sizes[4] = {fshift_len, Vc_len, Vvdw_len, f_p_len};
	static char *transfer_out_packet = NULL;
	size_t packet_out_size = compute_required_size(transfer_out_sizes, 4);
	if (transfer_out_packet == NULL)
	{
		transfer_out_packet = mmalloc(packet_out_size);
	}
	else
	{
		transfer_out_packet = mrenew(transfer_out_packet, packet_out_size);
	}

	// TODO: Figure out why we need this kludge for static variables and how to handle it best.
	char *tip = transfer_in_packet;
	char *top = transfer_out_packet;
	dprintf(2, "Packet sizes in out %d %d\n", packet_in_size, packet_out_size);
	// dprintf(2, "Transfer packet information: %p %d %d\n", transfer_out_packet, packet_out_size, current_packet_out_size);
	dprintf(2, "Sizes of stuff %d %d %d %d\n", sizeof(nbnxn_pairlist_set_t), sizeof(nbnxn_pairlist_t), sizeof(nbnxn_ci_t), sizeof(nbnxn_sci_t));
	int off_signal = 0;
#define NUM_TIMES 10
	double *phi_times;
	smalloc(phi_times, NUM_TIMES * sizeof(double));
	int j;
	for (j=0; j<NUM_TIMES; j++) phi_times[j] = 0;
	reset_timer(ct);
#pragma offload target(mic:0) nocopy(out_for_phi) \
	                          nocopy(nbl_lists) \
	                          nocopy(nbl_buffer) \
	                          nocopy(ci_buffer) \
	                          nocopy(sci_buffer) \
	                          nocopy(cj_buffer) \
	                          nocopy(cj4_buffer) \
                              in (OFFEXTPTR(tip, packet_in_size)) \
	                          out(OFFEXTPTR(top, packet_out_size)) \
							  inout(phi_times:length(NUM_TIMES) alloc_if(1) free_if(1))
							  // signal(&off_signal) // nocopy(excl:length(nbl[i]->nexcl) ALLOC)
	{
		// Unpack data
		packet_iter *it;
		smalloc(it, sizeof(packet_iter));

		create_packet_iter(tip, it);

		if (bRefreshNbl)
		{
			if (!firstRefresh)
			{
				free(nbl_lists);
				free(nbl_buffer);
				free(ci_buffer);
				free(sci_buffer);
				free(cj_buffer);
				free(cj4_buffer);
			}
			nbl_lists = anext(it);
			nbl_buffer = anext(it);
			ci_buffer = anext(it);
			sci_buffer = anext(it);
			cj_buffer = anext(it);
			cj4_buffer = anext(it);
			firstRefresh = FALSE;
		}
		else
		{
			int i;
			for (i=0; i<6; i++) next(it);
		}
		nbnxn_atomdata_t *nbat = next(it);
		nbat->nbfp = next(it);
		nbat->nbfp_comb = next(it);
		nbat->nbfp_s4 = next(it);
		nbat->type = next(it);
		nbat->lj_comb = next(it);
		nbat->q = next(it);
		nbat->energrp = next(it);
		nbat->shift_vec = next(it);
		nbat->x = next(it);
		nbat->out = out_for_phi;
		nbat->buffer_flags.flag = next(it);
		interaction_const_t *ic_buffer = next(it);
		rvec *shift_vec = next(it);
		real *fshift = next(it);
		real *Vc = next(it);
		real *Vvdw = next(it);
		struct nbnxn_search *nbs = next(it);
		nbs->cell = next(it);
		rvec *f_p = next(it);
        real *flat_f = (real *)f_p;
		sfree(it);

		code_timer *ct_phi = create_code_timer();
        reset_timer(ct_phi);

        // Neighbor list pointer assignments
        int ci_offset = 0;
        int sci_offset = 0;
        int cj_offset = 0;
        int cj4_offset = 0;

        // TODO: Should only need to malloc once, not every refresh. Transferring nbl_lists, though, wipes out the old nbl_lists->nbl pointer.
        if (bRefreshNbl) nbl_lists->nbl = malloc(sizeof(nbnxn_pairlist_t *)*nbl_lists->nnbl);

        int i;
        for (i=0; i<nbl_lists->nnbl; i++)
        {
        	nbl_lists->nbl[i] = nbl_buffer + i;
		    nbnxn_pairlist_t *nbl = nbl_lists->nbl[i];
		    nbl->ci   = ci_buffer  + ci_offset;
		    nbl->sci  = sci_buffer + sci_offset;
		    nbl->cj   = cj_buffer  + cj_offset;
		    nbl->cj4  = cj4_buffer + cj4_offset;
		    // nbl->excl = excl;
		    ci_offset  += nbl->nci;
		    sci_offset += nbl->nsci;
		    cj_offset  += nbl->ncj;
		    cj4_offset += nbl->ncj4;
        }

        // End unpacking of data and start actual computing
		//TODO: if tables are used, the coul_F and coul_V need to be copied
		//following not needed after. Instead we should call init_simple_exclusion_masks
		//nbat->simd_4xn_diagonal_j_minus_i  = simd_4xn_diagonal_j_minus_i_p;
		//                nbat->simd_2xnn_diagonal_j_minus_i = simd_2xnn_diagonal_j_minus_i_p;
		//                nbat->simd_exclusion_filter1       = simd_exclusion_filter1_p;
		//                nbat->simd_exclusion_filter2       = simd_exclusion_filter2_p;
        // Unpacking time
        phi_times[0] = get_elapsed_time(ct_phi);
        reset_timer(ct_phi);
		nbnxn_atomdata_init_simple_exclusion_masks(nbat); //TODO: much better to just init that on the MIC - but the function is static there. Probably the whole copy code should be moved there anyhow and then we call this functions
		/*TODO: ic: table (only if tables are used)
        		        nbl_lists->nbl. It needs to be done as the intel example "Transferring Arrays of Pointers". Because their are several lists and each contain several of the clusters
        		        nbat: quite a few (maybe not all are needed)

        		        verify that those marked as in/out are really only input/output
        		        do outputs need to be zeroed?
        		        if we test with reduction on CPU side the number of threads has to match

        		        the numa issue for nbl_lists might also be important for MIC so we might want to do a manual allocation

        		        it isn't OK to reuse the data and not free it. This is currently the case for the elements within nbl_lists and nbat
		 */
		// Mask time
		phi_times[1] = get_elapsed_time(ct_phi);
		reset_timer(ct_phi);
		nbnxn_kernel_simd_2xnn(nbl_lists,   //Neighbor list: needs to be send after each update
				//nbat contains cordinates (need to be send each step), forces (have to be send back), static information (e.g. charges)
				//ic seems to be all static. is fr->ic
				nbat, ic_buffer,
				ewald_excl, //might depend on Neighbor list or is static
				shift_vec,   //depends on box size (changes usually with neighbor list)
				flags,
				clearF,
				fshift,   //output
				Vc, //output
				Vvdw); //output
		// TODO: Put back in when we figure out how to know when to free.
		// free (nbl_lists->nbl);
		// Kernel only time
		phi_times[2] = get_elapsed_time(ct_phi);
		reset_timer(ct_phi);
		bRefreshNbl = FALSE;
		// Force and shift reductions

        nbnxn_atomdata_add_nbat_f_to_f(nbs, eatAll, nbat, f_p);
        // Force reduction time
		phi_times[3] = get_elapsed_time(ct_phi);
		reset_timer(ct_phi);
        // TODO: Figure out if we can always assume that this is done.
        nbnxn_atomdata_add_nbat_fshift_to_fshift(nbat, (rvec *)fshift);
		// Fshift reduction time
		phi_times[4] = get_elapsed_time(ct_phi);
		reset_timer(ct_phi);
        void *transfer_out_buffers[4] = {fshift, Vc, Vvdw, f_p};
        size_t transfer_out_sizes[4] = {get_buffer_size(tip, 19),  // fshift
        								get_buffer_size(tip, 20),  // Vc
										get_buffer_size(tip, 21),  // Vvdw
										get_buffer_size(tip, 24)}; // f_p
		packdata(top, transfer_out_buffers, transfer_out_sizes, 4);
		// Pack output buffer time
		phi_times[5] = get_elapsed_time(ct_phi);
		free_code_timer(ct_phi);
	}
	// while(!_Offload_signaled(0, &off_signal)) {}
	static int counter = -1;
	counter++;
	if (counter > 5)
	{
		dprintf(2, "Total offload time %f\n", get_elapsed_time(ct));
	}
	reset_timer(ct);
	void *phi_out_buffers[4] = {fshift, Vc, Vvdw, f_p};
	unpackdata(transfer_out_packet, phi_out_buffers, 4);
    dprintf(2, "Vc is %f and Vvdw is %f\n", Vc[0], Vvdw[0]);
    if (counter > 5)
    {
    	dprintf(2, "Unpack output buffer time %f\n", get_elapsed_time(ct));
    	dprintf(2, "Phi times:");
    	for (int i=0; i<NUM_TIMES; i++) dprintf(2, " %f", phi_times[i]);
    	dprintf(2, "\n");
	}
	sfree(phi_times);
	static int first_kernel_run = 0;
	if (first_kernel_run) first_kernel_run = 0;
	free_code_timer(ct);
}
