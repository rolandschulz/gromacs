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

#define REUSE alloc_if(0) free_if(0)

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

	packet_buffer ibuffers[25];
	ibuffers[0] =  (packet_buffer){nbl_lists, sizeof(nbnxn_pairlist_set_t) * (bRefreshNbl ? 1:0)};
	ibuffers[1] =  (packet_buffer){nbl_buffer, sizeof(nbnxn_pairlist_t) * (bRefreshNbl ? nbl_buffer_size : 0)};
	ibuffers[2] =  (packet_buffer){ci_buffer, sizeof(nbnxn_ci_t) * (bRefreshNbl ? ci_buffer_size : 0)};
	ibuffers[3] =  (packet_buffer){sci_buffer, sizeof(nbnxn_sci_t) * (bRefreshNbl ? sci_buffer_size : 0)};
	ibuffers[4] =  (packet_buffer){cj_buffer, sizeof(nbnxn_cj_t) * (bRefreshNbl ? cj_buffer_size : 0)};
	ibuffers[5] =  (packet_buffer){cj4_buffer, sizeof(nbnxn_cj4_t) * (bRefreshNbl ? cj4_buffer_size : 0)};
	nbnxn_atomdata_t *nbat = nbvg->nbat;
	ibuffers[6] =  (packet_buffer){nbat, sizeof(nbnxn_atomdata_t)};
	ibuffers[7] =  (packet_buffer){nbat->nbfp, sizeof(real) * (nbat->ntype*nbat->ntype*2)};
	ibuffers[8] =  (packet_buffer){nbat->nbfp_comb, sizeof(real) * (nbat->comb_rule != ljcrNONE ? nbat->ntype*2 : 0)};
	ibuffers[9] =  (packet_buffer){nbat->nbfp_s4, sizeof(real) * (nbat->ntype*nbat->ntype*4)};
	ibuffers[10] = (packet_buffer){nbat->type, sizeof(int) * (nbat->natoms)};
	ibuffers[11] = (packet_buffer){nbat->lj_comb, sizeof(real) * (nbat->natoms*2)};
	ibuffers[12] = (packet_buffer){nbat->q, sizeof(real) * (nbat->natoms)};
	ibuffers[13] = (packet_buffer){nbat->energrp, sizeof(int) * ((nbat->nenergrp>1) ? (nbat->natoms/nbat->na_c):0)};
	ibuffers[14] = (packet_buffer){nbat->shift_vec, sizeof(rvec) * SHIFTS};
	ibuffers[15] = (packet_buffer){nbat->x, sizeof(real) * (nbat->natoms * nbat->xstride)};
	ibuffers[16] = (packet_buffer){nbat->buffer_flags.flag, sizeof(gmx_bitmask_t) * (nbat->buffer_flags.flag_nalloc)};
	ibuffers[17] = (packet_buffer){ic, sizeof(interaction_const_t)};
	ibuffers[18] = (packet_buffer){fr->shift_vec, sizeof(rvec) * SHIFTS};
	void *fshift = fr->fshift[0];
	void *Vc = enerd->grpp.ener[egCOULSR];
	void *Vvdw = fr->bBHAM ? enerd->grpp.ener[egBHAMSR] : enerd->grpp.ener[egLJSR];
	ibuffers[19] = (packet_buffer){fshift, sizeof(real) * (DIM*SHIFTS)};
	ibuffers[20] = (packet_buffer){Vc, sizeof(real) * (enerd->grpp.nener)};
	ibuffers[21] = (packet_buffer){Vvdw, sizeof(real) * (enerd->grpp.nener)};
	ibuffers[22] = (packet_buffer){fr->nbv->nbs, sizeof(struct nbnxn_search)};
	ibuffers[23] = (packet_buffer){fr->nbv->nbs->cell, sizeof(int) * (fr->nbv->nbs->cell_nalloc)};
	ibuffers[24] = (packet_buffer){force_buffer, sizeof(rvec) * fb_size};

	// TODO: Figure out if we ever need this
	int                   ewald_excl = nbvg->ewald_excl;

	// Data needed for force and shift reductions
	static char *transfer_in_packet = NULL;
	size_t packet_in_size = compute_required_size(ibuffers, 25);
	if (transfer_in_packet == NULL)
	{
		transfer_in_packet = mmalloc(packet_in_size);
	}
	else
	{
		transfer_in_packet = mrenew(transfer_in_packet, packet_in_size);
	}
	packdata(transfer_in_packet, ibuffers, 25);

	packet_buffer obuffers[4];
	obuffers[0] = ibuffers[19]; // FShift
	obuffers[1] = ibuffers[20]; // Vc
	obuffers[2] = ibuffers[21]; // Vvdw
	obuffers[3] = ibuffers[24]; // Force
	static char *transfer_out_packet = NULL;
	size_t packet_out_size = compute_required_size(obuffers, 4);
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
	// dprintf(2, "Packet sizes in out %d %d\n", packet_in_size, packet_out_size);
	// dprintf(2, "Transfer packet information: %p %d %d\n", transfer_out_packet, packet_out_size, current_packet_out_size);
	// dprintf(2, "Sizes of stuff %d %d %d %d\n", sizeof(nbnxn_pairlist_set_t), sizeof(nbnxn_pairlist_t), sizeof(nbnxn_ci_t), sizeof(nbnxn_sci_t));
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
                              in (tip:length(packet_in_size)  REUSE) \
	                          out(top:length(packet_out_size) REUSE) \
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
		packet_buffer phi_buffers[4];
		phi_buffers[0] = get_buffer(tip, 19);
		phi_buffers[1] = get_buffer(tip, 20);
		phi_buffers[2] = get_buffer(tip, 21);
		phi_buffers[3] = get_buffer(tip, 24);
		packdata(top, phi_buffers, 4);
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
	void *phi_out_buffers[4] = {fshift, Vc, Vvdw, force_buffer};
	unpackdata(transfer_out_packet, phi_out_buffers, 4);
    // dprintf(2, "Vc is %f and Vvdw is %f\n", Vc[0], Vvdw[0]);
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
