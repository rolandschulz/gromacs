/* -*- mode: c; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4; c-file-style: "stroustrup"; -*-
 *
 *
 *                This source code is part of
 * 
 *                 G   R   O   M   A   C   S
 * 
 *          GROningen MAchine for Chemical Simulations
 * 
 *                        VERSION 3.2.0
 * Written by David van der Spoel, Erik Lindahl, Berk Hess, and others.
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team,
 * check out http://www.gromacs.org for more information.

 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 * 
 * If you want to redistribute modifications, please consider that
 * scientific software is very special. Version control is crucial -
 * bugs must be traceable. We will be happy to consider code for
 * inclusion in the official distribution, but derived work must not
 * be called official GROMACS. Details are found in the README & COPYING
 * files - if they are missing, get the official version at www.gromacs.org.
 * 
 * To help us fund GROMACS development, we humbly ask that you cite
 * the papers on the package - you can find them in the top README file.
 * 
 * For more info, check our website at http://www.gromacs.org
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifdef GMX_THREAD_SHM_FDECOMP
#include <pthread.h> 
#endif

#include <math.h>
#include <string.h>
#include "sysstuff.h"
#include "smalloc.h"
#include "macros.h"
#include "maths.h"
#include "vec.h"
#include "pbc.h"
#include "nsbox.h"

#ifdef GMX_OPENMP
#include <omp.h>
#endif

#if ( !defined(GMX_DOUBLE) && ( defined(GMX_IA32_SSE) || defined(GMX_X86_64_SSE) || defined(GMX_X86_64_SSE2) ) )
#include "gmx_sse2_single.h"

#include <xmmintrin.h>
#include <emmintrin.h>
#endif


/* Neighbor search box upper and lower bound in x,y,z: total 6 reals */
#define NNBSBB_D 2
#define NNBSBB   (3*NNBSBB_D)

#define NSBOX_SHIFT_BACKWARD


/* This define is a lazy way to avoid interdependence of the grid
 * and searching data structures.
 */
#define NBL_NAPC_MAX (NSUBCELL*16)


typedef gmx_bool
gmx_subcell_in_range_t(int naps,
                       int si,const real *x_or_bb_i,
                       int csj,int stride,const real *x_or_bb_j,
                       real rl2);

static gmx_subcell_in_range_t subc_in_range_x;
static gmx_subcell_in_range_t subc_in_range_sse8;

typedef struct {
    int *cxy_na;

    int  *sort_work;
    int  sort_work_nalloc;
} gmx_nbs_work_t;

typedef struct gmx_nbsearch {
    int  ePBC;
    matrix box;
    real atom_density;

    int  naps;  /* Number of atoms in the inner loop / sub cell */
    int  napc;  /* Number of atoms in the super cell            */

    int  ncx;
    int  ncy;
    int  nc;

    real sx;
    real sy;
    real inv_sx;
    real inv_sy;

    int  *cxy_na;
    int  *cxy_ind;
    int  cxy_nalloc;

    int  *cell;
    int  cell_nalloc;

    int  *a;
    int  *nsubc; /* The number of sub cells for each super cell */
    real *bbcz;  /* Bounding boxes in z for the super cells     */
    real *bb;    /* 3D bounding boxes for the sub cells         */
    int  nc_nalloc;

    int  nsubc_tot;

    gmx_subcell_in_range_t *subc_dc;

    int  nthread_max;
    gmx_nbs_work_t *work;
} gmx_nbsearch_t_t;

typedef struct gmx_nbl_work {
    real *bb_si;
    real *x_si;
} gmx_nbl_work_t;

void gmx_nbsearch_init(gmx_nbsearch_t * nbs_ptr,int natoms_subcell)
{
    gmx_nbsearch_t nbs;
    int t;

    snew(nbs,1);
    *nbs_ptr = nbs;
    
    nbs->naps = natoms_subcell;
    nbs->napc = natoms_subcell*NSUBCELL;

    if (nbs->napc > NBL_NAPC_MAX)
    {
        gmx_fatal(FARGS,
                  "napc (%d) is larger than the maximum allowed value (%d)",
                  nbs->napc,NBL_NAPC_MAX);
    }

    nbs->cxy_na      = NULL;
    nbs->cxy_ind     = NULL;
    nbs->cxy_nalloc  = 0;
    nbs->cell        = NULL;
    nbs->cell_nalloc = 0;
    nbs->a           = NULL;
    nbs->bb          = NULL;
    nbs->nc_nalloc   = 0;

    if (getenv("GMX_NSBOX_BB") != NULL)
    {
        /* Use only bounding box sub cell pair distances,
         * fast, but produces slighlty more sub cell pairs.
         */
        nbs->subc_dc = NULL;
    }
    else
    {
#if ( !defined(GMX_DOUBLE) && ( defined(GMX_IA32_SSE) || defined(GMX_X86_64_SSE) || defined(GMX_X86_64_SSE2) ) )
        if (natoms_subcell == 8 && getenv("GMX_NSBOX_NOSSE") == NULL)
        {
            nbs->subc_dc = subc_in_range_sse8;
        }
        else
#endif
        { 
            nbs->subc_dc = subc_in_range_x;
        }
    }

    nbs->nthread_max = 1;
#ifdef GMX_OPENMP
    {
        nbs->nthread_max = omp_get_max_threads();
    }
#endif

    snew(nbs->work,nbs->nthread_max);
    for(t=0; t<nbs->nthread_max; t++)
    {
        nbs->work[t].sort_work   = NULL;
        nbs->work[t].sort_work_nalloc = 0;
    }
}

static int set_grid_size_xy(gmx_nbsearch_t nbs,int n,matrix box)
{
    real adens,tlen,tlen_x,tlen_y,nc_max;
    int  t;

    nbs->atom_density = n/(box[XX][XX]*box[YY][YY]*box[ZZ][ZZ]);

    if (n > nbs->napc)
    {
        /* target cell length */
#if 0
        /* Approximately cubic super cells */
        tlen   = pow(nbs->napc/nbs->atom_density,1.0/3.0);
        tlen_x = tlen;
        tlen_y = tlen;
#else
        /* Approximately cubic sub cells */
        tlen   = pow(nbs->naps/nbs->atom_density,1.0/3.0);
        tlen_x = tlen*NSUBCELL_X;
        tlen_y = tlen*NSUBCELL_Y;
#endif
        /* We round ncx and ncy down, because we get less cell pairs
         * in the nbsist when the fixed cell dimensions (x,y) are
         * larger than the variable one (z) than the other way around.
         */
        nbs->ncx = max(1,(int)(box[XX][XX]/tlen_x));
        nbs->ncy = max(1,(int)(box[YY][YY]/tlen_y));
    }
    else
    {
        nbs->ncx = 1;
        nbs->ncy = 1;
    }

    if (nbs->ncx*nbs->ncy+1 > nbs->cxy_nalloc)
    {
        nbs->cxy_nalloc = over_alloc_large(nbs->ncx*nbs->ncy);
        srenew(nbs->cxy_na,nbs->cxy_nalloc);
        srenew(nbs->cxy_ind,nbs->cxy_nalloc+1);

        for(t=0; t<nbs->nthread_max; t++)
        {
            srenew(nbs->work[t].cxy_na,nbs->cxy_nalloc);
        }
    }

    /* Worst case scenario of 1 atom in east last cell */
    nc_max = n/nbs->napc + nbs->ncx*nbs->ncy;
    if (nc_max > nbs->nc_nalloc)
    {
        nbs->nc_nalloc = over_alloc_large(nc_max);
        srenew(nbs->a,nbs->nc_nalloc*nbs->napc);
        srenew(nbs->nsubc,nbs->nc_nalloc);
        srenew(nbs->bbcz,nbs->nc_nalloc*NNBSBB_D);
        srenew(nbs->bb,nbs->nc_nalloc*NSUBCELL*NNBSBB);
    }

    nbs->sx = box[XX][XX]/nbs->ncx;
    nbs->sy = box[YY][YY]/nbs->ncy;
    nbs->inv_sx = 1/nbs->sx;
    nbs->inv_sy = 1/nbs->sy;

    return nc_max;
}

#define SORT_GRID_OVERSIZE 2
#define SGSF (SORT_GRID_OVERSIZE + 1)

static void sort_atoms(int dim,gmx_bool Backwards,
                       int *a,int n,rvec *x,
                       real h0,real invh,int nsort,int *sort)
{
    int i,c;
    int zi,zim;
    int cp,tmp;

    if (n <= 1)
    {
        /* Nothing to do */
        return;
    }

    /* For small oversize factors clearing the whole area is fastest.
     * For large oversize we should clear the used elements after use.
     */
    for(i=0; i<nsort; i++)
    {
        sort[i] = -1;
    }
    /* Sort the particles using a simple index sort */
    for(i=0; i<n; i++)
    {
        /* The cast takes care of float-point rounding effects below zero.
         * This code assumes particles are less than 1/SORT_GRID_OVERSIZE
         * times the box height out of the box.
         */
        zi = (int)((x[a[i]][dim] - h0)*invh);

        /* Ideally this particle should go in sort cell zi,
         * but that might already be in use,
         * in that case find the first empty cell higher up
         */
        if (sort[zi] < 0)
        {
            sort[zi] = a[i];
        }
        else
        {
            /* We have multiple atoms in the same sorting slot.
             * Sort on real z for minimal bounding box size.
             * There is an extra check for identical z to ensure
             * well-defined output order, independent of input order
             * to ensure binary reproducibility after restarts.
             */
            while(sort[zi] >= 0 && ( x[a[i]][dim] >  x[sort[zi]][dim] ||
                                    (x[a[i]][dim] == x[sort[zi]][dim] &&
                                     a[i] > sort[zi])))
            {
                zi++;
            }

            if (sort[zi] >= 0)
            {
                /* Shift all elements by one slot until we find an empty slot */
                cp = sort[zi];
                zim = zi + 1;
                while (sort[zim] >= 0)
                {
                    tmp = sort[zim];
                    sort[zim] = cp;
                    cp  = tmp;
                    zim++;
                }
                sort[zim] = cp;
            }
            sort[zi] = a[i];
        }
    }

    c = 0;
    if (!Backwards)
    {
        for(zi=0; zi<nsort; zi++)
        {
            if (sort[zi] >= 0)
            {
                a[c++] = sort[zi];
            }
        }
    }
    else
    {
        for(zi=nsort-1; zi>=0; zi--)
        {
            if (sort[zi] >= 0)
            {
                a[c++] = sort[zi];
            }
        }
    }
    if (c < n)
    {
        gmx_incons("Lost particles while sorting");
    }
}

static void calc_bounding_box(int na,int stride,const real *x,real *bb)
{
    int  i,j;
    real xl,xh,yl,yh,zl,zh;

    i = 0;
    xl = x[i+XX];
    xh = x[i+XX];
    yl = x[i+YY];
    yh = x[i+YY];
    zl = x[i+ZZ];
    zh = x[i+ZZ];
    i += stride;
    for(j=1; j<na; j++)
    {
        xl = min(xl,x[i+XX]);
        xh = max(xh,x[i+XX]);
        yl = min(yl,x[i+YY]);
        yh = max(yh,x[i+YY]);
        zl = min(zl,x[i+ZZ]);
        zh = max(zh,x[i+ZZ]);
        i += stride;
    }
    bb[0] = xl;
    bb[1] = xh;
    bb[2] = yl;
    bb[3] = yh;
    bb[4] = zl;
    bb[5] = zh;
}

static void print_bbsizes(FILE *fp,const gmx_nbsearch_t nbs)
{
    int  ns,c,s,cs,d;
    dvec ba;

    clear_dvec(ba);
    ns = 0;
    for(c=0; c<nbs->nc; c++)
    {
        for(s=0; s<nbs->nsubc[c]; s++)
        {
            cs = c*NSUBCELL + s;
            for(d=0; d<DIM; d++)
            {
                ba[d] += nbs->bb[cs*NNBSBB+d*2+1] - nbs->bb[cs*NNBSBB+d*2];
            }
            ns++;
        }
    }
    dsvmul(1.0/ns,ba,ba);

    fprintf(fp,"ns bb: %4.2f %4.2f %4.2f  %4.2f %4.2f %4.2f rel %4.2f %4.2f %4.2f\n",
            nbs->box[XX][XX]/(nbs->ncx*NSUBCELL_X),
            nbs->box[YY][YY]/(nbs->ncy*NSUBCELL_Y),
            nbs->box[ZZ][ZZ]*nbs->ncx*nbs->ncy/(nbs->nc*NSUBCELL_Z),
            ba[XX],ba[YY],ba[ZZ],
            ba[XX]*nbs->ncx*NSUBCELL_X/nbs->box[XX][XX],
            ba[YY]*nbs->ncy*NSUBCELL_Y/nbs->box[YY][YY],
            ba[ZZ]*nbs->nc*NSUBCELL_Z/(nbs->ncx*nbs->ncy*nbs->box[ZZ][ZZ]));
}

static void copy_int_to_nbat_int(const int *a,int na,int na_round,
                                 const int *in,int fill,int *innb)
{
    int i,j;

    j = 0;
    for(i=0; i<na; i++)
    {
        innb[j++] = in[a[i]];
    }
    /* Complete the partially filled last cell with fill */
    for(; i<na_round; i++)
    {
        innb[j++] = fill;
    }
}

static void clear_nbat_real(int na,int stride,real *xnb)
{
    int i;

    for(i=0; i<na*stride; i++)
    {
        xnb[i] = 0;
    }
}

static void copy_rvec_to_nbat_real(const int *a,int na,int na_round,
                                   rvec *x,int nbatXFormat,real *xnb,
                                   int cx,int cy,int cz)
{
    int i,j;

/* We might need to place filler particles to fill ub the cell to na_round.
 * The coefficients (LJ and q) for such particles are zero.
 * But we might still get NaN as 0*NaN when distances are too small.
 * We hope that -107 nm is far away enough from to zero
 * to avoid accidental short distances to particles shifted down for pbc.
 */
#define NBAT_FAR_AWAY 107

    switch (nbatXFormat)
    {
    case nbatXYZ:
        j = 0;
        for(i=0; i<na; i++)
        {
            xnb[j++] = x[a[i]][XX];
            xnb[j++] = x[a[i]][YY];
            xnb[j++] = x[a[i]][ZZ];
        }
        /* Complete the partially filled last cell with copies of the last element.
         * This simplifies the bounding box calculation and avoid
         * numerical issues with atoms that are coincidentally close.
         */
        for(; i<na_round; i++)
        {
            xnb[j++] = -NBAT_FAR_AWAY*(1 + cx);
            xnb[j++] = -NBAT_FAR_AWAY*(1 + cy);
            xnb[j++] = -NBAT_FAR_AWAY*(1 + cz + i);
        }
        break;
    case nbatXYZQ:
        j = 0;
        for(i=0; i<na; i++)
        {
            xnb[j++] = x[a[i]][XX];
            xnb[j++] = x[a[i]][YY];
            xnb[j++] = x[a[i]][ZZ];
            j++;
        }
        /* Complete the partially filled last cell with copies of the last element.
         * This simplifies the bounding box calculation and avoid
         * numerical issues with atoms that are coincidentally close.
         */
        for(; i<na_round; i++)
        {
            xnb[j++] = -NBAT_FAR_AWAY*(1 + cx);
            xnb[j++] = -NBAT_FAR_AWAY*(1 + cy);
            xnb[j++] = -NBAT_FAR_AWAY*(1 + cz + i);
            j++;
        }
        break;
    default:
        gmx_incons("Unsupported stride");
    }
}
static void sort_columns(gmx_nbsearch_t nbs,
                         int n,rvec *x,
                         gmx_nb_atomdata_t *nbat,
                         int cxy_start,int cxy_end,
                         int *sort_work)
{
    int  i;
    int  cx,cy,cz,c=-1,ncz;
    int  na,ash,na_c;
    int  subdiv_z,sub_z,na_z,ash_z;
    int  subdiv_y,sub_y,na_y,ash_y;
    int  subdiv_x,sub_x,na_x,ash_x;
    real *xnb;

    subdiv_x = nbs->naps;
    subdiv_y = NSUBCELL_X*subdiv_x;
    subdiv_z = NSUBCELL_Y*subdiv_y;

    /* Sort the atoms within each x,y column in 3 dimensions */
    for(i=cxy_start; i<cxy_end; i++)
    {
        cx = i/nbs->ncy;
        cy = i - cx*nbs->ncy;

        na  = nbs->cxy_na[i];
        ncz = nbs->cxy_ind[i+1] - nbs->cxy_ind[i];
        ash = nbs->cxy_ind[i]*nbs->napc;

        /* Sort the atoms within each x,y column on z coordinate */
        sort_atoms(ZZ,FALSE,
                   nbs->a+ash,na,x,
                   0,
                   ncz*nbs->napc*SORT_GRID_OVERSIZE/nbs->box[ZZ][ZZ],
                   ncz*nbs->napc*SGSF,sort_work);

        /* This loop goes over the supercells and subcells along z at once */
        for(sub_z=0; sub_z<ncz*NSUBCELL_Z; sub_z++)
        {
            ash_z = ash + sub_z*subdiv_z;
            na_z  = min(subdiv_z,na-(ash_z-ash));

            /* We have already sorted on z */

            if (sub_z % NSUBCELL_Z == 0)
            {
                cz = sub_z/NSUBCELL_Z;
                c  = nbs->cxy_ind[i] + cz ;

                /* The number of atoms in this supercell */
                na_c = min(nbs->napc,na-(ash_z-ash));

                nbs->nsubc[c] = min(NSUBCELL,(na_c+nbs->naps-1)/nbs->naps);

                /* Store the z-boundaries of the super cell */
                nbs->bbcz[c*NNBSBB_D  ] = x[nbs->a[ash_z]][ZZ];
                nbs->bbcz[c*NNBSBB_D+1] = x[nbs->a[ash_z+na_c-1]][ZZ];
            }

#if NSUBCELL_Y > 1
            sort_atoms(YY,(sub_z & 1),
                       nbs->a+ash_z,na_z,x,
                       cy*nbs->sy,nbs->inv_sy,subdiv_y*SGSF,sort_work);
#endif

            for(sub_y=0; sub_y<NSUBCELL_Y; sub_y++)
            {
                ash_y = ash_z + sub_y*subdiv_y;
                na_y  = min(subdiv_y,na-(ash_y-ash));

#if NSUBCELL_X > 1
                sort_atoms(XX,((cz*NSUBCELL_Y + sub_y) & 1),
                           nbs->a+ash_y,na_y,x,
                           cx*nbs->sx,nbs->inv_sx,subdiv_x*SGSF,sort_work);
#endif

                for(sub_x=0; sub_x<NSUBCELL_X; sub_x++)
                {
                    ash_x = ash_y + sub_x*subdiv_x;
                    na_x  = min(subdiv_x,na-(ash_x-ash));

                    xnb   = nbat->x + ash_x*nbat->xstride;

                    if (na_x > 0)
                    {
                        copy_rvec_to_nbat_real(nbs->a+ash_x,na_x,nbs->naps,x,
                                               nbat->XFormat,xnb,
                                               nbs->naps*(cx*NSUBCELL_X+sub_x),
                                               nbs->naps*(cy*NSUBCELL_Y+sub_y),
                                               nbs->naps*sub_z);

                        calc_bounding_box(na_x,nbat->xstride,xnb,
                                          nbs->bb+(ash_x/nbs->naps)*NNBSBB);

                        if (gmx_debug_at)
                        {
                            fprintf(debug,"%2d %2d %2d %d %d %d bb %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f\n",
                                    cx,cy,cz,sub_x,sub_y,sub_z,
                                    (nbs->bb+(ash_x/nbs->naps)*NNBSBB)[0],
                                    (nbs->bb+(ash_x/nbs->naps)*NNBSBB)[1],
                                    (nbs->bb+(ash_x/nbs->naps)*NNBSBB)[2],
                                    (nbs->bb+(ash_x/nbs->naps)*NNBSBB)[3],
                                    (nbs->bb+(ash_x/nbs->naps)*NNBSBB)[4],
                                    (nbs->bb+(ash_x/nbs->naps)*NNBSBB)[5]);
                        }
                    }
                    else
                    {
                        clear_nbat_real(nbs->naps,nbat->xstride,xnb);
                    }
                }
            }
        }

        /*
        copy_rvec_to_nbat_real(axy,na,ncz*nbs->napc,x,nbat->xstride,xnb);

        calc_bounding_box(ncz,nbs->napc,nbat->xstride,xnb,
                          nbs->bb+nbs->cxy_ind[i]*NNBSBB);
        */
    }
}

static void calc_cell_indices(gmx_nbsearch_t nbs,
                              int n,rvec *x,
                              gmx_nb_atomdata_t *nbat)
{
    int  n0,n1,i;
    int  cx,cy,ncz_max,ncz;
    int  nthread,t;
    int  *cxy_na,cxy_na_i;

    nthread = omp_get_max_threads();

#pragma omp parallel  for private(cxy_na,n0,n1,i,cx,cy)
    for(t=0; t<nthread; t++)
    {
        cxy_na = nbs->work[t].cxy_na;

        for(i=0; i<nbs->ncx*nbs->ncy; i++)
        {
            cxy_na[i] = 0;
        }

        n0 = (int)((t+0)*n)/nthread;
        n1 = (int)((t+1)*n)/nthread;
        for(i=n0; i<n1; i++)
        {
            /* We need to be careful with rounding,
             * particles might be a few bits outside the local box.
             * The int cast takes care of the lower bound,
             * we need to explicitly take care of the upper bound.
             */
            cx = (int)(x[i][XX]*nbs->inv_sx);
            if (cx == nbs->ncx)
            {
                cx = nbs->ncx - 1;
            }
            cy = (int)(x[i][YY]*nbs->inv_sy);
            if (cy == nbs->ncy)
            {
                cy = nbs->ncy - 1;
            }
            /* For the moment cell contains only the x and y indices, not z */
            nbs->cell[i] = cx*nbs->ncy + cy;
            cxy_na[nbs->cell[i]]++;
        }
    }

    /* Make the cell index as a function of x and y */
    ncz_max = 0;
    nbs->cxy_ind[0] = 0;
    for(i=0; i<nbs->ncx*nbs->ncy; i++)
    {
        cxy_na_i = nbs->work[0].cxy_na[i];
        for(t=1; t<nthread; t++)
        {
            cxy_na_i += nbs->work[t].cxy_na[i];
        }
        ncz = (cxy_na_i + nbs->napc - 1)/nbs->napc;
        nbs->cxy_ind[i+1] = nbs->cxy_ind[i] + ncz;
        if (ncz > ncz_max)
        {
            ncz_max = ncz;
        }
        /* Clear cxy_na, so we can reuse the array below */
        nbs->cxy_na[i] = 0;
    }
    nbs->nc = nbs->cxy_ind[nbs->ncx*nbs->ncy];

    nbat->natoms = nbs->nc*nbs->napc;

    if (debug)
    {
        fprintf(debug,"ns napc %d naps %d super-cells: %d x %d y %d z %.1f maxz %d\n",
                nbs->napc,nbs->naps,nbs->nc,
                nbs->ncx,nbs->ncy,nbs->nc/((double)(nbs->ncx*nbs->ncy)),
                ncz_max);
        if (gmx_debug_at)
        {
            i = 0;
            for(cy=0; cy<nbs->ncy; cy++)
            {
                for(cx=0; cx<nbs->ncx; cx++)
                {
                    fprintf(debug," %2d",nbs->cxy_ind[i+1]-nbs->cxy_ind[i]);
                    i++;
                }
                fprintf(debug,"\n");
            }
        }
    }

    /* Make sure the work array for sorting is large enough */
    if (ncz_max*nbs->napc*SGSF > nbs->work[0].sort_work_nalloc)
    {
        for(t=0; t<nbs->nthread_max; t++)
        {
            nbs->work[t].sort_work_nalloc =
                over_alloc_large(ncz_max*nbs->napc*SGSF);
            srenew(nbs->work[t].sort_work,nbs->work[t].sort_work_nalloc);
        }
    }

    /* Now we know the dimensions we can fill the grid */
    for(i=0; i<n; i++)
    {
        nbs->a[nbs->cxy_ind[nbs->cell[i]]*nbs->napc+nbs->cxy_na[nbs->cell[i]]++] = i;
    }

    nthread = omp_get_max_threads();
#pragma omp parallel for schedule(static)
    for(t=0; t<nthread; t++)
    {
        sort_columns(nbs,n,x,nbat,
                     ((t+0)*nbs->ncx*nbs->ncy)/nthread,
                     ((t+1)*nbs->ncx*nbs->ncy)/nthread,
                     nbs->work[t].sort_work);
    }

    if (debug)
    {
        int c;

        nbs->nsubc_tot = 0;
        for(c=0; c<nbs->nc; c++)
        {
            nbs->nsubc_tot += nbs->nsubc[c];
        }
        fprintf(debug,"ns non-zero sub-cells: %d average atoms %.2f\n",
                nbs->nsubc_tot,n/(double)nbs->nsubc_tot);

        print_bbsizes(debug,nbs);
    }
}

static void nb_realloc_void(void **ptr,
                            int nbytes_copy,int nbytes_new,
                            gmx_nbat_alloc_t *ma,
                            gmx_nbat_free_t  *mf)
{
    void *ptr_new;

    ma(&ptr_new,nbytes_new);

    if (nbytes_copy > 0)
    {
        if (nbytes_new < nbytes_copy)
        {
            gmx_incons("In nb_realloc_void: new size less than copy size");
        }
        memcpy(ptr_new,*ptr,nbytes_copy);
    }
    if (*ptr != NULL)
    {
        mf(*ptr);
    }
    *ptr = ptr_new;
}

/* NOTE: does not preserve the contents! */
static void nb_realloc_int(int **ptr,int n,
                           gmx_nbat_alloc_t *ma,
                           gmx_nbat_free_t  *mf)
{
    if (*ptr != NULL)
    {
        mf(*ptr);
    }
    ma((void **)ptr,n*sizeof(**ptr));
}

/* NOTE: does not preserve the contents! */
static void nb_realloc_real(real **ptr,int n,
                            gmx_nbat_alloc_t *ma,
                            gmx_nbat_free_t  *mf)
{
    if (*ptr != NULL)
    {
        mf(*ptr);
    }
    ma((void **)ptr,n*sizeof(**ptr));
}

static void gmx_nb_atomdata_realloc(gmx_nb_atomdata_t *nbat,int n)
{
    nb_realloc_int(&nbat->type,n,nbat->alloc,nbat->free);
    if (nbat->XFormat != nbatXYZQ)
    {
        nb_realloc_real(&nbat->q,n,nbat->alloc,nbat->free);
    }
    nb_realloc_real(&nbat->x,n*nbat->xstride,nbat->alloc,nbat->free);
    nb_realloc_real(&nbat->f,n*nbat->xstride,nbat->alloc,nbat->free);
    nbat->nalloc = n;
    /* Zero f, since we always expect it to be zero (unless during use) */
    memset(nbat->f,0,nbat->nalloc*nbat->xstride*sizeof(*nbat->f));
}

void gmx_nbsearch_put_on_grid(gmx_nbsearch_t nbs,
                              int ePBC,matrix box,int n,rvec *x,
                              gmx_nb_atomdata_t *nbat)
{
    int nc_max;

    nbs->ePBC = ePBC;
    copy_mat(box,nbs->box);

    nc_max = set_grid_size_xy(nbs,n,nbs->box);

    if (n > nbs->cell_nalloc)
    {
        nbs->cell_nalloc = over_alloc_large(n);
        srenew(nbs->cell,nbs->cell_nalloc);
    }

    if (nc_max*nbs->napc > nbat->nalloc)
    {
        gmx_nb_atomdata_realloc(nbat,nc_max*nbs->napc);
    }

    calc_cell_indices(nbs,n,x,nbat);
}

static void get_cell_range(real b0,real b1,int nc,real s,real invs,
                           real d2,real r2,int *cf,int *cl)
{
    *cf = max((int)(b0*invs),0);
    
    while (*cf > 0 && d2 + sqr(b0 - (*cf-1+1)*s) < r2)
    {
        (*cf)--;
    }

    *cl = min((int)(b1*invs),nc-1);
    while (*cl < nc-1 && d2 + sqr((*cl+1)*s - b1) < r2)
    {
        (*cl)++;
    }
}

static real box_dist2(real bx0,real bx1,real by0,real by1,real bz0,real bz1,
                      const real *bb)
{
    real d2;
    real dl,dh,dm,dm0;

    d2 = 0;

    dl  = bx0 - bb[1];
    dh  = bb[0] - bx1;
    dm  = max(dl,dh);
    dm0 = max(dm,0);
    d2 += dm0*dm0;

    dl  = by0 - bb[3];
    dh  = bb[2] - by1;
    dm  = max(dl,dh);
    dm0 = max(dm,0);
    d2 += dm0*dm0;

    dl  = bz0 - bb[5];
    dh  = bb[4] - bz1;
    dm  = max(dl,dh);
    dm0 = max(dm,0);
    d2 += dm0*dm0;

    return d2;
}

static real subc_bb_dist2(int naps,
                          int si,const real *bb_i_ci,
                          int csj,const real *bb_j_all)
{
    const real *bb_i,*bb_j;
    real d2;
    real dl,dh,dm,dm0;

    bb_i = bb_i_ci  +  si*NNBSBB;
    bb_j = bb_j_all + csj*NNBSBB;

    d2 = 0;

    dl  = bb_i[0] - bb_j[1];
    dh  = bb_j[0] - bb_i[1];
    dm  = max(dl,dh);
    dm0 = max(dm,0);
    d2 += dm0*dm0;

    dl  = bb_i[2] - bb_j[3];
    dh  = bb_j[2] - bb_i[3];
    dm  = max(dl,dh);
    dm0 = max(dm,0);
    d2 += dm0*dm0;

    dl  = bb_i[4] - bb_j[5];
    dh  = bb_j[4] - bb_i[5];
    dm  = max(dl,dh);
    dm0 = max(dm,0);
    d2 += dm0*dm0;

    return d2;
}

static gmx_bool subc_in_range_x(int naps,
                                int si,const real *x_i,
                                int csj,int stride,const real *x_j,
                                real rl2)
{
    int  i,j,i0,j0;
    real d2;

    for(i=0; i<naps; i++)
    {
        i0 = (si*naps + i)*DIM;
        for(j=0; j<naps; j++)
        {
            j0 = (csj*naps + j)*stride;

            d2 = sqr(x_i[i0  ] - x_j[j0  ]) +
                 sqr(x_i[i0+1] - x_j[j0+1]) +
                 sqr(x_i[i0+2] - x_j[j0+2]);

            if (d2 < rl2)
            {
                return TRUE;
            }
        }
    }

    return FALSE;
}

static gmx_bool subc_in_range_sse8(int naps,
                                   int si,const real *x_i,
                                   int csj,int stride,const real *x_j,
                                   real rl2)
{
#if ( !defined(GMX_DOUBLE) && ( defined(GMX_IA32_SSE) || defined(GMX_X86_64_SSE) || defined(GMX_X86_64_SSE2) ) )
    __m128 ix_SSE0,iy_SSE0,iz_SSE0;
    __m128 ix_SSE1,iy_SSE1,iz_SSE1;
    __m128 jx0_SSE,jy0_SSE,jz0_SSE;
    __m128 jx1_SSE,jy1_SSE,jz1_SSE;

    __m128     dx_SSE0,dy_SSE0,dz_SSE0;
    __m128     dx_SSE1,dy_SSE1,dz_SSE1;
    __m128     dx_SSE2,dy_SSE2,dz_SSE2;
    __m128     dx_SSE3,dy_SSE3,dz_SSE3;

    __m128     rsq_SSE0;
    __m128     rsq_SSE1;
    __m128     rsq_SSE2;
    __m128     rsq_SSE3;

    __m128     wco_SSE0;
    __m128     wco_SSE1;
    __m128     wco_SSE2;
    __m128     wco_SSE3;
    __m128     wco_any_SSE01,wco_any_SSE23,wco_any_SSE;
    
    __m128 rc2_SSE;

    float      wco_any_array[7],*wco_any_align;

    int naps_sse;
    int j0,j1;

    rc2_SSE   = _mm_set1_ps(rl2);

    wco_any_align = (float *)(((size_t)(wco_any_array+3)) & (~((size_t)15)));

    naps_sse = 8/4;
    ix_SSE0 = _mm_load_ps(x_i+(si*naps_sse*DIM+0)*4);
    iy_SSE0 = _mm_load_ps(x_i+(si*naps_sse*DIM+1)*4);
    iz_SSE0 = _mm_load_ps(x_i+(si*naps_sse*DIM+2)*4);
    ix_SSE1 = _mm_load_ps(x_i+(si*naps_sse*DIM+3)*4);
    iy_SSE1 = _mm_load_ps(x_i+(si*naps_sse*DIM+4)*4);
    iz_SSE1 = _mm_load_ps(x_i+(si*naps_sse*DIM+5)*4);

    j0 = csj*naps;
    j1 = j0 + naps - 1;
    while (j0 < j1)
    {
        jx0_SSE = _mm_load1_ps(x_j+j0*stride+0);
        jy0_SSE = _mm_load1_ps(x_j+j0*stride+1);
        jz0_SSE = _mm_load1_ps(x_j+j0*stride+2);

        jx1_SSE = _mm_load1_ps(x_j+j1*stride+0);
        jy1_SSE = _mm_load1_ps(x_j+j1*stride+1);
        jz1_SSE = _mm_load1_ps(x_j+j1*stride+2);
        
        /* Calculate distance */
        dx_SSE0            = _mm_sub_ps(ix_SSE0,jx0_SSE);
        dy_SSE0            = _mm_sub_ps(iy_SSE0,jy0_SSE);
        dz_SSE0            = _mm_sub_ps(iz_SSE0,jz0_SSE);
        dx_SSE1            = _mm_sub_ps(ix_SSE1,jx0_SSE);
        dy_SSE1            = _mm_sub_ps(iy_SSE1,jy0_SSE);
        dz_SSE1            = _mm_sub_ps(iz_SSE1,jz0_SSE);
        dx_SSE2            = _mm_sub_ps(ix_SSE0,jx1_SSE);
        dy_SSE2            = _mm_sub_ps(iy_SSE0,jy1_SSE);
        dz_SSE2            = _mm_sub_ps(iz_SSE0,jz1_SSE);
        dx_SSE3            = _mm_sub_ps(ix_SSE1,jx1_SSE);
        dy_SSE3            = _mm_sub_ps(iy_SSE1,jy1_SSE);
        dz_SSE3            = _mm_sub_ps(iz_SSE1,jz1_SSE);

        /* rsq = dx*dx+dy*dy+dz*dz */
        rsq_SSE0           = gmx_mm_calc_rsq_ps(dx_SSE0,dy_SSE0,dz_SSE0);
        rsq_SSE1           = gmx_mm_calc_rsq_ps(dx_SSE1,dy_SSE1,dz_SSE1);
        rsq_SSE2           = gmx_mm_calc_rsq_ps(dx_SSE2,dy_SSE2,dz_SSE2);
        rsq_SSE3           = gmx_mm_calc_rsq_ps(dx_SSE3,dy_SSE3,dz_SSE3);

        wco_SSE0           = _mm_cmplt_ps(rsq_SSE0,rc2_SSE);
        wco_SSE1           = _mm_cmplt_ps(rsq_SSE1,rc2_SSE);
        wco_SSE2           = _mm_cmplt_ps(rsq_SSE2,rc2_SSE);
        wco_SSE3           = _mm_cmplt_ps(rsq_SSE3,rc2_SSE);
        
        wco_any_SSE01      = _mm_or_ps(wco_SSE0,wco_SSE1);
        wco_any_SSE23      = _mm_or_ps(wco_SSE2,wco_SSE3);
        wco_any_SSE        = _mm_or_ps(wco_any_SSE01,wco_any_SSE23);

        _mm_store_ps(wco_any_align,wco_any_SSE);

        if (wco_any_align[0] != 0 ||
            wco_any_align[1] != 0 || 
            wco_any_align[2] != 0 ||
            wco_any_align[3] != 0)
        {
            return TRUE;
        }
        
        j0++;
        j1--;
    }
    return FALSE;

#else
    /* No SSE */
    gmx_incons("SSE function called without SSE support");

    return TRUE;
#endif
}

static void check_subcell_list_space(gmx_nblist_t *nbl,int npair)
{
    /* We can have maximally npair*NSUBCELL sj lists */
    /* Note that we need one extra sj entry at the end for the last index */
    if (nbl->nsj + 1 + npair*NSUBCELL > nbl->sj_nalloc)
    {
        nbl->sj_nalloc = over_alloc_small(nbl->nsj + 1 + npair*NSUBCELL);
        nb_realloc_void((void **)&nbl->sj,
                        (nbl->nsj+1)*sizeof(*nbl->sj),
                        nbl->sj_nalloc*sizeof(*nbl->sj),
                        nbl->alloc,nbl->free);
    }

    /* We can have maximally npair*NSUBCELL*NSUBCELL si lists */
    if (nbl->nsi + npair*NSUBCELL*NSUBCELL > nbl->si_nalloc)
    {
        nbl->si_nalloc = over_alloc_small(nbl->nsi + npair*NSUBCELL*NSUBCELL);
        nb_realloc_void((void **)&nbl->si,
                        nbl->nsi*sizeof(*nbl->si),
                        nbl->si_nalloc*sizeof(*nbl->si),
                        nbl->alloc,nbl->free);
    }
}

static void nblist_alloc_aligned(void **ptr,size_t nbytes)
{
    *ptr = save_calloc_aligned("ptr",__FILE__,__LINE__,nbytes,1,16);
}

static void nblist_free_aligned(void *ptr)
{
    sfree_aligned(ptr);
}

void gmx_nblist_init(gmx_nblist_t *nbl,
                     gmx_nbat_alloc_t *alloc,
                     gmx_nbat_free_t  *free)
{
    if (alloc == NULL)
    {
        nbl->alloc = nblist_alloc_aligned;
    }
    else
    {
        nbl->alloc = alloc;
    }
    if (free == NULL)
    {
        nbl->free = nblist_free_aligned;
    }
    else
    {
        nbl->free = free;
    }

    nbl->napc        = 0;
    nbl->naps        = 0;
    nbl->nci         = 0;
    nbl->ci          = NULL;
    nbl->ci_nalloc   = 0;
    nbl->nsj         = 0;
    /* We need one element extra in sj, so alloc initially with 1 */
    nbl->sj_nalloc   = over_alloc_large(1);
    nbl->sj          = NULL;
    nb_realloc_void((void **)&nbl->sj,0,nbl->sj_nalloc*sizeof(*nbl->sj),
                    nbl->alloc,nbl->free);
    nbl->nsi         = 0;
    nbl->si          = NULL;
    nbl->si_nalloc   = 0;

    snew(nbl->work,1);
    snew_aligned(nbl->work->bb_si,NSUBCELL*NNBSBB,16);
    snew_aligned(nbl->work->x_si,NBL_NAPC_MAX*DIM,16);
}

static void print_nblist_statistics(FILE *fp,const gmx_nblist_t *nbl,
                                    const gmx_nbsearch_t nbs,real rl)
{
    int *count;
    int lmax,i,j,cjp,cj,l;
    double sl;

    fprintf(fp,"nbl nci %d nsj %d nsi %d\n",nbl->nci,nbl->nsj,nbl->nsi);
    fprintf(fp,"nbl naps %d rl %g ncp %d per cell %.1f atoms %.1f ratio %.2f\n",
            nbl->naps,rl,nbl->nsi,nbl->nsi/(double)nbs->nsubc_tot,
            nbl->nsi/(double)nbs->nsubc_tot*nbs->naps,
            nbl->nsi/(double)nbs->nsubc_tot*nbs->naps/(0.5*4.0/3.0*M_PI*rl*rl*rl*nbs->nsubc_tot*nbs->naps/det(nbs->box)));

    fprintf(fp,"nbl average j super cell list length %.1f\n",
            nbl->nsj/(double)nbl->nci);
    fprintf(fp,"nbl average i sub cell list length %.1f\n",
            nbl->nsi/(double)nbl->nsj);
}

static void make_subcell_list(const gmx_nbsearch_t nbs,
                              gmx_nblist_t *nbl,
                              int ci,int cj,
                              gmx_bool ci_equals_cj,
                              int stride,const real *x,
                              real rl2,real rbb2)
{
    int  npair;
    int  sj,si1,si,csj;
    const real *bb_si,*x_si;
    real d2;
    gmx_bool InRange;
#define ISUBCELL_GROUP 1

    bb_si = nbl->work->bb_si;
    x_si  = nbl->work->x_si;

    for(sj=0; sj<nbs->nsubc[cj]; sj++)
    {
        csj = cj*NSUBCELL + sj;

        if (!nbl->TwoWay && ci_equals_cj)
        {
            si1 = sj + 1;
        }
        else
        {
            si1 = nbs->nsubc[ci];
        }

        npair = 0;
        for(si=0; si<si1; si++)
        {
            d2 = subc_bb_dist2(nbs->naps,si,bb_si,csj,nbs->bb);

            if (nbs->subc_dc == NULL)
            {
                InRange = (d2 < rl2);
            }
            else
            {
                InRange = (d2 < rbb2 ||
                           (d2 < rl2 &&
                            nbs->subc_dc(nbs->naps,si,x_si,
                                         csj,stride,x,rl2)));
            }

            if (InRange)
            {
                nbl->si[nbl->nsi].si = ci*NSUBCELL + si;
                if (ci_equals_cj && si == sj)
                {
                    if (nbl->TwoWay)
                    {
                        /* Only minor != major bits set */
                        nbl->si[nbl->nsi].excl = 0x7fbfdfeff7fbfdfeL;
                    }
                    else
                    {
                        /* Only minor > major bits set */
                        nbl->si[nbl->nsi].excl = 0x80c0e0f0f8fcfe;
                    }
                }
                else
                {
                    /* All 8x8 bits set */
                    nbl->si[nbl->nsi].excl = 0xffffffffffffffffL;
                }
                nbl->nsi++;
                npair++;
                si += ISUBCELL_GROUP - 1;
            }
        }

        if (npair > 0)
        {
#if ISUBCELL_GROUP > 1
            /* Check for indexing out of the i super cell */
            if (nbl->si[nbl->nsi-1] > (ci + 1)*NSUBCELL - ISUBCELL_GROUP)
            {
                /* Move back all i sub cells until it fits */
                nbl->si[nbl->nsi-1] = (ci + 1)*NSUBCELL - ISUBCELL_GROUP;
                si = nbl->nsi - 2;
                while (si >= nbl->sj[nbl->nsj].si_ind &&
                       nbl->si[si] > nbl->si[si+1] - ISUBCELL_GROUP)
                {
                    nbl->si[si] = nbl->si[si+1] - ISUBCELL_GROUP;
                    si--;
                }
            }
#endif

            /* We have a useful sj entry,
             * close it now.
             */
            nbl->sj[nbl->nsj].sj = cj*NSUBCELL + sj;
            nbl->nsj++;
            /* Set the closing index in the j sub-cell list */
            nbl->sj[nbl->nsj].si_ind = nbl->nsi;

            /* Increase the closing index in i super-cell list */
            nbl->ci[nbl->nci].sj_ind_end = nbl->nsj;
        }
    }
}

static void nb_realloc_ci(gmx_nblist_t *nbl,int n)
{
    nbl->ci_nalloc = over_alloc_small(n);
    nb_realloc_void((void **)&nbl->ci,
                    nbl->nci*sizeof(*nbl->ci),
                    nbl->ci_nalloc*sizeof(*nbl->ci),
                    nbl->alloc,nbl->free);
}

static void new_ci_entry(gmx_nblist_t *nbl,int ci,int shift)
{
    if (nbl->nci + 1 > nbl->ci_nalloc)
    {
        nb_realloc_ci(nbl,nbl->nci+1);
    }
    nbl->ci[nbl->nci].ci           = ci;
    nbl->ci[nbl->nci].shift        = shift;
    nbl->ci[nbl->nci].sj_ind_start = (nbl->nci == 0 ? 0 :
                                          nbl->ci[nbl->nci-1].sj_ind_end);
    nbl->ci[nbl->nci].sj_ind_end   = nbl->ci[nbl->nci].sj_ind_start;
}

static void close_ci_entry(gmx_nblist_t *nbl,int max_jlist_av,int nc_bal)
{
    int jlen,tlen;
    int nb,b;
    int max_jlist;

    /* All content of the new ci entry have already been filled correctly,
     * we only need to increase the count here (for non empty lists).
     */
    jlen = nbl->ci[nbl->nci].sj_ind_end - nbl->ci[nbl->nci].sj_ind_start;
    if (jlen > 0)
    {
        nbl->nci++;

        if (max_jlist_av > 0)
        {
            /* The first ci blocks should be larger, to avoid overhead.
             * The last ci blocks should be smaller, to improve load balancing.
             */
            max_jlist = max(1,
                            max_jlist_av*nc_bal*3/(2*(nbl->nci - 1 + nc_bal)));

            if (jlen > max_jlist)
            {
                /* Split ci in the minimum number of blocks <=jlen */
                nb = (jlen + max_jlist - 1)/max_jlist;
                /* Make blocks similar sized, last one smallest */
                tlen = (jlen + nb - 1)/nb;
                
                if (nbl->nci + nb - 1 > nbl->ci_nalloc)
                {
                    nb_realloc_ci(nbl,nbl->nci+nb-1);
                }
                
                /* Set the end of the last block to the current end */
                nbl->ci[nbl->nci+nb-2].sj_ind_end = nbl->ci[nbl->nci-1].sj_ind_end;
                for(b=1; b<nb; b++)
                {
                    nbl->ci[nbl->nci-1].sj_ind_end =
                        nbl->ci[nbl->nci-1].sj_ind_start + tlen;
                    nbl->ci[nbl->nci].sj_ind_start = nbl->ci[nbl->nci-1].sj_ind_end;
                    nbl->ci[nbl->nci].ci    = nbl->ci[nbl->nci-1].ci;
                    nbl->ci[nbl->nci].shift = nbl->ci[nbl->nci-1].shift;
                    nbl->nci++;
                }
            }
        }
    }
}

static void clear_nblist(gmx_nblist_t *nbl)
{
    nbl->nci = 0;
    nbl->nsj = 0;
    nbl->sj[0].si_ind = 0;
    nbl->nsi = 0;
}

static void set_subcell_i_bb(const real *bb,int ci,
                             real shx,real shy,real shz,
                             real *bb_si)
{
    int ia,i;
    
    ia = ci*NSUBCELL*NNBSBB;
    for(i=0; i<NSUBCELL*NNBSBB; i+=NNBSBB)
    {
        bb_si[i+0] = bb[ia+i+0] + shx;
        bb_si[i+1] = bb[ia+i+1] + shx;
        bb_si[i+2] = bb[ia+i+2] + shy;
        bb_si[i+3] = bb[ia+i+3] + shy;
        bb_si[i+4] = bb[ia+i+4] + shz;
        bb_si[i+5] = bb[ia+i+5] + shz;
    }
}

static void set_subcell_i_x(const gmx_nbsearch_t nbs,int ci,
                            real shx,real shy,real shz,
                            int stride,const real *x,
                            real *x_si)
{
    int si,io,ia,i,j;

    if (nbs->subc_dc == subc_in_range_sse8)
    {
        for(si=0; si<NSUBCELL; si++)
        {
            for(i=0; i<nbs->naps; i+=4)
            {
                io = si*nbs->naps + i;
                ia = ci*NSUBCELL*nbs->naps + io;
                for(j=0; j<4; j++)
                {
                    x_si[io*3 + j + 0] = x[(ia+j)*stride+0] + shx;
                    x_si[io*3 + j + 4] = x[(ia+j)*stride+1] + shy;
                    x_si[io*3 + j + 8] = x[(ia+j)*stride+2] + shz;
                }
            }
        }
    }
    else
    {
        ia = ci*NSUBCELL*nbs->naps;
        for(i=0; i<nbs->napc; i++)
        {
            x_si[i*3 + 0] = x[(ia+i)*stride + 0] + shx;
            x_si[i*3 + 1] = x[(ia+i)*stride + 1] + shy;
            x_si[i*3 + 2] = x[(ia+i)*stride + 2] + shz;
        }
    }
}

static int get_max_jlist(const gmx_nbsearch_t nbs,
                         gmx_nblist_t *nbl,
                         int min_ci_balanced)
{
    real xy_diag,r_eff_sup;
    int  nj_est,nparts;
    int  max_jlist;

    /* The average diagonal of a super cell */
    xy_diag = sqrt(sqr(nbs->box[XX][XX]/nbs->ncx) +
                   sqr(nbs->box[YY][YY]/nbs->ncy) +
                   sqr(nbs->box[ZZ][ZZ]*nbs->ncx*nbs->ncy/nbs->nc));

    /* The formulas below are a heuristic estimate of the average nj per ci*/
    r_eff_sup = nbl->rlist + 0.4*xy_diag;
    
    nj_est = (int)(0.5*4.0/3.0*M_PI*pow(r_eff_sup,3)*nbs->atom_density/nbs->naps + 0.5);

    if (min_ci_balanced <= 0 || nbs->nc >= min_ci_balanced)
    {
        /* We don't need to worry */
        max_jlist = -1;
    }
    else
    {
        /* Estimate the number of parts we need to cut each full list
         * for one i super cell into.
         */
        nparts = (min_ci_balanced + nbs->nc - 1)/nbs->nc;
        /* Thus the (average) maximum j-list size should be as follows */
        max_jlist = max(1,(nj_est + nparts - 1)/nparts);
    }

    if (debug)
    {
        fprintf(debug,"nbl nj estimate %d, max_jlist %d\n",nj_est,max_jlist);
    }

    return max_jlist;
}

static void print_nblist_ci_sj(FILE *fp,const gmx_nblist_t *nbl)
{
    int i,j;

    for(i=0; i<nbl->nci; i++)
    {
        fprintf(fp,"ci %4d  shift %2d  nsj %2d\n",
                nbl->ci[i].ci,nbl->ci[i].shift,
                nbl->ci[i].sj_ind_end - nbl->ci[i].sj_ind_start);

        for(j=nbl->ci[i].sj_ind_start; j<nbl->ci[i].sj_ind_end; j++)
        {
            fprintf(fp,"  sj %5d  nsi %3d\n",
                    nbl->sj[j].sj,nbl->sj[j+1].si_ind - nbl->sj[j].si_ind);
#if 0
            {
                int k;
                fprintf(fp,"    si");
                for(k=nbl->sj[j].si_ind; k<nbl->sj[j+1].si_ind; k++)
                {
                    fprintf(fp," %5d",nbl->si[k].si);
                }
                fprintf(fp,"\n");
            }
#endif
        }
    }
}

static void combine_nblists(int nnbl,const gmx_nblist_t *nbl,
                            gmx_nblist_t *nblc)
{
    int nci,nsj,nsi;
    int i,j;
    int sj_offset,si_offset;
    const gmx_nblist_t *nbli;

    nci = nblc->nci;
    nsj = nblc->nsj;
    nsi = nblc->nsi;
    for(i=0; i<nnbl; i++)
    {
        nci += nbl[i].nci;
        nsj += nbl[i].nsj;
        nsi += nbl[i].nsi;
    }

    if (nci > nblc->ci_nalloc)
    {
        nb_realloc_ci(nblc,nci);
    }
    if (nsj + 1 > nblc->sj_nalloc)
    {
        nblc->sj_nalloc = over_alloc_small(nsj+1);
        nb_realloc_void((void **)&nblc->sj,
                        (nblc->nsj+1)*sizeof(*nblc->sj),
                        nblc->sj_nalloc*sizeof(*nblc->sj),
                        nblc->alloc,nblc->free);
    }
    if (nsi > nblc->si_nalloc)
    {
        nblc->si_nalloc = over_alloc_small(nsi);
        nb_realloc_void((void **)&nblc->si,
                        nblc->nsi*sizeof(*nblc->si),
                        nblc->si_nalloc*sizeof(*nblc->si),
                        nblc->alloc,nblc->free);
    }

    for(i=0; i<nnbl; i++)
    {
        sj_offset = nblc->nsj;
        si_offset = nblc->nsi;

        nbli = &nbl[i];

        /* We could instead omp prallelizing the two loops below
         * parallelize the copy of integral parts of the nblist.
         * However this requires a lot more bookkeeping and does not
         * lead to a performance improvement.
         */
        /* The ci list copy is probably not work parallelizing */
        for(j=0; j<nbli->nci; j++)
        {
            nblc->ci[nblc->nci] = nbli->ci[j];
            nblc->ci[nblc->nci].sj_ind_start += sj_offset;
            nblc->ci[nblc->nci].sj_ind_end   += sj_offset;
            nblc->nci++;
        }
#pragma omp parallel for schedule(static)
        for(j=0; j<nbli->nsj; j++)
        {
            nblc->sj[nblc->nsj+j].sj       = nbli->sj[j].sj;
            nblc->sj[nblc->nsj+j+1].si_ind = nbli->sj[j+1].si_ind + si_offset;
        }
        nblc->nsj += nbli->nsj;
#pragma omp parallel for schedule(static)
        for(j=0; j<nbli->nsi; j++)
        {
            nblc->si[nblc->nsi+j] = nbli->si[j];
        }
        nblc->nsi += nbli->nsi;
    }
}

static void gmx_nbsearch_make_nblist_part(const gmx_nbsearch_t nbs,
                                          gmx_nbs_work_t *work,
                                          const gmx_nb_atomdata_t *nbat,
                                          real rcut,real rlist,
                                          int min_ci_balanced,
                                          int ci_start,int ci_end,
                                          gmx_nblist_t *nbl)
{
    gmx_bool bDomDec;
    int  max_jlist;
    matrix box;
    real rl2,rbb2;
    int  d,ci,ci_xy,ci_x,ci_y,cj;
    ivec shp;
    int  tx,ty,tz;
    int  shift;
    gmx_bool bMakeList;
    real shx,shy,shz;
    real *bbcz,*bb;
    real bx0,bx1,by0,by1,bz0,bz1;
    real bz1_frac;
    real d2z,d2zx,d2zxy,d2xy;
    int  cxf,cxl,cyf,cyf_x,cyl;
    int  cx,cy;
    int  c0,c1,cs,cf,cl;

    bDomDec = FALSE;

    nbl->napc = nbs->napc;
    nbl->naps = nbs->naps;

    /* Currently this code only makes two-way lists */
    nbl->TwoWay = FALSE;

    nbl->rcut   = rcut;
    nbl->rlist  = rlist;

    max_jlist = get_max_jlist(nbs,nbl,min_ci_balanced);

    clear_nblist(nbl);

    copy_mat(nbs->box,box);

    rl2 = nbl->rlist*nbl->rlist;

    if (nbs->subc_dc == NULL)
    {
        rbb2 = rl2;
    }
    else
    {
        /* If the distance between two sub-cell bounding boxes is less
         * than the nblist cut-off minus half of the average x/y diagonal
         * spacing of the sub-cells, do not check the distance between
         * all particle pairs in the sub-cell, since this pairs is very
         * likely to have atom pairs within the cut-off.
         */
        rbb2 = sqr(max(0,
                       nbl->rlist -
                       0.5*sqrt(sqr(box[XX][XX]/(nbs->ncx*NSUBCELL_X)) +
                                sqr(box[YY][YY]/(nbs->ncy*NSUBCELL_Y)))));
    }
    if (debug)
    {
        fprintf(debug,"nbl bounding box only distance %f\n",sqrt(rbb2));
    }

    /* Set the shift range */
    for(d=0; d<DIM; d++)
    {
        /* We need to add these domain shift limits for DD
        sh0[d] = -1;
        sh1[d] = 1;
        */
        /* Check if we need periodicity shifts.
         * Without PBC or with domain decomposition we don't need them.
         */
        /*
        if (d >= ePBC2npbcdim(fr->ePBC) || (bDomDec && dd->nc[d] > 1))
        */
        if (d >= ePBC2npbcdim(nbs->ePBC))
        {
            shp[d] = 0;
        }
        else
        {
            if (d == XX &&
                box[XX][XX] - fabs(box[YY][XX]) - fabs(box[ZZ][XX]) < sqrt(rl2))
            {
                shp[d] = 2;
            }
            else
            {
                shp[d] = 1;
            }
        }
    }

    bbcz = nbs->bbcz;
    bb   = nbs->bb;

    ci_xy = 0;
    for(ci=ci_start; ci<ci_end; ci++)
    {
        while (ci >= nbs->cxy_ind[ci_xy+1])
        {
            ci_xy++;
        }
        ci_x = ci_xy/nbs->ncy;
        ci_y = ci_xy - ci_x*nbs->ncy;

        /* Loop over shift vectors in three dimensions */
        for (tz=-shp[ZZ]; tz<=shp[ZZ]; tz++)
        {
            shz = tz*box[ZZ][ZZ];

            bz0 = bbcz[ci*NNBSBB_D  ] + shz;
            bz1 = bbcz[ci*NNBSBB_D+1] + shz;

            if (tz == 0)
            {
                d2z = 0;
            }
            else if (tz < 0)
            {
                d2z = sqr(bz1);
            }
            else
            {
                d2z = sqr(bz0 - box[ZZ][ZZ]);
            }

            if (d2z >= rl2)
            {
                continue;
            }

            bz1_frac =
                bz1/((real)(nbs->cxy_ind[ci_xy+1] - nbs->cxy_ind[ci_xy]));
            if (bz1_frac < 0)
            {
                bz1_frac = 0;
            }
            /* The check with bz1_frac close to or larger than 1 comes later */

            for (ty=-shp[YY]; ty<=shp[YY]; ty++)
            {
                shy = ty*box[YY][YY] + tz*box[ZZ][YY];
            
                by0 = (ci_y  )*nbs->sy + shy;
                by1 = (ci_y+1)*nbs->sy + shy;

                get_cell_range(by0,by1,nbs->ncy,nbs->sy,nbs->inv_sy,d2z,rl2,
                               &cyf,&cyl);

                for (tx=-shp[XX]; tx<=shp[XX]; tx++)
                {
                    shift = XYZ2IS(tx,ty,tz);

#ifdef NSBOX_SHIFT_BACKWARD
                    if (shift > CENTRAL)
                    {
                        continue;
                    }
#endif

                    shx = tx*box[XX][XX] + ty*box[YY][XX] + tz*box[ZZ][XX];

                    bx0 = (ci_x  )*nbs->sx + shx;
                    bx1 = (ci_x+1)*nbs->sx + shx;

                    get_cell_range(bx0,bx1,nbs->ncx,nbs->sx,nbs->inv_sx,d2z,rl2,
                                   &cxf,&cxl); 

                    new_ci_entry(nbl,ci,shift);

#ifndef NSBOX_SHIFT_BACKWARD
                    if (!nbl->TwoWay && cxf < ci_x)
#else
                    if (!nbl->TwoWay && shift == CENTRAL && cxf < ci_x)
#endif
                    {
                        /* Leave the pairs with i > j.
                         * x is the major index, so skip half of it.
                         */
                        cxf = ci_x;
                    }

                    set_subcell_i_bb(nbs->bb,ci,shx,shy,shz,nbl->work->bb_si);

                    if (nbs->subc_dc != NULL)
                    {
                        set_subcell_i_x(nbs,ci,shx,shy,shz,
                                        nbat->xstride,nbat->x,nbl->work->x_si);
                    }

                    for(cx=cxf; cx<=cxl; cx++)
                    {
                        d2zx = d2z;
                        if (cx*nbs->sx > bx1)
                        {
                            d2zx += sqr(cx*nbs->sx - bx1);
                        }
                        else if ((cx+1)*nbs->sx < bx0)
                        {
                            d2zx += sqr((cx+1)*nbs->sx - bx0);
                        }

#ifndef NSBOX_SHIFT_BACKWARD
                        if (!nbl->TwoWay && cx == 0 && cyf < ci_y)
#else
                        if (!nbl->TwoWay &&
                            cx == 0 && shift == CENTRAL && cyf < ci_y)
#endif
                        {
                            /* Leave the pairs with i > j.
                             * Skip half of y when i and j have the same x.
                             */
                            cyf_x = ci_y;
                        }
                        else
                        {
                            cyf_x = cyf;
                        }

                        for(cy=cyf_x; cy<=cyl; cy++)
                        {
                            c0 = nbs->cxy_ind[cx*nbs->ncy+cy];
                            c1 = nbs->cxy_ind[cx*nbs->ncy+cy+1];
#ifdef NSBOX_SHIFT_BACKWARD
                            if (!nbl->TwoWay && shift == CENTRAL && c0 < ci)
                            {
                                c0 = ci;
                            }
#endif

                            d2zxy = d2zx;
                            if (cy*nbs->sy > by1)
                            {
                                d2zxy += sqr(cy*nbs->sy - by1);
                            }
                            else if ((cy+1)*nbs->sy < by0)
                            {
                                d2zxy += sqr((cy+1)*nbs->sy - by0);
                            }
                            if (c1 > c0 && d2zxy < rl2)
                            {
                                cs = c0 + (int)(bz1_frac*(c1 - c0));
                                if (cs >= c1)
                                {
                                    cs = c1 - 1;
                                }

                                d2xy = d2zxy - d2z;

                                /* Find the lowest cell that can possibly
                                 * be within range.
                                 */
                                cf = cs;
                                while(cf > c0 &&
                                      (bbcz[cf*NNBSBB_D+1] >= bz0 ||
                                       d2xy + sqr(bbcz[cf*NNBSBB_D+1] - bz0) < rl2))
                                {
                                    cf--;
                                }

                                /* Find the highest cell that can possibly
                                 * be within range.
                                 */
                                cl = cs;
                                while(cl < c1-1 &&
                                      (bbcz[cl*NNBSBB_D] <= bz1 ||
                                       d2xy + sqr(bbcz[cl*NNBSBB_D] - bz1) < rl2))
                                {
                                    cl++;
                                }

#ifdef NSBOX_REFCODE
                                {
                                    /* Simple reference code */
                                    int k;
                                    cf = c1;
                                    cl = -1;
                                    for(k=c0; k<c1; k++)
                                    {
                                        if (box_dist2(bx0,bx1,by0,by1,bz0,bz1,
                                                      bb+k*NNBSBB) < rl2 &&
                                            k < cf)
                                        {
                                            cf = k;
                                        }
                                        if (box_dist2(bx0,bx1,by0,by1,bz0,bz1,
                                                      bb+k*NNBSBB) < rl2 &&
                                            k > cl)
                                        {
                                            cl = k;
                                        }
                                    }
                                }
#endif

                                /* We want each atom/cell pair only once,
                                 * only use cj >= ci.
                                 */
#ifndef NSBOX_SHIFT_BACKWARD
                                cf = max(cf,ci);
#else
                                if (shift == CENTRAL)
                                {
                                    cf = max(cf,ci);
                                }
#endif

                                if (cf <= cl)
                                {
                                    check_subcell_list_space(nbl,cl-cf+1);

                                    for(cj=cf; cj<=cl; cj++)
                                    {
                                        make_subcell_list(nbs,nbl,ci,cj,
                                                          (shift == CENTRAL && ci == cj),
                                                          nbat->xstride,nbat->x,
                                                          rl2,rbb2);
                                    }
                                }
                            }
                        }  
                    }
        
                    close_ci_entry(nbl,max_jlist,min_ci_balanced);
                }
            }
        }
    }
    
    if (debug)
    {
        print_nblist_statistics(debug,nbl,nbs,rlist);

        if (gmx_debug_at)
        {
            //print_nblist_ci_sj(debug,nbl);
        }
    }
}

void gmx_nbsearch_make_nblist(const gmx_nbsearch_t nbs,
                              const gmx_nb_atomdata_t *nbat,
                              real rcut,real rlist,
                              int min_ci_balanced,
                              int nnbl,gmx_nblist_t *nbl,
                              gmx_bool CombineNBLists)
{
    int nth,th;

    if (debug)
    {
        fprintf(debug,"ns making %d nblists\n",nnbl);
    }
#pragma omp parallel for schedule(static)
    for(th=0; th<nnbl; th++)
    {
        /* Divide the i super cell equally over the nblists */
        gmx_nbsearch_make_nblist_part(nbs,&nbs->work[th],nbat,
                                      rcut,rlist,min_ci_balanced,
                                      ((th+0)*nbs->nc)/nnbl,
                                      ((th+1)*nbs->nc)/nnbl,
                                      &nbl[th]);
    }

    if (CombineNBLists && nnbl > 1)
    {
        combine_nblists(nnbl-1,nbl+1,nbl);
    }

    if (debug)
    {
        print_nblist_statistics(debug,&nbl[0],nbs,rlist);
    }
    if (gmx_debug_at)
    {
        print_nblist_ci_sj(debug,nbl);
    }
}

void gmx_nb_atomdata_init(gmx_nb_atomdata_t *nbat,int nbatXFormat,
                          int ntype,const real *nbfp,
                          gmx_nbat_alloc_t *alloc,
                          gmx_nbat_free_t  *free)
{
    int i,j;

    if (alloc == NULL)
    {
        nbat->alloc = nblist_alloc_aligned;
    }
    else
    {
        nbat->alloc = alloc;
    }
    if (free == NULL)
    {
        nbat->free = nblist_free_aligned;
    }
    else
    {
        nbat->free = free;
    }

    if (debug)
    {
        fprintf(debug,"There are %d atom types in the system, adding one for gmx_nb_atomdata_t\n",ntype);
    }
    nbat->ntype = ntype + 1;
    nbat->alloc((void **)&nbat->nbfp,
                nbat->ntype*nbat->ntype*2*sizeof(*nbat->nbfp));
    for(i=0; i<nbat->ntype; i++)
    {
        for(j=0; j<nbat->ntype; j++)
        {
            if (i < ntype && j < ntype)
            {
                nbat->nbfp[(i*nbat->ntype+j)*2  ] = nbfp[(i*ntype+j)*2  ];
                nbat->nbfp[(i*nbat->ntype+j)*2+1] = nbfp[(i*ntype+j)*2+1];
            }
            else
            {
                /* Add zero parameters for the additional dummy atom type */
                nbat->nbfp[(i*nbat->ntype+j)*2  ] = 0;
                nbat->nbfp[(i*nbat->ntype+j)*2+1] = 0;
            }
        }
    }

    nbat->natoms  = 0;
    nbat->type    = NULL;
    nbat->XFormat = nbatXFormat;
    nbat->q       = NULL;
    nbat->xstride = (nbatXFormat == nbatXYZQ ? 4 : 3);
    nbat->x       = NULL;
    nbat->nalloc  = 0;
}

void gmx_nb_atomdata_set_atomtypes(gmx_nb_atomdata_t *nbat,
                                   const gmx_nbsearch_t nbs,
                                   const int *type)
{
    int i,ncz,ash;

    /* Loop over all columns and copy and fill */
    for(i=0; i<nbs->ncx*nbs->ncy; i++)
    {
        ncz = nbs->cxy_ind[i+1] - nbs->cxy_ind[i];
        ash = nbs->cxy_ind[i]*nbs->napc;

        copy_int_to_nbat_int(nbs->a+ash,nbs->cxy_na[i],ncz*nbs->napc,
                             type,nbat->ntype-1,nbat->type+ash);
    }
}

void gmx_nb_atomdata_set_charges(gmx_nb_atomdata_t *nbat,
                                 const gmx_nbsearch_t nbs,
                                 const real *charge)
{
    int  cxy,ncz,ash,na,na_round,i,j;
    real *q;

    /* Loop over all columns and copy and fill */
    for(cxy=0; cxy<nbs->ncx*nbs->ncy; cxy++)
    {
        ash = nbs->cxy_ind[cxy]*nbs->napc;
        na  = nbs->cxy_na[cxy];
        na_round = (nbs->cxy_ind[cxy+1] - nbs->cxy_ind[cxy])*nbs->napc;

        if (nbat->XFormat == nbatXYZQ)
        {
            q = nbat->x + ash*nbat->xstride + 3;
            for(i=0; i<na; i++)
            {
                *q = charge[nbs->a[ash+i]];
                q += 4;
            }
            /* Complete the partially filled last cell with zeros */
            for(; i<na_round; i++)
            {
                *q = 0;
                q += 4;
            }
        }
        else
        {
            q = nbat->q + ash;
            for(i=0; i<na; i++)
            {
                *q = charge[nbs->a[ash+i]];
                q++;
            }
            /* Complete the partially filled last cell with zeros */
            for(; i<na_round; i++)
            {
                *q = 0;
                q++;
            }
        }
    }
}

void gmx_nb_atomdata_copy_x_to_nbat_x(const gmx_nbsearch_t nbs,
                                      rvec *x,
                                      gmx_nb_atomdata_t *nbat)
{
    int cxy,na,ash;

#pragma omp parallel for schedule(static) private(na,ash)
    for(cxy=0; cxy<nbs->ncx*nbs->ncy; cxy++)
    {
        na  = nbs->cxy_na[cxy];
        ash = nbs->cxy_ind[cxy]*nbs->napc;

        /* We fill only the real particle locations.
         * We assume the filling entries at the end have been
         * properly set before during ns.
         */
        copy_rvec_to_nbat_real(nbs->a+ash,na,na,x,
                               nbat->XFormat,nbat->x+ash*nbat->xstride,
                               0,0,0);
    }
}

static void gmx_nb_atomdata_add_nbat_f_to_f_part(const gmx_nbsearch_t nbs,
                                                 gmx_nb_atomdata_t *nbat,
                                                 int cxy0,int cxy1,
                                                 rvec *f)
{
    int  cxy,na,ash,i;
    const int  *a;
    real *fnb;

    /* Loop over all columns and copy and fill */
    for(cxy=cxy0; cxy<cxy1; cxy++)
    {
        na  = nbs->cxy_na[cxy];
        ash = nbs->cxy_ind[cxy]*nbs->napc;

        a   = nbs->a + ash;
        fnb = nbat->f + ash*nbat->xstride;

        switch (nbat->XFormat)
        {
        case nbatXYZ:
            for(i=0; i<na; i++)
            {
                f[a[i]][XX] += fnb[i*3];
                f[a[i]][YY] += fnb[i*3+1];
                f[a[i]][ZZ] += fnb[i*3+2];

                fnb[i*3]   = 0;
                fnb[i*3+1] = 0;
                fnb[i*3+2] = 0;
            }
            break;
        case nbatXYZQ:
            for(i=0; i<na; i++)
            {
                f[a[i]][XX] += fnb[i*4];
                f[a[i]][YY] += fnb[i*4+1];
                f[a[i]][ZZ] += fnb[i*4+2];

                fnb[i*4]   = 0;
                fnb[i*4+1] = 0;
                fnb[i*4+2] = 0;
            }
            break;
        default:
            gmx_incons("Unsupported stride");
        }
    }
}

void gmx_nb_atomdata_add_nbat_f_to_f(const gmx_nbsearch_t nbs,
                                     gmx_nb_atomdata_t *nbat,
                                     rvec *f)
{
    int nth,th;

    nth = omp_get_max_threads();
#pragma omp parallel for schedule(static)
    for(th=0; th<nth; th++)
    {
        gmx_nb_atomdata_add_nbat_f_to_f_part(nbs,nbat,
                                             ((th+0)*nbs->ncx*nbs->ncy)/nth,
                                             ((th+1)*nbs->ncx*nbs->ncy)/nth,
                                             f);
    }
}
