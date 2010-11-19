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

/* Neighbor search box upper and lower bound in x,y,z: total 6 reals */
#define NNBSBB_D 2
#define NNBSBB   (3*NNBSBB_D)

typedef struct gmx_nbsearch {
    int  ePBC;
    matrix box;

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

    int  *sort_work;
    int  sort_work_nalloc;
} gmx_nbsearch_t_t;


void gmx_nbsearch_init(gmx_nbsearch_t * nbs,int natoms_subcell)
{
    snew(*nbs,1);
    (*nbs)->naps = natoms_subcell;
    (*nbs)->napc = natoms_subcell*NSUBCELL;

    (*nbs)->cxy_na      = NULL;
    (*nbs)->cxy_ind     = NULL;
    (*nbs)->cxy_nalloc  = 0;
    (*nbs)->cell        = NULL;
    (*nbs)->cell_nalloc = 0;
    (*nbs)->a           = NULL;
    (*nbs)->bb          = NULL;
    (*nbs)->nc_nalloc   = 0;
    (*nbs)->sort_work   = NULL;
    (*nbs)->sort_work_nalloc = 0;
}

static int set_grid_size_xy(gmx_nbsearch_t nbs,int n,matrix box)
{
    real adens,tlen,tlen_x,tlen_y,nc_max;

    if (n > nbs->napc)
    {
        adens = n/(box[XX][XX]*box[YY][YY]*box[ZZ][ZZ]);
        /* target cell length */
#if 0
        /* Approximately cubic super cells */
        tlen   = pow(nbs->napc/adens,1.0/3.0);
        tlen_x = tlen;
        tlen_y = tlen;
#else
        /* Approximately cubic sub cells */
        tlen   = pow(nbs->naps/adens,1.0/3.0);
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

static void sort_column(int dim,int *a,int n,rvec *x,real invh,
                        int nsort,int *sort)
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
        zi = (int)(x[a[i]][dim]*invh);

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
    for(zi=0; zi<nsort; zi++)
    {
        if (sort[zi] >= 0)
        {
            a[c++] = sort[zi];
        }
    }
    if (c < n)
    {
        gmx_incons("Lost particles while sorting");
    }
}

static void calc_bounding_box(int na,int stride,real *x,real *bb)
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

static void print_bbsizes(FILE *fp,gmx_nbsearch_t nbs,matrix box)
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
            box[XX][XX]/(nbs->ncx*NSUBCELL_X),
            box[YY][YY]/(nbs->ncy*NSUBCELL_Y),
            box[ZZ][ZZ]*nbs->ncx*nbs->ncy/(nbs->nc*NSUBCELL_Z),
            ba[XX],ba[YY],ba[ZZ],
            ba[XX]*nbs->ncx*NSUBCELL_X/box[XX][XX],
            ba[YY]*nbs->ncy*NSUBCELL_Y/box[YY][YY],
            ba[ZZ]*nbs->nc*NSUBCELL_Z/(nbs->ncx*nbs->ncy*box[ZZ][ZZ]));
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
                                   rvec *x,int stride,real *xnb,
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

    switch (stride)
    {
    case 3:
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
    case 4:
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

static void calc_cell_indices(gmx_nbsearch_t nbs,
                              matrix box,int n,rvec *x,
                              gmx_nb_atomdata_t *nbat)
{
    int  i;
    int  cx,cy,cz,c=-1,ncz_max,ncz;
    int  na,ash,na_c;
    int  subdiv_z,sub_z,na_z,ash_z;
    int  subdiv_y,sub_y,na_y,ash_y;
    int  subdiv_x,sub_x,na_x,ash_x;
    real *xnb;

    for(i=0; i<nbs->ncx*nbs->ncy; i++)
    {
        nbs->cxy_na[i] = 0;
    }

    for(i=0; i<n; i++)
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
        nbs->cxy_na[nbs->cell[i]]++;
    }

    /* Make the cell index as a function of x and y */
    ncz_max = 0;
    nbs->cxy_ind[0] = 0;
    for(i=0; i<nbs->ncx*nbs->ncy; i++)
    {
        ncz = (nbs->cxy_na[i] + nbs->napc - 1)/nbs->napc;
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
    if (ncz_max*nbs->napc*SGSF > nbs->sort_work_nalloc)
    {
        nbs->sort_work_nalloc = over_alloc_large(ncz_max*nbs->napc*SGSF);
        srenew(nbs->sort_work,nbs->sort_work_nalloc);
    }

    /* Now we know the dimensions we can fill the grid */
    for(i=0; i<n; i++)
    {
        nbs->a[nbs->cxy_ind[nbs->cell[i]]*nbs->napc+nbs->cxy_na[nbs->cell[i]]++] = i;
    }

    subdiv_x = nbs->naps;
    subdiv_y = NSUBCELL_X*subdiv_x;
    subdiv_z = NSUBCELL_Y*subdiv_y;

    /* Sort the atoms within each x,y column in 3 dimensions */
    nbs->nsubc_tot = 0;
    for(i=0; i<nbs->ncx*nbs->ncy; i++)
    {
        cx = i/nbs->ncy;
        cy = i - cx*nbs->ncy;

        na  = nbs->cxy_na[i];
        ncz = nbs->cxy_ind[i+1] - nbs->cxy_ind[i];
        ash = nbs->cxy_ind[i]*nbs->napc;

        /* Sort the atoms within each x,y column on z coordinate */
        sort_column(ZZ,nbs->a+ash,na,x,
                    ncz*nbs->napc*SORT_GRID_OVERSIZE/box[ZZ][ZZ],
                    ncz*nbs->napc*SGSF,nbs->sort_work);

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

                nbs->nsubc_tot += nbs->nsubc[c];

                /* Store the z-boundaries of the super cell */
                nbs->bbcz[c*NNBSBB_D  ] = x[nbs->a[ash_z]][ZZ];
                nbs->bbcz[c*NNBSBB_D+1] = x[nbs->a[ash_z+na_c-1]][ZZ];
            }

#if NSUBCELL_Y > 1
            sort_column(YY,nbs->a+ash_z,na_z,x,
                        nbs->inv_sy,subdiv_z*SGSF,nbs->sort_work);
#endif

            for(sub_y=0; sub_y<NSUBCELL_Y; sub_y++)
            {
                ash_y = ash_z + sub_y*subdiv_y;
                na_y  = min(subdiv_y,na-(ash_y-ash));

#if NSUBCELL_X > 1
                sort_column(XX,nbs->a+ash_y,na_y,x,
                            nbs->inv_sy,subdiv_y*SGSF,nbs->sort_work);
#endif

                for(sub_x=0; sub_x<NSUBCELL_X; sub_x++)
                {
                    ash_x = ash_y + sub_x*subdiv_x;
                    na_x  = min(subdiv_x,na-(ash_x-ash));

                    xnb   = nbat->x + ash_x*nbat->xstride;

                    if (na_x > 0)
                    {
                        copy_rvec_to_nbat_real(nbs->a+ash_x,na_x,nbs->naps,x,
                                               nbat->xstride,xnb,
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

    if (debug)
    {
        fprintf(debug,"ns non-zero sub-cells: %d average atoms %.2f\n",
                nbs->nsubc_tot,n/(double)nbs->nsubc_tot);
        print_bbsizes(debug,nbs,box);
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
    if (nbat->xstride == 3)
    {
        nb_realloc_real(&nbat->q,n,nbat->alloc,nbat->free);
    }
    nb_realloc_real(&nbat->x,n*nbat->xstride,nbat->alloc,nbat->free);
    nb_realloc_real(&nbat->f,n*nbat->xstride,nbat->alloc,nbat->free);
    nbat->nalloc = n;
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

    calc_cell_indices(nbs,box,n,x,nbat);
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

static real box_dist2_cond(real bx0,real bx1,real by0,real by1,real bz0,real bz1,
                           const real *bb)
{
    real d2;

    d2 = 0;
    if (bb[1] < bx0)
    {
        d2 += sqr(bb[1] - bx0);
    }
    else if (bb[0] > bx1)
    {
        d2 += sqr(bb[0] - bx1);
    }
    if (bb[3] < by0)
    {
        d2 += sqr(bb[3] - by0);
    }
    else if (bb[2] > by1)
    {
        d2 += sqr(bb[2] - by1);
    }
    if (bb[5] < bz0)
    {
        d2 += sqr(bb[5] - bz0);
    }
    else if (bb[4] > bz1)
    {
        d2 += sqr(bb[4] - bz1);
    }

    return d2;
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
}

static void print_nblist_statistics(FILE *fp,const gmx_nblist_t *nbl,
                                    const gmx_nbsearch_t nbs,real rl)
{
    int *count;
    int lmax,i,j,cjp,cj,l;
    double sl;

    fprintf(fp,"nbl naps %d rl %g ncp %d per cell %.1f atoms %.1f ratio %.2f\n",
            nbl->naps,rl,nbl->nsi,nbl->nsi/(double)nbs->nsubc_tot,
            nbl->nsi/(double)nbs->nsubc_tot*nbs->naps,
            nbl->nsi/(double)nbs->nsubc_tot*nbs->naps/(0.5*4.0/3.0*M_PI*rl*rl*rl*nbs->nsubc_tot*nbs->naps/det(nbs->box)));

    fprintf(fp,"nbl average i sub cell list length %.1f\n",
            nbl->nsi/(double)nbl->nsj);

    /*
    snew(count,nbs->nc);
    lmax = 0;
    for(i=0; i<nbl->nci; i++) {
        l = 0;
        cjp = -1;
        for(j=nbl->ci[i].sj_ind_start; j<nbl->ci[i].sj_ind_end; j++)
        {
            cj = nbl->cj[j];
            if (cj != cjp + 1)
            {
                count[l]++;
                lmax = max(lmax,l);
                l = 0;
            }
            l++;
            cjp = cj;
        }
        count[l]++;
    }
    fprintf(fp,"Series");
    sl = 0;
    for(l=1; l<=lmax; l++)
    {
        fprintf(fp," %d %.1f%%",l,100*count[l]/(double)nbl->ncj);
        sl += l*l*count[l];
    }
    fprintf(fp,"\n");
    fprintf(fp,"Weighted average series size: %.1f\n",sl/nbl->ncj);
    sfree(count);
    */
}

static void make_subcell_list(gmx_nblist_t *nbl,
                              int ci,int cj,const int *nsubc,
                              real shx,real shy,real shz,
                              const real *bb,
                              real rl2)
{
    int npair;
    int sj,si1,si,csj,csi;

    for(sj=0; sj<nsubc[cj]; sj++)
    {
        csj = cj*NSUBCELL + sj;

        if (!nbl->TwoWay && ci == cj)
        {
            si1 = sj + 1;
        }
        else
        {
            si1 = nsubc[ci];
        }

        npair = 0;
        for(si=0; si<si1; si++)
        {
            csi = ci*NSUBCELL + si;

            if (box_dist2(bb[csi*NNBSBB  ] + shx,
                          bb[csi*NNBSBB+1] + shx,
                          bb[csi*NNBSBB+2] + shy,
                          bb[csi*NNBSBB+3] + shy,
                          bb[csi*NNBSBB+4] + shz,
                          bb[csi*NNBSBB+5] + shz,
                          bb+csj*NNBSBB) < rl2)
            {
                nbl->si[nbl->nsi++] = ci*NSUBCELL + si;
                npair++;
            }
        }

        if (npair > 0)
        {
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

static void new_ci_entry(gmx_nblist_t *nbl,int ci,int shift)
{
    if (nbl->nci + 1 > nbl->ci_nalloc)
    {
        nbl->ci_nalloc = over_alloc_small(nbl->nci+1);
        nb_realloc_void((void **)&nbl->ci,
                        nbl->nci*sizeof(*nbl->ci),
                        nbl->ci_nalloc*sizeof(*nbl->ci),
                        nbl->alloc,nbl->free);
    }
    nbl->ci[nbl->nci].ci           = ci;
    nbl->ci[nbl->nci].shift        = shift;
    nbl->ci[nbl->nci].sj_ind_start = (nbl->nci == 0 ? 0 :
                                          nbl->ci[nbl->nci-1].sj_ind_end);
    nbl->ci[nbl->nci].sj_ind_end   = nbl->ci[nbl->nci].sj_ind_start;
}

static void close_ci_entry(gmx_nblist_t *nbl)
{
    /* All content of the new ci entry have already been filled correctly,
     * we only need to increase the count here (for non empty lists).
     */
    if (nbl->ci[nbl->nci].sj_ind_end > nbl->ci[nbl->nci].sj_ind_start)
    {
        nbl->nci++;
    }
}

static void clear_nblist(gmx_nblist_t *nbl)
{
    nbl->nci = 0;
    nbl->nsj = 0;
    nbl->sj[0].si_ind = 0;
    nbl->nsi = 0;
}

void gmx_nbsearch_make_nblist(const gmx_nbsearch_t nbs,real rcut,real rlist,
                              gmx_nblist_t *nbl)
{
    gmx_bool bDomDec;
    matrix box;
    real rl2;
    int  d,ci,ci_xy,ci_x,ci_y,cj;
    ivec shp;
    int  tx,ty,tz;
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

    clear_nblist(nbl);

    rl2 = nbl->rlist*nbl->rlist;

    copy_mat(nbs->box,box);

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
    for(ci=0; ci<nbs->nc; ci++)
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
                    shx = tx*box[XX][XX] + ty*box[YY][XX] + tz*box[ZZ][XX];

                    bx0 = (ci_x  )*nbs->sx + shx;
                    bx1 = (ci_x+1)*nbs->sx + shx;

                    get_cell_range(bx0,bx1,nbs->ncx,nbs->sx,nbs->inv_sx,d2z,rl2,
                                   &cxf,&cxl); 

                    new_ci_entry(nbl,ci,XYZ2IS(tx,ty,tz));

                    if (!nbl->TwoWay && cxf < ci_x)
                    {
                        /* Leave the pairs with i > j.
                         * x is the major index, so skip half of it.
                         */
                        cxf = ci_x;
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

                        if (!nbl->TwoWay && cx == 0 && cyf < ci_y)
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
                                cf = max(cf,ci);

                                if (cf <= cl)
                                {
                                    check_subcell_list_space(nbl,cl-cf+1);

                                    for(cj=cf; cj<=cl; cj++)
                                    {
                                        make_subcell_list(nbl,ci,cj,nbs->nsubc,
                                                          shx,shy,shz,
                                                          bb,rl2);
                                    }
                                }
                            }
                        }  
                    }
        
                    close_ci_entry(nbl);
                }
            }
        }
    }
    
    if (debug)
    {
        print_nblist_statistics(debug,nbl,nbs,rlist);
    }
}

void gmx_nb_atomdata_init(gmx_nb_atomdata_t *nbat,int xstride,
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
    snew(nbat->nbfp,nbat->ntype*nbat->ntype*2);
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
    nbat->q       = NULL;
    nbat->xstride = xstride;
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

        if (nbat->xstride == 4)
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
    }
}

void gmx_nb_atomdata_add_nbat_f_to_f(const gmx_nbsearch_t nbs,
                                     const gmx_nb_atomdata_t *nbat,
                                     rvec *f)
{
    int  cxy,na,ash,j;
    const int  *a;
    const real *fnb;

    /* Loop over all columns and copy and fill */
    for(cxy=0; cxy<nbs->ncx*nbs->ncy; cxy++)
    {
        na  = nbs->cxy_na[cxy];
        ash = nbs->cxy_ind[cxy]*nbs->napc;

        a   = nbs->a + ash;
        fnb = nbat->f + ash*nbat->xstride;

        switch (nbat->xstride)
        {
        case 3:
            for(j=0; j<na; j++)
            {
                f[a[j]][XX] += fnb[j*3];
                f[a[j]][YY] += fnb[j*3+1];
                f[a[j]][ZZ] += fnb[j*3+2];
            }
            break;
        case 4:
            for(j=0; j<na; j++)
            {
                f[a[j]][XX] += fnb[j*4];
                f[a[j]][YY] += fnb[j*4+1];
                f[a[j]][ZZ] += fnb[j*4+2];
            }
            break;
        default:
            gmx_incons("Unsupported stride");
        }
    }
}
