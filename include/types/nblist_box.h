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

#ifndef _nblist_box_h
#define _nblist_box_h

#ifdef __cplusplus
extern "C" {
#endif

#define NSUBCELL_Z 2
#define NSUBCELL_Y 2
#define NSUBCELL_X 2
#define NSUBCELL   (NSUBCELL_Z*NSUBCELL_Y*NSUBCELL_X)

/* Abstract type for neighbor searching data */
typedef struct gmx_nbsearch * gmx_nbsearch_t;

/* Function that should return a pointer *ptr to memory
 * of size nbytes.
 * Error handling should be done within this function.
 */
typedef void gmx_nbat_alloc_t(void **ptr,size_t nbytes);

/* Function that should free the memory pointed to by *ptr.
 * NULL should not be passed to this function.
 */
typedef void gmx_nbat_free_t(void *ptr);

/* Smaller neighbor list list unit */
typedef struct {
    int ci;            /* i-cell              */
    int shift;         /* Shift vector index  */
    int sj_ind_start;  /* Start index into sj */
    int sj_ind_end;    /* End index into sj   */
} gmx_nbl_ci_t;

typedef struct {
    int sj;            /* The j innerloop sub cell                   */
    int si_ind;        /* Index into i innerloop sub cells,          *
                        * the index end is ii_ind of the next j cell */
} gmx_nbl_sj_t;

typedef struct {
    int si;            /* The j innerloop sub cell                   */
    unsigned long excl;/* Exclusion bits                             */
} gmx_nbl_si_t;

typedef struct {
    gmx_nbat_alloc_t *alloc;
    gmx_nbat_free_t  *free;
    int      napc;         /* The number of atoms per super cell       */
    int      naps;         /* The number of atoms per sub cell         */
    gmx_bool TwoWay;       /* Each pair once or twice in the list?     */
    real     rcut;         /* The cut-off distance                     */
    real     rlist;        /* The radius for constructing the list     */
    int      nci;          /* The number of i super cells in the list  */
    gmx_nbl_ci_t *ci;      /* The i super cell list                    */
    int      ci_nalloc;    /* The allocation size of ci                */
    int      nsj;          /* The total number of j sub cell           */
    gmx_nbl_sj_t *sj;      /* The j super cell list (size nsj+1)       */
    int      sj_nalloc;    /* The allocation isze of sj                */
    int      nsi;          /* The total number of i sub cells          */
    gmx_nbl_si_t *si;      /* Array of i sub-cells (in pairs with j)   */
    int      si_nalloc;    /* Allocation size of ii                    */
} gmx_nblist_t;

enum { nbatXYZ, nbatXYZQ };

typedef struct {
    gmx_nbat_alloc_t *alloc;
    gmx_nbat_free_t  *free;
    int  ntype;   /* The number of different atom types                 */
    real *nbfp;   /* The Lennard-Jones C6 and C12 params, size ntype^2  */
    int  natoms;  /* Number of atoms                                    */
    int  *type;   /* Atom types                                         */
    int  XFormat; /* The format of x (and q), enum                      */
    real *q;      /* Charges, could be NULL if incorporated in x        */
    int  xstride; /* stride for a coordinate in x (usually 3 or 4)      */
    real *x;      /* x and possibly q, size natoms*xstride              */
    real *f;      /* f, size natoms*xstride                             */
    int  nalloc;  /* Allocation size of all arrays (time xstride for x) */
} gmx_nb_atomdata_t;

#ifdef __cplusplus
}
#endif

#endif
