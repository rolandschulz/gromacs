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

/* Abstract type for neighbor searching data */
typedef struct gmx_nbsearch * gmx_nbsearch_t;

/* Smaller neighbor list list unit */
typedef struct {
    int ci;         /* i-cell              */
    int shift;      /* Shift vector index  */
    int jind_start; /* Start index into cj */
    int jind_end;   /* End index into cj   */
} gmx_nbs_jlist_t;

typedef struct {
    int      napc;         /* Number of atoms per cell             */
    gmx_bool TwoWay;       /* Each pair once or twice in the list? */
    int      nlist;        /* The number of lists                  */
    gmx_nbs_jlist_t *list; /* The lists                            */
    int      list_nalloc;  /* Allocation size of list              */
    int      ncj;          /* The total number of i-j-cell pairs   */
    int      *cj;          /* Array of j-cells                     */
    int      cj_nalloc;    /* Allocation size of cj                */
} gmx_nblist_t;

typedef struct {
    int  natoms;  /* Number of atoms                                    */
    int  *type;   /* Atom types                                         */
    real *q;      /* Charges, could be NULL if incorporated in x        */
    int  xstride; /* stride for a coordinate in x (usually 3 or 4)      */
    real *x;      /* x and possibly q, size natoms*xstride              */
    int  nalloc;  /* Allocation size of all arrays (time xstride for x) */
} gmx_nb_atomdata_t;

#ifdef __cplusplus
}
#endif

#endif
