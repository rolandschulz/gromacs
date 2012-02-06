/* -*- mode: c; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4; c-file-style: "stroustrup"; -*-
 *
 *
 *                This source code is part of
 *
 *                 G   R   O   M   A   C   S
 *
 *          GROningen MAchine for Chemical Simulations
 *
 * Written by David van der Spoel, Erik Lindahl, Berk Hess, and others.
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2012, The GROMACS development team,
 * check out http://www.gromacs.org for more information.
 *
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
 *
 * And Hey:
 * Gallium Rubidium Oxygen Manganese Argon Carbon Silicon
 */

#ifndef GMX_OMP_NTHREADS
#define GMX_OMP_NTHREADS

/*! Initializes the per-module thread count. It is compatible with tMPI, 
 *  thread-safety is ensured (for the features available with tMPI). 
 *  This function should only be caled once during the execution. */
void init_module_nthreads(t_commrec *cr);

/*! Returns the default number of threads. */
int gmx_omp_get_default_nthreads();

/*! Returns the number of threads for domain decomposition. */
int gmx_omp_get_domdec_nthreads();

/*! Returns the number of threads for pair search. */
int gmx_omp_get_pairsearch_nthreads();

/*! Returns the number of threads for non-bonded force calculations. */
int gmx_omp_get_nonbonded_nthreads();

/*! Returns the number of threads for bonded force calculations. */
int gmx_omp_get_bonded_nthreads();

/*! Returns the number of threads for PME. */
int gmx_omp_get_pme_nthreads();

/*! Returns the number of threads for update. */
int gmx_omp_get_update_nthreads();

/*! Returns the number of threads for LINCS. */
int gmx_omp_get_lincs_nthreads();

/*! Returns the number of threads for SETTLE. */
int gmx_omp_get_settle_nthreads();

#endif /* GMX_OMP_NTHREADS */