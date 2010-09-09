#ifndef CUTYPEDEFS_H
#define CUTYPEDEFS_H

#include "cutypedefs_ext.h"

struct cudata 
{
    int         natoms;
    int         nalloc;
    float3 * f;  /* forces, size natoms */
    float3 * x;  /* coordinates, size ntypes */

    int     ntypes;
    int *   atom_types; /* atom type indices, size natoms */
    
    float * charges; /* size natoms */
    float * exclusions;

    float   eps_r;
    float   eps_rf;
    float   ewald_beta;
    float   cutoff;
    float * nbfp; /* nonbonded parameters C12, C6 */    
};

#endif
