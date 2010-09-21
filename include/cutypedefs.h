#ifndef CUTYPEDEFS_H
#define CUTYPEDEFS_H

#include "cutypedefs_ext.h"

struct cudata 
{
    int         natoms; /* number of atoms for all 8 neighbouring domains 
                           !!! ATM only one number as we're not running in parallel */
    int         nalloc; /* amount of space allocated for the atom data, contains 
                           natoms * 20% + 100 buffer zone to avod frequent reallocation */
    float3 * f;  /* forces, size natoms */
    float3 * x;  /* coordinates, size ntypes */

    int     ntypes; /* number of atom types */
    int *   atom_types; /* atom type indices, size natoms */
    
    float * charges; /* size natoms */
    /* float * exclusions;  not used ATM */
    
    /* nonbonded paramters 
       TODO -> constant so some of them should be moved to constant memory */
    float   eps_r; 
    float   eps_rf;
    float   ewald_beta;
    float   cutoff;
    float * nbfp; /* nonbonded parameters C12, C6 */    

    /* NB events*/
    cudaEvent_t start_nb, stop_nb;
};

#endif
