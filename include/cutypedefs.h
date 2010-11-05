#ifndef CUTYPEDEFS_H
#define CUTYPEDEFS_H

#include "types/nblist_box.h"
#include "cutypedefs_ext.h"

#ifdef __cplusplus
extern "C" {
#endif


struct cudata 
{
    int         natoms; /* number of atoms for all 8 neighbouring domains 
                           !!! ATM only one value, with MPI it'll be 8 */
    int         nalloc; /* allocation size for the atom data (xq, f), 
                           when needed it's reallocated to natoms * 20% + 100 buffer zone */ 
    
    float4  *f;  /* forces, size natoms */
    float4  *xq;  /* atom coordinates + charges, size natoms */

    int     ntypes; /* number of atom types */
    int     *atom_types; /* atom type indices, size natoms */
    
    /* ???? * exclusions;  not used */
    
    /* nonbonded paramters 
       TODO -> constant so some of them should be moved to constant memory */
    float   eps_r; 
    float   eps_rf;
    float   ewald_beta;
    float   cutoff;
    float   *nbfp; /* nonbonded parameters C12, C6 */    

    /* async execution stuff*/
    gmx_bool        streamGPU;
    cudaStream_t    nb_stream;
    cudaEvent_t     start_nb, stop_nb;

    /* neighbor list data */
    int             napc;   /* number of atoms per cell */
    int             nlist;  /* size of nblist */
    int             nblist_nalloc; /* allocation size for nblist */
    gmx_nbs_jlist_t *nblist; /* list of cell interactions, corresponds to the gmx_nbs_jlist_t*/
    int             ncj; /* # of i-j cell pairs */
    int             cj_nalloc; /* allocation size for cj */
    int             *cj; /* j cells */
};

#ifdef __cplusplus
}
#endif


#endif	/* _CUTYPEDEFS_H_ */
