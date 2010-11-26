#ifndef CUTYPEDEFS_H
#define CUTYPEDEFS_H

#include "types/nblist_box.h"
#include "cutypedefs_ext.h"

#define GPU_CELL_PAIR_GROUP 1

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
    
    unsigned long * excl;
    
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
    cudaEvent_t     start_atomdata, stop_atomdata;

#if 1 // WC malloc stuff
    cudaEvent_t     start_x_trans, stop_x_trans;
    float   x_trans_time;
    float4  *h_xq;
#endif 

    /* neighbor list data */
    int             naps;   /* number of atoms per cell */

    int             nci;  /* size of ci */
    int             ci_nalloc; /* allocation size for ci */
    gmx_nbs_ci_t     *ci; /* list of i-cells ("supercells") */

    int             nsj_1;        /* # of i-j cell subcell pairs +1 for closing the list */
    int             sj_nalloc;  /* allocation size for sj */
    gmx_nbs_sj_t    *sj;        /* j subcell list, contain j subcell number and index into the i subcell list */

    int      nsi;          /* The total number of i sub cells          */
    int      *si;          /* Array of i sub-cells (in pairs with j)   */
    int      si_nalloc;    /* Allocation size of ii                    */

    float3          *shiftvec;

    int             cell_pair_group;
};

#ifdef __cplusplus
}
#endif


#endif	/* _CUTYPEDEFS_H_ */
