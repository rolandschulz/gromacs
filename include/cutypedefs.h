#ifndef CUTYPEDEFS_H
#define CUTYPEDEFS_H

#include "types/nblist_box.h"
#include "cutypedefs_ext.h"


#ifdef __cplusplus
extern "C" {
#endif


struct cudata 
{
    int     natoms;     /* number of atoms for all 8 neighbouring domains 
                           !!! ATM only one value, with MPI it'll be 8                      */
    int     nalloc;     /* allocation size for the atom data (xq, f), 
                           when needed it's reallocated to natoms * 20% + 100 buffer zone   */ 
    
    float4  *f;         /* forces, size natoms                      */
    float4  *xq;        /* atom coordinates + charges, size natoms  */

    int     ntypes;     /* number of atom types             */
    int     *atom_types;/* atom type indices, size natoms   */
    
    /* nonbonded paramters 
       TODO -> constant so some of them should be moved to constant memory */
    float   eps_r; 
    float   eps_rf;
    float   ewald_beta;
    float   cutoff_sq;
    float   *nbfp;      /* nonbonded parameters C12, C6 */    

    /* tabulated erfc */
    int     erfc_tab_size;
    float   erfc_tab_scale;
    float   *erfc_tab;

    /* async execution stuff */
    gmx_bool        streamGPU;                  /* are we streaming of not (debugging)              */
    cudaStream_t    nb_stream;                  /* XXX nonbonded calculation stream - not in use    */
    cudaEvent_t     start_nb, stop_nb;          /* events for timing nonbonded calculation + related 
                                                   data transfers                                   */
    cudaEvent_t     start_atdat, stop_atdat;    /* event for timing atom data (every NS step)       */

    /* neighbor list data */
    int             naps;       /* number of atoms per subcell                  */
    
    int             nci;        /* size of ci, # of i cells in the list         */
    int             ci_nalloc;  /* allocation size for ci                       */
    gmx_nbl_ci_t     *ci;       /* list of i-cells ("supercells")               */

    int             nsj_1;      /* total # of i-j cell subcell pairs 
                                   +1 element for closing the list              */
    int             sj_nalloc;  /* allocation size for sj                       */
    gmx_nbl_sj_t    *sj;        /* j subcell list, contains j subcell number and 
                                   index into the i subcell list                */

    int             nsi;        /* The total number of i subcells               */
    gmx_nbl_si_t    *si;        /* Array of i sub-cells (in pairs with j)       */
    int             si_nalloc;  /* Allocation size of ii                        */

    float3          *shift_vec;  /* shifts */
};

#ifdef __cplusplus
}
#endif


#endif	/* _CUTYPEDEFS_H_ */
