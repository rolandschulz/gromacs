#include <stdlib.h>
#include <stdio.h>

#include "gmx_fatal.h"
#include "smalloc.h"

#include "cutypedefs.h"
#include "cudautils.h"
#include "gpu_data.h"

/*** CUDA MD Data operations ***/

void init_cudata_ff(FILE *fplog, 
                    t_cudata *dp_data,
                    const t_forcerec *fr)
{
    t_cudata    d_data = NULL; 
    cudaError_t stat;
    int         ntypes = fr->ntype;;

    if (dp_data == NULL) return;
    
    snew(d_data, 1);

    d_data->ntypes  = ntypes;
    d_data->nalloc  = 0;
    
    d_data->eps_r = fr->epsilon_r;
    d_data->eps_rf = fr->epsilon_rf;   

    stat = cudaMalloc((void **)&d_data->nbfp, 2*ntypes*sizeof(*(d_data->nbfp)));
    CU_RET_ERR(stat, "cudaMalloc failed on d_data->nbfp"); 
    upload_cudata(d_data->nbfp, fr->nbfp, 2*ntypes*sizeof(*(d_data->nbfp)));

    if (fplog != NULL)
    {
        fprintf(fplog, "Initialized CUDA data structures.\n");
        
        printf("Initialized CUDA data structures.\n");
        fflush(stdout);
    }

    *dp_data = d_data;
}

/* natoms gets the value of fr->natoms_force */
void init_cudata_atoms(t_cudata d_data,
                        const t_mdatoms *mdatoms,
                        int natoms)
{
    cudaError_t stat;

    /* need to reallocate if we have to copy more atoms than the amount */
    /* of space available */
    if (natoms > d_data->nalloc)
    {
        d_data->nalloc = natoms * 1.2 + 100;
    }
    d_data->natoms = natoms;
    
    /* TODO reallocate x and f if the size is not enough ! */
    stat = cudaMalloc((void **)&d_data->f, natoms*sizeof(*(d_data->f)));
    CU_RET_ERR(stat, "cudaMalloc failed on d_data->f");
    stat = cudaMalloc((void **)&d_data->x, natoms*sizeof(*(d_data->x)));
    CU_RET_ERR(stat, "cudaMalloc failed on d_data->x");

    stat = cudaMalloc((void **)&d_data->atom_types, natoms*sizeof(*(d_data->atom_types)));
    CU_RET_ERR(stat, "cudaMalloc failed on d_data->atom_types"); 
    upload_cudata(d_data->atom_types, mdatoms->typeA, natoms*sizeof(*(d_data->atom_types)));

    stat = cudaMalloc((void **)&d_data->charges, natoms*sizeof(*(d_data->charges)));
    CU_RET_ERR(stat, "cudaMalloc failed on d_data->charges");
    upload_cudata(d_data->charges, mdatoms->chargeA, natoms*sizeof(*(d_data->charges)));

}

void destroy_cudata(FILE *fplog, t_cudata d_data)
{
    cudaError_t stat;

    if (d_data == NULL) return;

    stat = cudaFree(d_data->f);
    CU_RET_ERR(stat, "cudaFree failed on d_data->f");
    stat = cudaFree(d_data->x);   
    CU_RET_ERR(stat, "cudaFree failed on d_data->x");
    stat = cudaFree(d_data->atom_types);   
    CU_RET_ERR(stat, "cudaFree failed on d_data->atom_types");
    stat = cudaFree(d_data->charges);
    CU_RET_ERR(stat, "cudaFree failed on d_data->charges");
    stat = cudaFree(d_data->nbfp);
    CU_RET_ERR(stat, "cudaFree failed on d_data->nbfp");

    fprintf(fplog, "Cleaned up CUDA data structures.\n");

}


int cu_upload_X(t_cudata d_data, rvec h_x[])
{
    return upload_cudata(d_data->x, h_x, d_data->natoms*sizeof(*d_data->x));
}

int cu_download_F(rvec h_f[], t_cudata d_data)
{
    return download_cudata(h_f, d_data->f, d_data->natoms*sizeof(*h_f));
}
