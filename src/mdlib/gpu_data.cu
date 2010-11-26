#include <stdlib.h>
#include <stdio.h>

#include "gmx_fatal.h"
#include "smalloc.h"

#include "cutypedefs.h"
#include "cudautils.h"
#include "gpu_data.h"
#include "cupmalloc.h"

#define USE_CUDA_EVENT_BLOCKING_SYNC FALSE /* makes the CPU thread busy-wait! */

/*** CUDA MD Data operations ***/

/* forward declaration*/
void destroy_cudata_atoms(t_cudata /*d_data*/);
void destroy_cudata_ci(t_cudata /*d_data*/);
void destroy_cudata_sj(t_cudata /*d_data*/);
void destroy_cudata_si(t_cudata /*d_data*/);

void init_cudata_ff(FILE *fplog, 
                    t_cudata *dp_data,
                    const t_forcerec *fr)
{
    t_cudata            d_data = NULL;    
    cudaError_t         stat;
    gmx_nb_atomdata_t   *nbat = fr->nbat;
    int                 ntypes = nbat->ntype;    
    char                *env_var;
    int                 itmp;

#if 0 /* texture business */
    cudaChannelFormatDesc   cd;
    const textureReference  *texnbfp;
#endif

    int eventflags = ( USE_CUDA_EVENT_BLOCKING_SYNC ? cudaEventBlockingSync: cudaEventDefault );

    if (dp_data == NULL) return;
    
    snew(d_data, 1);

    d_data->ntypes  = ntypes;
    d_data->nalloc  = 0;
    
    d_data->eps_r = fr->epsilon_r;
    d_data->eps_rf = fr->epsilon_rf;   

    /* events for NB async ops */
    d_data->streamGPU = fr->streamGPU;
    if (d_data->streamGPU)
    {
        stat = cudaEventCreateWithFlags(&(d_data->start_nb), eventflags);
        CU_RET_ERR(stat, "cudaEventCreate on start_nb failed");
        stat = cudaEventCreateWithFlags(&(d_data->stop_nb), eventflags);
        CU_RET_ERR(stat, "cudaEventCreate on stop_nb failed");
        stat = cudaEventCreateWithFlags(&(d_data->start_atomdata), eventflags);
        CU_RET_ERR(stat, "cudaEventCreate on start_atomdata failed");
        stat = cudaEventCreateWithFlags(&(d_data->stop_atomdata), eventflags);
        CU_RET_ERR(stat, "cudaEventCreate on stop_atomdata failed");       
#if 0 // WC malloc stuff
        stat = cudaEventCreateWithFlags(&(d_data->start_x_trans), eventflags);
        stat = cudaEventCreateWithFlags(&(d_data->stop_x_trans), eventflags);
#endif
    }   

    stat = cudaStreamCreate(&d_data->nb_stream);
    CU_RET_ERR(stat, "cudaSteamCreate on nb_stream failed");

    /* NB params */
    stat = cudaMalloc((void **)&d_data->nbfp, 2*ntypes*ntypes*sizeof(*(d_data->nbfp)));
    CU_RET_ERR(stat, "cudaMalloc failed on d_data->nbfp"); 
    upload_cudata(d_data->nbfp, nbat->nbfp, 2*ntypes*ntypes*sizeof(*(d_data->nbfp)));
#if 0 /* texture business */
    stat = cudaGetTextureReference(&texnbfp, "texnbfp");
    CU_RET_ERR(stat, "cudaGetTextureReference on texnbfp failed");
    cd = cudaCreateChannelDesc<float>();
    stat = cudaBindTexture(NULL, texnbfp, d_data->nbfp, &cd, 2*d_data->ntypes*d_data->ntypes);
    CU_RET_ERR(stat, "cudaBindTexture on texnbfp failed");
#endif
    stat = cudaMalloc((void**)&d_data->shiftvec, SHIFTS*sizeof(*d_data->shiftvec));
    CU_RET_ERR(stat, "cudaMalloc failed on d_data->shiftvec");

    if (fplog != NULL)
    {
        fprintf(fplog, "Initialized CUDA data structures.\n");
        
        printf("Initialized CUDA data structures.\n");
        fflush(stdout);
    }

    /* initilize to NULL all data structures that might need reallocation 
       in init_cudata_atoms */
    d_data->xq      = NULL;
    d_data->f       = NULL;
    d_data->ci      = NULL;
    d_data->sj      = NULL;
    d_data->si      = NULL;

    d_data->h_xq    = NULL;

    /* size -1 just means that it has not been initialized yet */
    d_data->natoms          = -1;
    d_data->nalloc          = -1;
    d_data->nci             = -1;
    d_data->ci_nalloc       = -1;
    d_data->nsj_1           = -1;
    d_data->sj_nalloc       = -1;
    d_data->nsi             = -1;
    d_data->si_nalloc       = -1;
    d_data->naps            = -1;

    if ((env_var = getenv("GMX_CELL_PAIR_GROUP")) != NULL)
    {
        sscanf(env_var, "%d", &itmp);
        if (itmp < 1)
        {
            gmx_fatal(FARGS, "Invalid GMX_CELL_PAIR_GROUP value (%d)!", itmp);
        }
        else
        {
            printf("CELL_PAIR_GROUP=%d\n", itmp);
            d_data->cell_pair_group = itmp;
        }
    }
    else 
    {
        d_data->cell_pair_group = GPU_CELL_PAIR_GROUP;
    }

    *dp_data = d_data;
}

/* TODO: move initilizations into a function! */
void init_cudata_atoms(t_cudata d_data, 
                       const gmx_nb_atomdata_t *atomdata, 
                       const gmx_nblist_t *nblist,
                       gmx_bool doStream)
{
    cudaError_t stat;
    int         nalloc, ci_nalloc, sj_nalloc, si_nalloc;
    int         natoms  = atomdata->natoms;
    int         nci     = nblist->nci;
    int         nsj_1   = nblist->nsj + 1;
    int         nsi     = nblist->nsi;
   
    /* asynch copy all data */
    cudaEventRecord(d_data->start_atomdata, 0);

    if (d_data->naps < 0)
    {
        d_data->naps = nblist->naps;
    }
    else
    {
        if (d_data->naps != nblist->naps)
        {
            gmx_fatal(FARGS, "Internal error: the #atoms per cell has changed (from %d to %d)",
                    d_data->naps, nblist->naps);
        }
    }

    /* need to reallocate if we have to copy more atoms than the amount of space
       available and only allocate if we haven't initilzed yet, i.e d_data->natoms == -1 */
    if (natoms > d_data->nalloc)
    {
        nalloc = natoms * 1.2 + 100;
    
        /* free up first if the arrays have already been initialized */
        if (d_data->nalloc != -1)
        {
            destroy_cudata_atoms(d_data);                
        }
        
        stat = cudaMalloc((void **)&d_data->f, nalloc*sizeof(*(d_data->f)));
        CU_RET_ERR(stat, "cudaMalloc failed on d_data->f");                   
        stat = cudaMalloc((void **)&d_data->xq, nalloc*sizeof(*(d_data->xq)));
        CU_RET_ERR(stat, "cudaMalloc failed on d_data->xq");            
        stat = cudaMalloc((void **)&d_data->atom_types, nalloc*sizeof(*(d_data->atom_types)));
        CU_RET_ERR(stat, "cudaMalloc failed on d_data->atom_types"); 

        d_data->nalloc = nalloc;
    }
    /* XXX for the moment we just set all 8 values to the same value... 
       ATM not, we'll do that later */    
    d_data->natoms = natoms;

    if (nci > d_data->ci_nalloc) 
    {
        ci_nalloc = nci * 1.2 + 100;

        /* free up first if the arrays have already been initialized */
        if (d_data->ci_nalloc != -1)
        {
            destroy_cudata_ci(d_data);                
        }

        stat = cudaMalloc((void **)&d_data->ci, ci_nalloc*sizeof(*(d_data->ci)));
        CU_RET_ERR(stat, "cudaMalloc failed on d_data->ci");           

        d_data->ci_nalloc = ci_nalloc;
    }
    d_data->nci = nci;

    if (nsj_1 > d_data->nsj_1) 
    {
        sj_nalloc = nsj_1 * 1.2 + 100;

        /* free up first if the arrays have already been initialized */
        if (d_data->sj_nalloc != -1)
        {
            destroy_cudata_sj(d_data);                
        }

        stat = cudaMalloc((void **)&d_data->sj, sj_nalloc*sizeof(*(d_data->sj)));
        CU_RET_ERR(stat, "cudaMalloc failed on d_data->sj");    

        d_data->sj_nalloc = sj_nalloc;
    }
    d_data->nsj_1 = nsj_1;

    if (nsi > d_data->nsi)
    {
        si_nalloc = nsi * 1.2 + 100;

        /* free up first if the arrays have already been initialized */
        if (d_data->si_nalloc != -1)
        {
            destroy_cudata_si(d_data);                
        }

        stat = cudaMalloc((void **)&d_data->si, si_nalloc*sizeof(*(d_data->si)));
        CU_RET_ERR(stat, "cudaMalloc failed on d_data->si");    

        d_data->si_nalloc = si_nalloc;
    }
    d_data->nsi = nsi;

#if 0 // WC malloc stuff
    if (d_data->h_xq) 
    {
        pfree(d_data->h_xq);
    }
    pmalloc_wc((void**)&d_data->h_xq, d_data->nalloc*sizeof(*d_data->h_xq));
    memcpy(d_data->h_xq, atomdata->x, d_data->nalloc*sizeof(*d_data->h_xq));
#endif

    if(doStream)
    {
        upload_cudata_async(d_data->atom_types, atomdata->type, natoms*sizeof(*(d_data->atom_types)), 0);
        upload_cudata_async(d_data->ci, nblist->ci, nci*sizeof(*(d_data->ci)), 0);
        upload_cudata_async(d_data->sj, nblist->sj, nsj_1*sizeof(*(d_data->sj)), 0);
        upload_cudata_async(d_data->si, nblist->si, nsi*sizeof(*(d_data->si)), 0);       
    }
    else 
    {
        upload_cudata(d_data->atom_types, atomdata->type, natoms*sizeof(*(d_data->atom_types)));
        upload_cudata(d_data->ci, nblist->ci, nci*sizeof(*(d_data->ci)));
        upload_cudata(d_data->sj, nblist->sj, nsj_1*sizeof(*(d_data->sj)));
        upload_cudata(d_data->si, nblist->si, nsi*sizeof(*(d_data->si)));    
    }
    cudaEventRecord(d_data->stop_atomdata, 0);
 
}

void cu_blockwait_atomdata(t_cudata d_data, float *time)
{    
    cudaError_t stat;     

    stat = cudaEventSynchronize(d_data->stop_atomdata);
    CU_RET_ERR(stat, "the async trasfer of atomdata has failed");   

    cudaEventElapsedTime(time, d_data->start_atomdata, d_data->stop_atomdata);
}

void destroy_cudata(FILE *fplog, t_cudata d_data)
{
    cudaError_t stat;
    const textureReference *texnbfp;

    if (d_data == NULL) return;

    if (d_data->streamGPU)
    {
        stat = cudaEventDestroy(d_data->start_nb);
        CU_RET_ERR(stat, "cudaEventDestroy failed on d_data->start_nb");
        stat = cudaEventDestroy(d_data->stop_nb);
        CU_RET_ERR(stat, "cudaEventDestroy failed on d_data->stop_nb");
        stat = cudaStreamDestroy(d_data->nb_stream); 
        CU_RET_ERR(stat, "cudaStreamDestroy failed on d_data->nb_stream");
    }

    stat = cudaGetTextureReference(&texnbfp, "texnbfp");
    CU_RET_ERR(stat, "cudaGetTextureReference on texnbfp failed");
    stat = cudaUnbindTexture(texnbfp);
    CU_RET_ERR(stat, "cudaUnbindTexture failed on texnbfp");

    stat = cudaFree(d_data->nbfp);
    CU_RET_ERR(stat, "cudaFree failed on d_data->nbfp");

    destroy_cudata_atoms(d_data);

    destroy_cudata_ci(d_data);
    destroy_cudata_sj(d_data);
    destroy_cudata_si(d_data);

    stat = cudaThreadExit();
    CU_RET_ERR(stat, "cudaThreadExit failed");

    fprintf(fplog, "Cleaned up CUDA data structures.\n");
}

void destroy_cudata_atoms(t_cudata d_data)
{
    cudaError_t stat;

    stat = cudaFree(d_data->f);
    CU_RET_ERR(stat, "cudaFree failed on d_data->f");
    stat = cudaFree(d_data->xq);   
    CU_RET_ERR(stat, "cudaFree failed on d_data->xq");
    stat = cudaFree(d_data->atom_types);   
    CU_RET_ERR(stat, "cudaFree failed on d_data->atom_types");
    d_data->natoms = -1;
    d_data->nalloc = -1;
}

void destroy_cudata_ci(t_cudata d_data)
{
    cudaError_t stat;

    stat = cudaFree(d_data->ci);
    CU_RET_ERR(stat, "cudaFree failed on d_data->ci");
    d_data->nci = -1;
    d_data->ci_nalloc = -1;
}

void destroy_cudata_sj(t_cudata d_data)
{
    cudaError_t stat;

    stat = cudaFree(d_data->sj);
    CU_RET_ERR(stat, "cudaFree failed on d_data->sj");
    d_data->nsj_1 = -1;
    d_data->sj_nalloc = -1;
}

void destroy_cudata_si(t_cudata d_data)
{
    cudaError_t stat;

    stat = cudaFree(d_data->si);
    CU_RET_ERR(stat, "cudaFree failed on d_data->si");
    d_data->nsi = -1;
    d_data->si_nalloc = -1;
}

int cu_upload_X(t_cudata d_data, real *h_x) 
{
    if (debug) printf(">> uploading X\n");
    return upload_cudata(d_data->xq, h_x, d_data->natoms*sizeof(*d_data->xq));
}

int cu_download_F(real *h_f, t_cudata d_data)
{
    if (debug) printf(">> downloading F\n");
    return download_cudata(h_f, d_data->f, d_data->natoms*sizeof(*d_data->f));
}
