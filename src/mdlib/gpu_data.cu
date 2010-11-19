#include <stdlib.h>
#include <stdio.h>

#include "gmx_fatal.h"
#include "smalloc.h"

#include "cutypedefs.h"
#include "cudautils.h"
#include "gpu_data.h"

#define USE_CUDA_ENVENT_BLOCKING_SYNC TRUE

/*** CUDA MD Data operations ***/

/* forward declaration*/
void destroy_cudata_atoms(t_cudata /*d_data*/);
void destroy_cudata_nblist(t_cudata /*d_data*/);
void destroy_cudata_cj(t_cudata /*d_data*/);

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

    int eventflags = ( USE_CUDA_ENVENT_BLOCKING_SYNC ? cudaEventBlockingSync: cudaEventDefault );

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
    d_data->nblist  = NULL;
    d_data->cj      = NULL;

    /* size -1 just means that it has not been initialized yet */
    d_data->natoms          = -1;
    d_data->nalloc          = -1;
    d_data->nlist           = -1;
    d_data->nblist_nalloc   = -1;
    d_data->ncj             = -1;
    d_data->cj_nalloc       = -1;
    d_data->napc            = -1;

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

/* natoms gets the value of fr->natoms_force */
void init_cudata_atoms(t_cudata d_data, 
                       const gmx_nb_atomdata_t *atomdata, 
                       const gmx_nblist_t *nblist,
                       gmx_bool doStream)
{
    cudaError_t stat;
    int         nalloc, nblist_nalloc, cj_nalloc;
    int         natoms  = atomdata->natoms;
    int         nlist   = nblist->nlist;
    int         ncj     = nblist->ncj;
   
    /* asynch copy all data */
    cudaEventRecord(d_data->start_atomdata, 0);

    if (d_data->napc < 0)
    {
        d_data->napc = nblist->napc;
    }
    else
    {
        if (d_data->napc != nblist->napc)
        {
            gmx_fatal(FARGS, "Internal error: the #atoms per cell has changed (from %d to %d)",
                    d_data->nblist, nblist->napc);
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

    if (nlist > d_data->nblist_nalloc) 
    {
        nblist_nalloc = nlist * 1.2 + 100; // FIXME

        /* free up first if the arrays have already been initialized */
        if (d_data->nblist_nalloc != -1)
        {
            destroy_cudata_nblist(d_data);                
        }

        stat = cudaMalloc((void **)&d_data->nblist, nblist_nalloc*sizeof(*(d_data->nblist)));
        CU_RET_ERR(stat, "cudaMalloc failed on d_data->nblist");           

        d_data->nblist_nalloc = nblist_nalloc;
    }
    d_data->nlist = nlist;

    if (ncj > d_data->ncj) 
    {
        cj_nalloc = ncj * 1.2 + 100; // FIXME

        /* free up first if the arrays have already been initialized */
        if (d_data->cj_nalloc != -1)
        {
            destroy_cudata_cj(d_data);                
        }

        stat = cudaMalloc((void **)&d_data->cj, cj_nalloc*sizeof(*(d_data->cj)));
        CU_RET_ERR(stat, "cudaMalloc failed on d_data->nblist");    

        d_data->cj_nalloc = cj_nalloc;
    }
    d_data->ncj = ncj;

    if(doStream)
    {
        upload_cudata_async(d_data->atom_types, atomdata->type, natoms*sizeof(*(d_data->atom_types)), 0);
        upload_cudata_async(d_data->nblist, nblist->list, nlist*sizeof(*(d_data->nblist)), 0);
        upload_cudata_async(d_data->cj, nblist->cj, ncj*sizeof(*(d_data->cj)), 0);
    }
    else 
    {
        upload_cudata(d_data->atom_types, atomdata->type, natoms*sizeof(*(d_data->atom_types)));
        upload_cudata(d_data->nblist, nblist->list, nlist*sizeof(*(d_data->nblist)));
        upload_cudata(d_data->cj, nblist->cj, ncj*sizeof(*(d_data->cj)));
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
        stat = cudaStreamDestroy(d_data->nb_stream); /* TODO check stat !*/
        CU_RET_ERR(stat, "cudaStreamDestroy failed on d_data->nb_stream");
    }

    stat = cudaGetTextureReference(&texnbfp, "texnbfp");
    CU_RET_ERR(stat, "cudaGetTextureReference on texnbfp failed");
    stat = cudaUnbindTexture(texnbfp);
    CU_RET_ERR(stat, "cudaUnbindTexture failed on texnbfp");

    stat = cudaFree(d_data->nbfp);
    CU_RET_ERR(stat, "cudaFree failed on d_data->nbfp");

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

void destroy_cudata_nblist(t_cudata d_data)
{
    cudaError_t stat;

    stat = cudaFree(d_data->nblist);
    CU_RET_ERR(stat, "cudaFree failed on d_data->nblist");
    d_data->nlist = -1;
    d_data->nblist_nalloc = -1;
}

void destroy_cudata_cj(t_cudata d_data)
{
    cudaError_t stat;

    stat = cudaFree(d_data->cj);
    CU_RET_ERR(stat, "cudaFree failed on d_data->cj");
    d_data->ncj = -1;
    d_data->cj_nalloc = -1;
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
