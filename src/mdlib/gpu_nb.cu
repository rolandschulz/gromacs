#include "stdlib.h"

#include "smalloc.h"

#include "types/simple.h" 
#include "types/nblist_box.h"
#include "cutypedefs.h"
#include "cudautils.h"

#include "gpu_nb.h"
#include "gpu_data.h"

#define CELL_SIZE           (GPU_NS_CELL_SIZE)
#define NB_DEFAULT_THREADS  (CELL_SIZE * CELL_SIZE)
#define GPU_FACEL           (138.935485)

texture<float, 1, cudaReadModeElementType> texnbfp;
// __device__ __constant__ c_nbfp;

#include "gpu_nb_kernels.h"

__global__ void __empty_kernel() {}

inline int calc_nb_blocknr(int nwork_units)
{
    int retval = (nwork_units <= GRID_MAX_DIM ? nwork_units : GRID_MAX_DIM);
    if (retval != nwork_units)
    {
        gmx_warning("Watch out, the number of nonbonded work units exceeds the maximum grid size (%d > %d)!",
                nwork_units, GRID_MAX_DIM);
    }
    return retval;
}

void cu_do_nb(t_cudata d_data,rvec shiftvec[]) 
{
    int     nb_blocks = calc_nb_blocknr(d_data->nci)/d_data->cell_pair_group;
    dim3    dim_block(CELL_SIZE, CELL_SIZE, 1); 
    dim3    dim_grid(nb_blocks, 1, 1); 
    int     shmem = (1 + NSUBCELL) * CELL_SIZE * CELL_SIZE * sizeof(float4); /* force buffer */

    if (debug)
    {
        printf("~> Thread block: %dx%dx%d\n~> Grid: %dx%d\n~> #SubCell pairs: %d (%d)\n", 
            dim_block.x, dim_block.y, dim_block.z, dim_grid.x, dim_grid.y, d_data->nsi, 
            d_data->naps);
    }

    /* set the forces to 0 */
    cudaMemset(d_data->f, 0, d_data->natoms*sizeof(*d_data->f));

    /* upload shift vec */
    upload_cudata(d_data->shiftvec, shiftvec, SHIFTS*sizeof(*d_data->shiftvec));   

    /* sync nonbonded calculations */   
    k_calc_nb<<<dim_grid, dim_block, shmem>>>(d_data->ci,
                                                  d_data->sj, 
                                                  d_data->si,
                                                  d_data->atom_types, 
                                                  d_data->ntypes, 
                                                  d_data->xq, 
                                                  d_data->nbfp,
                                                  d_data->shiftvec,
                                                  d_data->f);
    CU_LAUNCH_ERR_SYNC("k_calc_nb");
}

void cu_stream_nb(t_cudata d_data, 
                  /*const gmx_nblist_t *nblist, */
                  const gmx_nb_atomdata_t *nbatom,
                  rvec shiftvec[])
{
    int     nb_blocks = calc_nb_blocknr(d_data->nci)/d_data->cell_pair_group;
    dim3    dim_block(CELL_SIZE, CELL_SIZE, 1); 
    dim3    dim_grid(nb_blocks, 1, 1); 
    int     shmem =  (1 + NSUBCELL) * CELL_SIZE * CELL_SIZE * sizeof(float4); /* force buffer 4*4*CELL_SIZE^2 */
    // cudaStream_t st = d_data->nb_stream;
    static int     cacheConf = 0;


    /* XXX XXX */
    if (cacheConf == 0)
    {
        printf("~> Thread block: %dx%dx%d\n~> Grid: %dx%d\n~> #Cells/Subcells: %d/%d (%d)\n",         
        dim_block.x, dim_block.y, dim_block.z, dim_grid.x, dim_grid.y, d_data->nsi, 
        NSUBCELL, d_data->naps);

        printf("cell_pair_group=%d\n", d_data->cell_pair_group);
        cudaFuncSetCacheConfig(&k_calc_nb, cudaFuncCachePreferShared); 
        cacheConf++;
    }

    cudaEventRecord(d_data->start_nb, 0);
    
    /* set the forces to 0 */
    cudaMemsetAsync(d_data->f, 0, d_data->natoms*sizeof(*d_data->f), 0);
    /* upload x, Q */
    upload_cudata_async(d_data->xq, nbatom->x, d_data->natoms*sizeof(*d_data->xq), 0);
    /* upload shift vec */
    upload_cudata_async(d_data->shiftvec, shiftvec, SHIFTS*sizeof(*d_data->shiftvec), 0);   

    /* async nonbonded calculations */        
#if 0
    k_calc_nb<<<dim_grid, dim_block, shmem, 0>>>(*d_data);
#else
    k_calc_nb<<<dim_grid, dim_block, shmem, 0>>>(d_data->ci,             
                                                  d_data->sj, 
                                                  d_data->si,
                                                  d_data->atom_types, 
                                                  d_data->ntypes, 
                                                  d_data->xq, 
                                                  d_data->nbfp,
                                                  d_data->shiftvec,
                                                  d_data->f);    
#endif
    CU_LAUNCH_ERR("k_calc_nb");
   
    /* async copy DtoH f */    
    download_cudata_async(nbatom->f, d_data->f, d_data->natoms*sizeof(*d_data->f), 0);
    cudaEventRecord(d_data->stop_nb, 0);
}

gmx_bool cu_checkstat_nb(t_cudata d_data, float *time)
{
    cudaError_t stat; 
    
    time = NULL;
    stat = cudaEventQuery(d_data->stop_nb);

    /* we're done, let's calculate times*/
    if (stat == cudaSuccess)
    {
        cudaEventElapsedTime(time, d_data->start_nb, d_data->stop_nb);
    }
    else 
    {
        /* do we have an error? */
        if (stat != cudaErrorNotReady) 
        {
            CU_RET_ERR(stat, "the execution of the nonbonded calculations has failed");
        }
    }
    
    return (stat == cudaSuccess ? TRUE : FALSE);
}

void cu_blockwait_nb(t_cudata d_data, float *time)
{    
    cudaError_t stat;     

    // stat = cudaStreamSynchronize(d_data->nb_stream);
    stat = cudaEventSynchronize(d_data->stop_nb);
    CU_RET_ERR(stat, "the async execution of nonbonded calculations has failed");   
   
    cudaEventElapsedTime(time, d_data->start_nb, d_data->stop_nb);
}
