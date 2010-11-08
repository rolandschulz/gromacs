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

texture<float, 1, cudaReadModeElementType> texnbfp;

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

void cu_do_nb(t_cudata d_data)
{
    int     nb_blocks = calc_nb_blocknr(d_data->nlist);
    dim3    dim_block(CELL_SIZE, CELL_SIZE, 1); 
    dim3    dim_grid(nb_blocks, 1, 1); 
    int     shmem = CELL_SIZE * CELL_SIZE * sizeof(float4); /* force buffer */

    if (debug)
    {
        printf("~> Thread block: %dx%dx%d\n~> Grid: %dx%d\n~> #Cells: %d (%d)\n", 
            dim_block.x, dim_block.y, dim_block.z, dim_grid.x, dim_grid.y, d_data->ncj, d_data->napc);
    }


    /* sync nonbonded calculations */   
#if 0    
    k_calc_nb<<<dim_grid, dim_block, shmem>>>(*d_data);
#else
    k_calc_nb<<<dim_grid, dim_block, shmem>>>(d_data->nblist,             
                                                  d_data->ntypes, 
                                                  d_data->xq, 
                                                  d_data->atom_types, 
                                                  d_data->cj, 
                                                  d_data->nbfp,
                                                  d_data->f,
                                                  d_data->cell_pair_group);
#endif    
    CU_LAUNCH_ERR_SYNC("k_calc_nb");
}

void cu_stream_nb(t_cudata d_data, 
                  /*const gmx_nblist_t *nblist, */
                  const gmx_nb_atomdata_t *nbatom)
{
    int     nb_blocks = calc_nb_blocknr(d_data->nlist)/d_data->cell_pair_group;
    dim3    dim_block(CELL_SIZE, CELL_SIZE, 1); 
    dim3    dim_grid(nb_blocks, 1, 1); 
    int     shmem = CELL_SIZE * CELL_SIZE * sizeof(float4); /* force buffer 4*4*CELL_SIZE^2 */
    // cudaStream_t st = d_data->nb_stream;
    static int     cacheConf = 0;


    /* XXX XXX */
    if (cacheConf == 0)
    {
        cudaFuncSetCacheConfig(&k_calc_nb, cudaFuncCachePreferL1); 
        cacheConf++;
    }


    if (debug)
    {
        printf("~> Thread block: %dx%dx%d\n~> Grid: %dx%d\n~> #Cells: %d (%d)\n", 
            dim_block.x, dim_block.y, dim_block.z, dim_grid.x, dim_grid.y, d_data->ncj, d_data->napc);
    }  

    /* async copy HtoD x */
    cudaEventRecord(d_data->start_nb, 0);
    cudaMemcpyAsync(d_data->xq, nbatom->x, d_data->natoms*sizeof(*d_data->xq), 
                    cudaMemcpyHostToDevice, 0);    

    /* async nonbonded calculations */        
#if 0
    k_calc_nb<<<dim_grid, dim_block, shmem, 0>>>(*d_data);
#else
    k_calc_nb<<<dim_grid, dim_block, shmem, 0>>>(d_data->nblist,             
                                                  d_data->ntypes, 
                                                  d_data->xq, 
                                                  d_data->atom_types, 
                                                  d_data->cj, 
                                                  d_data->nbfp,
                                                  d_data->f, 
                                                  d_data->cell_pair_group);
#endif
    CU_LAUNCH_ERR("k_calc_nb");
   
    /* async copy DtoH f */    
    cudaMemcpyAsync(nbatom->f, d_data->f, d_data->natoms*sizeof(*d_data->f), 
                    cudaMemcpyDeviceToHost, 0);
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
