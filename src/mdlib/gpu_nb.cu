#include "stdlib.h"

#include "smalloc.h"

#include "types/simple.h" 
#include "types/nblist_box.h"
#include "cutypedefs.h"
#include "cudautils.h"

#include "gpu_nb.h"
#include "gpu_data.h"

#define CELL_SIZE           32
#define NB_DEFAULT_THREADS  (CELL_SIZE * CELL_SIZE)// 256

#include "gpu_nb_kernels.h"

__global__ void __empty_kernel() {}

inline int calc_nb_blocknr(int nwork_units)
{
    return (nwork_units % NB_DEFAULT_THREADS == 0 ? 
                nwork_units/NB_DEFAULT_THREADS : 
                nwork_units/NB_DEFAULT_THREADS + 1);
}

void cu_do_nb(t_cudata d_data)
{
    int     nb_blocks = calc_nb_blocknr(d_data->ncj);
    dim3    dim_block(CELL_SIZE, CELL_SIZE, 1); 
    dim3    dim_grid(nb_blocks, 1, 1); 

    printf("~> Thread block: %dx%dx%d\n~> Grid: %dx%d\n~> #Cells: %d (%d)", 
        dim_block.x, dim_block.y, dim_block.z, dim_grid.x, dim_grid.y, d_data->ncj, d_data->napc);
    /* sync nonbonded calculations */
    printf(">> executing nb kernel\n");
    k_calc_nb<<<dim_block, dim_grid>>>(*d_data);
    CU_LAUNCH_ERR("k_calc_nb");
}

void cu_stream_nb(t_cudata d_data, gmx_nblist_t *nblist, gmx_nb_atomdata_t *nbatom, rvec f[])
{
    int     nb_blocks = calc_nb_blocknr(d_data->ncj);
    dim3    dim_block(CELL_SIZE, CELL_SIZE, 1); 
    dim3    dim_grid(nb_blocks, 1, 1); 
   // int     shmem = 0;

    /* async copy HtoD x */
    cudaEventRecord(d_data->start_nb, 0);
    cudaMemcpyAsync(d_data->xq, nbatom->x, d_data->natoms, cudaMemcpyHostToDevice, 0);    

    /* async nonbonded calculations */
    k_calc_nb<<<dim_block, dim_grid, 0>>>(*d_data);
    CU_LAUNCH_ERR("k_calc_nb");
   
    /* async copy DtoH f */
    cudaMemcpyAsync(f, d_data->f, d_data->natoms, cudaMemcpyDeviceToHost, 0);
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

    stat = cudaEventSynchronize(d_data->stop_nb);
    CU_RET_ERR(stat, "the execution of the nonbonded calculations has failed");   
   
    cudaEventElapsedTime(time, d_data->start_nb, d_data->stop_nb);
}
