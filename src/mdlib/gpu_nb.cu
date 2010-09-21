#include "stdlib.h"

#include "smalloc.h"

#include "cutypedefs.h"
#include "cudautils.h"

#include "gpu_nb.h"
#include "gpu_data.h"
#include "gpu_nb_kernels.h"


__global__ void __empty_kernel() {}

void cu_do_nb(t_cudata d_data, rvec x[], rvec f[])
{
    int     nb_blocks = (d_data->natoms % NB_DEFAULT_THREADS == 0 ? 
                d_data->natoms/NB_DEFAULT_THREADS : 
                d_data->natoms/NB_DEFAULT_THREADS + 1);
    dim3    dim_block(nb_blocks, 1, 1);
    dim3    dim_grid(NB_DEFAULT_THREADS, 1, 1);

    /* do the nonbonded calculations */
    k_calc_nb<<<dim_block, dim_grid>>>(d_data->f, d_data->x, d_data->natoms);
    CU_LAUNCH_ERR("k_calc_nb");
}


void cu_stream_nb(t_cudata d_data, rvec x[], rvec f[])
{
    int     nb_blocks = (d_data->natoms % NB_DEFAULT_THREADS == 0 ? 
                d_data->natoms/NB_DEFAULT_THREADS : 
                d_data->natoms/NB_DEFAULT_THREADS + 1);
    dim3    dim_block(nb_blocks, 1, 1);
    dim3    dim_grid(NB_DEFAULT_THREADS, 1, 1);

        /* async copy HtoD x */
    cudaEventRecord(d_data->start_nb, 0);
    cudaMemcpyAsync(d_data->x, x, d_data->natoms, cudaMemcpyHostToDevice, 0);

    /* do the nonbonded calculations */
    //k_calc_nb<<<dim_block, dim_grid>>>(d_data->f, d_data->x, d_data->natoms);
    //CU_LAUNCH_ERR("k_calc_nb");
   
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
