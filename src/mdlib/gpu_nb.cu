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


/* texture reference bound to the cudata.coulomb_tab array */
texture<float, 1, cudaReadModeElementType> tex_coulomb_tab;

/* source: OpenMM */
static __device__ float interpolate_coulomb_force_r(float r, float scale)
{  
    float   normalized = scale * r;
    int     index = (int) normalized;
    float   fract2 = normalized - index;
    float   fract1 = 1.0f - fract2;

    return  fract1 * tex1Dfetch(tex_coulomb_tab, index) 
            + fract2 * tex1Dfetch(tex_coulomb_tab, index + 1);
}

#include "gpu_nb_kernels.h"

/* based on the number of work units, return the number of blocks to be used 
   for the nonbonded GPU kernel */
inline int calc_nb_blocknr(int nwork_units)
{
    int retval = (nwork_units <= GRID_MAX_DIM ? nwork_units : GRID_MAX_DIM);
    if (retval != nwork_units)
    {
        gmx_fatal(FARGS, "Watch out, the number of nonbonded work units exceeds the maximum grid size (%d > %d)!",
                nwork_units, GRID_MAX_DIM);
    }
    return retval;
}

/*  Launch asynchronously the nonbonded force calculations. 

    This consists of the following (async) steps launched in the default stream 0: 
   - initilize to zero force output
   - upload x and q
   - upload shift vector
   - launch kernel
   - download forces
    
    Timing is done using the start_nb and stop_nb events.
 */
void cu_stream_nb(t_cudata d_data, 
                  const gmx_nb_atomdata_t *nbatom,
                  gmx_bool sync)
{
    int     shmem; 
    int     nb_blocks = calc_nb_blocknr(d_data->nci);
    dim3    dim_block(CELL_SIZE, CELL_SIZE, 1); 
    dim3    dim_grid(nb_blocks, 1, 1); 
    
    static gmx_bool doKernel2 = (getenv("GMX_NB_K2") != NULL);        

    /* size of force buffers in shmem */
    if (!doKernel2)
    {
        shmem =  (1 + NSUBCELL) * CELL_SIZE * CELL_SIZE * 3 * sizeof(float);
    }
    else 
    {
        shmem =  CELL_SIZE * CELL_SIZE * 3 * sizeof(float);
    }

    if (debug)
    {
        fprintf(debug, "GPU launch configuration:\n\tThread block: %dx%dx%d\n\tGrid: %dx%d\n\t#Cells/Subcells: %d/%d (%d)\n",         
        dim_block.x, dim_block.y, dim_block.z, dim_grid.x, dim_grid.y, d_data->nsi, 
        NSUBCELL, d_data->naps);
    }

    cudaEventRecord(d_data->start_nb, 0);
    
    /* set the forces to 0 */
    cudaMemsetAsync(d_data->f, 0, d_data->natoms * sizeof(*d_data->f), 0);

    /* upload x, q */    
    upload_cudata_async(d_data->xq, nbatom->x, d_data->natoms * sizeof(*d_data->xq), 0);

    /* upload shift vec */
    upload_cudata_async(d_data->shift_vec, nbatom->shift_vec, SHIFTS * sizeof(*d_data->shift_vec), 0);   

    /* async nonbonded calculations */        
    if (!doKernel2)
    {
        k_calc_nb_1 <<<dim_grid, dim_block, shmem, 0>>>(d_data->ci,
                                                        d_data->sj,
                                                        d_data->si,
                                                        d_data->atom_types, 
                                                        d_data->ntypes, 
                                                        d_data->xq, 
                                                        d_data->nbfp,
                                                        d_data->shift_vec,
                                                        d_data->ewald_beta,
                                                        d_data->cutoff_sq,
                                                        d_data->coulomb_tab_scale,
                                                        d_data->f);
    }
    else
    {
        k_calc_nb_2 <<<dim_grid, dim_block, shmem, 0>>>(d_data->ci,
                                                        d_data->sj,
                                                        d_data->si,
                                                        d_data->atom_types,
                                                        d_data->ntypes,
                                                        d_data->xq,
                                                        d_data->nbfp,
                                                        d_data->shift_vec,
                                                        d_data->ewald_beta,
                                                        d_data->cutoff_sq,
                                                        d_data->coulomb_tab_scale,
                                                        d_data->f);
    }
   
    if (sync)
    {
        CU_LAUNCH_ERR_SYNC("k_calc_nb");
    }
    else
    {
        CU_LAUNCH_ERR("k_calc_nb");
    }
   
    /* async copy DtoH f */
    download_cudata_async(nbatom->f, d_data->f, d_data->natoms*sizeof(*d_data->f), 0);
    cudaEventRecord(d_data->stop_nb, 0);
}

/* Blocking wait for the asynchrounously launched nonbonded calculations to finish. */
void cu_blockwait_nb(t_cudata d_data, float *time)
{    
    cu_blockwait_event(d_data->stop_nb, d_data->start_nb, time);
}

/* Blocking wait for the asynchrounously launched nonbonded calculations to finish. */
void cu_blockwait_nb_OLD(t_cudata d_data, float *time)
{    
    cudaError_t stat;

    // stat = cudaStreamSynchronize(d_data->nb_stream);    
    stat = cudaEventSynchronize(d_data->stop_nb);
    CU_RET_ERR(stat, "the async execution of nonbonded calculations has failed"); 

    stat = cudaEventElapsedTime(time, d_data->start_nb, d_data->stop_nb);
    CU_RET_ERR(stat, "cudaEventElapsedTime on start_nb and stop_nb failed");
}

/* Check if the nonbonded calculation has finished. */
gmx_bool cu_checkstat_nb(t_cudata d_data, float *time)
{
    cudaError_t stat; 
    
    time = NULL;
    stat = cudaEventQuery(d_data->stop_nb);

    /* we're done, let's calculate times*/
    if (stat == cudaSuccess)
    {
        stat = cudaEventElapsedTime(time, d_data->start_nb, d_data->stop_nb);
        CU_RET_ERR(stat, "cudaEventElapsedTime on start_nb and stop_nb failed");
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


/* XXX:  remove, not used anyomore! */
void cu_do_nb(t_cudata d_data, rvec shift_vec[]) 
{
#if 0
    int     nb_blocks = calc_nb_blocknr(d_data->nci);
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
    upload_cudata(d_data->shift_vec, shift_vec, SHIFTS*sizeof(*d_data->shift_vec));   

    /* sync nonbonded calculations */      
    k_calc_nb_1<<<dim_grid, dim_block, shmem>>>(d_data->ci,
                                                  d_data->sj, 
                                                  d_data->si,
                                                  d_data->atom_types, 
                                                  d_data->ntypes, 
                                                  d_data->xq, 
                                                  d_data->nbfp,
                                                  d_data->shift_vec,
                                                  d_data->ewald_beta,
                                                  d_data->f);
    CU_LAUNCH_ERR_SYNC("k_calc_nb");
#endif 
}
