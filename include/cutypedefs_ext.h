#ifndef CUTYPEDEFS_EXT_H
#define CUTYPEDEFS_EXT_H

#define GPU_NS_CELL_SIZE    8

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cudata * t_cudata;

typedef struct gpu_times t_gpu_times;

struct gpu_times
{
    float   nb_total_time;  /* total execution time of the nonbonded gpu operations:
                               - trasfer to/from GPU: x, q, shifts, f
                               - kernel exection */
    int     nb_count;       /* call count of the nonbonded gpu operations */

    float   atomdt_trans_total_time;/* total time of the data trasnfer after a neighbor search step */
    int     atomdt_count;           /* cll count   - || - */
};

#ifdef __cplusplus
}
#endif

#endif
