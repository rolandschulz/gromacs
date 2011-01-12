#ifndef CUTYPEDEFS_EXT_H
#define CUTYPEDEFS_EXT_H

#define GPU_NS_CELL_SIZE    8

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cudata * t_cudata;

typedef struct gpu_times gpu_times_t;

struct gpu_times
{
    float   nb_total_time;  /* total execution time of the nonbonded gpu operations:
                               - trasfer to/from GPU: x, q, shifts, f
                               - kernel exection */
    float   nb_h2d_time;    /* host to device transfer time of data */
    float   nb_d2h_time;    /* device to host transfer time of data */
    int     nb_count;       /* total call count of the nonbonded gpu operations */
    int     nb_count_ene;   /* callc count of the force + energy nonbonded gpu operations */

    float   atomdt_h2d_total_time;  /* total time of the data trasnfer after a neighbor search step */
    int     atomdt_count;           /* cll count   - || - */
    
};

#ifdef __cplusplus
}
#endif

#endif
