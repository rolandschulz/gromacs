#ifndef GPU_NB_H
#define GPU_NB_H

#include "cutypedefs_ext.h"

#ifdef __cplusplus
extern "C" {
#endif

void cu_stream_nb(t_cudata /*d_data*/, gmx_nblist_t * /*nblist*/, gmx_nb_atomdata_t * /*nbdata*/, rvec f[]);
void cu_do_nb(t_cudata /*d_data*/);
gmx_bool cu_checkstat_nb(t_cudata /*d_data*/, float * /*time*/);
void cu_blockwait_nb(t_cudata /*d_data*/, float * /*time*/);

#ifdef __cplusplus
}
#endif

#endif
