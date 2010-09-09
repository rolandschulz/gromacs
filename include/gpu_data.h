#ifndef _GPU_DATA_H_
#define _GPU_DATA_H_

#include "typedefs.h" 
#include "cutypedefs_ext.h"

#ifdef __cplusplus
extern "C" {
#endif

void init_cudata_ff(FILE * /*fplog*/, 
                    t_cudata * /*dp_data*/,      
                    const t_forcerec * /*fr*/);

void init_cudata_atoms(t_cudata /*d_data*/,
                        const t_mdatoms * /*mdatoms*/,
                        int /*natoms*/);

void destroy_cudata(FILE * /*fplog*/, 
                    t_cudata  /*d_data*/);

int cu_upload_X(t_cudata /*d_data*/, 
                 rvec h_x[]);

int cu_download_F(rvec h_f[], 
                   t_cudata /*d_data*/);

#ifdef __cplusplus
}
#endif

#endif // _GPU_DATA_H_
