#include <sys/time.h>

#define gmx_offload __declspec(target(mic))

gmx_offload
typedef struct code_timer_struct
{
  double cp;
  struct timeval t;
} code_timer;

#ifdef __cplusplus
extern "C" {
#endif
gmx_offload code_timer *create_code_timer();
gmx_offload void free_code_timer(code_timer *ct);
gmx_offload void reset_timer(code_timer *ct);
gmx_offload double get_elapsed_time(code_timer *ct);
#ifdef __cplusplus
}
#endif
