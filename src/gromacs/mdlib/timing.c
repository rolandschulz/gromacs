#include "timing.h"
#include <malloc.h>

code_timer *create_code_timer()
{
  code_timer *ct = malloc(sizeof(code_timer));
  return ct;
}

void free_code_timer(code_timer *ct)
{
  free(ct);
}

void reset_timer(code_timer *ct)
{
  gettimeofday(&(ct->t), NULL);
  ct->cp = (double)ct->t.tv_sec + 1e-6*(double)ct->t.tv_usec;
}

double get_elapsed_time(code_timer *ct)
{
  double tnow;
  struct timeval tnowval;

  gettimeofday(&tnowval, NULL);
  tnow = (double)tnowval.tv_sec + 1e-6*(double)tnowval.tv_usec;
  return (tnow - ct->cp);
}
