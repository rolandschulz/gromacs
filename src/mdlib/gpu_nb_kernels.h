
#define LJ_C6       0
#define LJ_C12      0

/*
inline __host__ __device__ float3 make_float3(float x)
{
    return make_float3(x, x, x);
}
*/

inline __host__ __device__ float3 make_float3(float s)
{
    return make_float3(s, s, s);
}
inline __host__ __device__ float3 operator-(float3 &a)
{
    return make_float3(-a.x, -a.y, -a.z);
}
inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ float3 operator*(float3 a, float k)
{
    return make_float3(k * a.x, k * a.y, k * a.z);
}
inline __host__ __device__ float norm(float3 a)
{
    return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}
inline __host__ __device__ float dist3(float3 a, float3 b)
{
//    float dx, dy, dz;
//    dx = a.x - b.x;
//    dy = a.y - b.y;
//    dz = a.z - b.z;
    return norm(b - a);
}


inline __host__ __device__ float4 make_float4(float s)
{
    return make_float4(s, s, s, s);
}
inline __host__ __device__ float4 operator+=(float4 a, float3 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w);
}


/*  Launch parameterts: 
        - #bloks    = 
        - #threads  = #pairs of atoms in cell i and j = |Ci| x |Cj| 
        - shmem     
            - EXT - for i atoms 
            - for all j atoms 
        - registers  =
    Each thread calculated one  pair of atoms from neigbouring i-j cells.
*/
__global__ void k_calc_nb(struct cudata devData)
{
    t_cudata devD = &devData;
    unsigned int tidxx  = threadIdx.x; // local
    unsigned int tidxy  = threadIdx.y; 
    unsigned int tidx   = tidxx * gridDim.x + tidxy;
    //unsigned int tpb    = gridDim.x * gridDim.y;
    unsigned int bidx   = blockIdx.x;

    int ci, cj, ai, aj, cij_start, cij_end; 
    int nbidx;
    float qi, qj, inv_r, inv_r2, inv_r6, c6, c12, V_LJ, dVdr;
    float4 f4tmp;
    float3 xi, xj, rv;

    __shared__ float3 forcebuf[CELL_SIZE * CELL_SIZE];  // force buffer 
    forcebuf[tidx] = make_float3(0.0f);

    ci          = bidx; // i cell index = current block index
    ai          = ci * CELL_SIZE + tidxx;
    // f2tmp       = devD->cell_nblist_idx[ci];
    // nb_count    = f2tmp.y - f2tmp.x + 1;
    cij_start   = devD->nblist[ci].jind_start;
    cij_end     = devD->nblist[ci].jind_end;

    // load i atom data into registers
    f4tmp       = devD->xq[ai];
    xi          = make_float3(f4tmp.x, f4tmp.y, f4tmp.z);
    qi          = f4tmp.w;

    // loop over i-s neighboring cells
    // If not, cell_nblist_idx should be loaded into constant mem
    for (nbidx = cij_start ; nbidx < cij_end; nbidx++)
    {        
        /*
        printf("Thread %d(%d, %d) in block %d working on ci=%d, ai=%d; cj=%d, aj=%d\n",
                tidx, tidxx, tidxy, bidx, ci, ai, cj, aj);
        */
        cj      = devD->cj[nbidx]; // TODO quite uncool operation
        aj      = cj * CELL_SIZE + tidxy;

        // all threads load an atom from j cell into shmem!
        f4tmp   = devD->xq[aj];
        xj      = make_float3(f4tmp.x, f4tmp.y, f4tmp.y);
        qj      = f4tmp.w;

        c6          = LJ_C6; 
        c12         = LJ_C12; 
        rv          = xj - xi;
        inv_r       = 1.0f / norm(rv);
        inv_r2      = inv_r * inv_r;
        inv_r6      = inv_r2 * inv_r2 * inv_r2;
        V_LJ        = inv_r6 * (c12 * inv_r2 - c6);
        dVdr        = V_LJ * inv_r2;
        // accumulate forces into shmem
        forcebuf[tidx].x += rv.x * dVdr; 
        forcebuf[tidx].y += rv.y * dVdr; 
        forcebuf[tidx].z += rv.z * dVdr;                  
        __syncthreads();
    }        
    // reduce forces         
    // XXX lame reduction 
    float3 fsum = make_float3(0.0f);
    if (tidxy == 0)
    {
        for (int i = ai * CELL_SIZE; i < (ai + 1) * CELL_SIZE; i++)
        {
            fsum.x += forcebuf[i].x;
            fsum.y += forcebuf[i].y;
            fsum.z += forcebuf[i].z;
        }
        devD->f[ai] = fsum;
    }
}

__global__ void k_calc_bboxes()
{

}
