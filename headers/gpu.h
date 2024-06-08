#ifndef gpu_h
#define gpu_h

int get_SP_cores(cudaDeviceProp devProp);
__global__ void initializer(unsigned long long *G, int E_max, int M_max, int n);
__global__ void calculate_dos(short *J, short *S, unsigned long long *G, int E_max, int M_max, int n, bool bc);

#endif