#include <iostream>
#include "gpu.h"

int get_SP_cores(cudaDeviceProp devProp)
{
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major){
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        case 7: // Volta and Turing
            if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        case 8: // Ampere
            if (devProp.minor == 0) cores = mp * 64;
            else if (devProp.minor == 6) cores = mp * 128;
            else printf("Unknown device type\n");
            break;
        default:
            printf("Unknown device type\n");
            break;
    }
    return cores;
}

__global__ void initializer(unsigned long long *G, int E_max, int M_max, int n)
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    for(auto i = x; i < (E_max * M_max); i += blockDim.x * gridDim.x)
    {
        G[i] = 0;
    }
}

__global__ void calculate_dos(short *J, short *S, unsigned long long *G, int E_max, int M_max, int n, bool bc)
{
    if(!bc)
    {
        for(unsigned long long conf = 0; conf < 1 << (n * n); ++conf)
        {
            int E = 0;
            int M = 0;
            int bit = conf;
            for(auto i = 0; i < (n * n); ++i)
            {
                S[i] = bit & 1 ? 1 : -1;
                bit >>= 1; 
            }
            for(auto i = 0; i < n; ++i)
            {
                int k = 1;
                for(auto j = 0; j < n; ++j)
                {
                    E += - S[n * i + j] * S[n * i + (j + 1) % n];
                    E += - S[n * i + j] * S[(n * i + j + n) % (n * n)];
                    M += S[n * i + j];
                    k += n * n + 1;
                }
            }
            E = E + E_max;
            M = M + M_max;
            atomicAdd(&G[E * M_max + M], 1);
        }
    }
}