#include <chrono>
#include <iostream>
#include "cpu.h"
#include "gpu.h"

int n = 6;
unsigned long long length = 1 << n;
bool periodical_bc = false;

int main()
{
    std::cout << sizeof(unsigned long long int) << '\n';
    auto t1 = std::chrono::high_resolution_clock::now();
    short *J, *S;
    unsigned long long *G;
    cudaMallocManaged(&J, pow(n, 4) * sizeof(short));
    cudaMallocManaged(&S, n * n * sizeof(short));
    int E_max = 2 * n * n;
    int M_max = n * n;
    int E_num = 2 * E_max + 1;
    int M_num = 2 * M_max + 1;
    cudaMallocManaged(&G, E_num * M_num * sizeof(unsigned long long));
    std::string cell_name = "data/cell" + std::to_string(n * n) + "_J0_0.dat";
    cell_read(J, cell_name);
    cudaDeviceProp dev{};
    cudaGetDeviceProperties(&dev, 0);
    static size_t block_dim = 512;
    static size_t grid_dim = get_SP_cores(dev);
    std::cout << "sp_cores: " << get_SP_cores(dev) << "\n";
    initializer<<<grid_dim, block_dim>>>(G, E_max, M_max, n);
    //calculate_dos<<<grid_dim, block_dim>>>(J, S, G, E_num, M_max, n, periodical_bc);
    calculate_dos<<<1, 1>>>(J, S, G, E_max, M_max, n, periodical_bc);
    cudaDeviceSynchronize();
    out(G, E_max, M_max);
    cudaFree(J);
    cudaFree(S);
    cudaFree(G);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cout << "working time is " << time / 3600000 << " h " << (time % 3600000) / 60000 << " m "
              << ((time % 3600000) % 60000) / 1000 << " s " << ((time % 3600000) % 60000) % 1000 << " ms \n";
}
