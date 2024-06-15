#include <iostream>
#include <fstream>
#include <cmath>
#include "cpu.h"

int J_sum = 0;

void cell_read(short *J, const std::string& cell_name)
{
    std::ifstream J_file(cell_name);
    if(!J_file.is_open())
    {
        std::cout << "not found " << cell_name << "\n";
    }
    for(auto i = 0; i < std::pow(n, 4); ++i)
    {
        J_file >> J[i];
        J_sum += J[i];
    }
}

void out(unsigned long long *G, int E_max, int M_max)
{
    int E_num = 2 * E_max + 1;
    int M_num = 2 * M_max + 1;
    std::ofstream gem_out("data/dos" + std::to_string(n * n) + "_J" + std::to_string(J_sum) + "_0.dat");
    gem_out << n << "\n";
    int size = 0;
    for (auto E = 0; E < E_num; ++E)
        for (auto M = 0; M < M_num; ++M)
            if (G[E * M_max + M] > 0)
            {
                size++;
            }
    gem_out << size << "\n";
    gem_out << J_sum << "\n";
    gem_out << 0 << "\n";
    unsigned long long G_sum = 0;
    for (auto E = 0; E < E_num; ++E)
    {
        for (auto M = 0; M < M_num; ++M)
        {
            if (G[E * M_num + M] > 0)
            {
                gem_out << G[E * M_num + M] << " " << E - E_max << " " << M - M_max << "\n";
            }
            G_sum += G[E * M_num + M];
        }
    }
    gem_out.close();
    unsigned long long G_sum_t = ONE << (n * n);
    std::cout << "Gsum = " << G_sum << '\n'
              << "Gsum must be = " << G_sum_t << '\n';
}
