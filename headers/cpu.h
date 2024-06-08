#ifndef cpu_h
#define cpu_h

extern int n;

void cell_read(short *J, const std::string& cell_name);
void out(unsigned long long *G, int E_max, int M_max);

#endif