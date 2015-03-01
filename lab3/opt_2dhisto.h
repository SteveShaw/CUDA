#ifndef OPT_KERNEL
#define OPT_KERNEL

void opt_init(unsigned int** h_Data, int width, int height);
void opt_2dhisto(int size);
void opt_copyFromDevice(unsigned int* h_Histogoram);
void opt_free();


#endif
