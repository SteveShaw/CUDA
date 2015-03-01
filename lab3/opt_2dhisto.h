#ifndef OPT_KERNEL
#define OPT_KERNEL


extern "C" void opt_init(unsigned int** h_Data, int width, int height);
extern "C" void opt_2dhisto(int size);
extern "C" void opt_copyFromDevice(unsigned char* output);
extern "C" void opt_free();

#endif
