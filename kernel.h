#define uchar unsigned char

void mygpuinrange(int width,int height,gpu::PtrStep<uchar3> src,gpu::PtrStep<uchar3> dst,uchar3 min, uchar3 max);
__global__ void GpuinRange(gpu::PtrStep<uchar3> src,gpu::PtrStep<uchar3> dst,int width,int height,uchar3 valmin, uchar3 valmax);
