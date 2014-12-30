#define BLOCKSIZE 256
#include <iostream>

#include <opencv2/opencv_modules.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/opengl_interop.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

using namespace std;
using namespace cv;

#include "kernel.h"


void mygpuinrange(int width,int height,gpu::PtrStep<uchar3> src,gpu::PtrStep<uchar3> dst,uchar3 min, uchar3 max){
	dim3 threadsPerBlock(BLOCKSIZE,BLOCKSIZE);
	dim3 numblocks(width/BLOCKSIZE,height/BLOCKSIZE);
	GpuinRange<<<threadsPerBlock,numblocks>>>(src,dst,width,height,min,max);
}

__global__ void GpuinRange(gpu::PtrStep<uchar3> src,gpu::PtrStep<uchar3> dst,int width,int height,uchar3 valmin, uchar3 valmax){

    int x = (blockIdx.x * blockDim.x) + (threadIdx.x);
    int y = (blockIdx.y * blockDim.y) + (threadIdx.y);

    if(x < width && y < height){
        uchar3 myval = src.ptr(y)[x];
        uchar3 mydest = make_uchar3(255,255,255);

        if(myval.x < valmin.x || myval.x > valmax.x)
            mydest = make_uchar3(0,0,0);
        if(myval.y < valmin.y || myval.y > valmax.y)
            mydest = make_uchar3(0,0,0);
        if(myval.z < valmin.z || myval.z > valmax.z)
            mydest = make_uchar3(0,0,0);
    
        dst.ptr(y)[x] = mydest;
    }
}
