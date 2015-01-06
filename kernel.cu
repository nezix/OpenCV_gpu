/* <Written by Xavier Martinez : Shows how to manage CUDA kernels without using cudaMalloc >
    Copyright (C) <2014>  <Xavier Martinez>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

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


void mygpuinrange(gpu::GpuMat &src,gpu::GpuMat &dst,uchar3 min, uchar3 max){
    dim3 threadsPerBlock(BLOCKSIZE,BLOCKSIZE);
    int nbblocks1 = (int)ceil((float)src.cols/threadsPerBlock.x);
    int nbblocks2 = (int)ceil((float)src.rows/threadsPerBlock.y);
    dim3 numblocks(nbblocks1,nbblocks2);

	GpuinRange<<<threadsPerBlock,numblocks>>>(gpu::PtrStep<uchar3>(src),gpu::PtrStep<uchar>(dst),src.cols,src.rows,min,max);
}

__global__ void GpuinRange(gpu::PtrStep<uchar3> src,gpu::PtrStep<uchar> dst,int width,int height,uchar3 valmin, uchar3 valmax){

    int x = (blockIdx.x * blockDim.x) + (threadIdx.x);
    int y = (blockIdx.y * blockDim.y) + (threadIdx.y);

    if(x < width && y < height){
        uchar3 myval = src.ptr(y)[x];
        uchar mydest = (uchar)255;//max value
        
        //Out of range min/max
        if(myval.x < valmin.x || myval.x > valmax.x)
            mydest = (uchar)0;
        if(myval.y < valmin.y || myval.y > valmax.y)
            mydest = (uchar)0;
        if(myval.z < valmin.z || myval.z > valmax.z)
            mydest = (uchar)0;
    
        dst.ptr(y)[x] = mydest;
    }
}
