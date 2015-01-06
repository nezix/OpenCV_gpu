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

//To do : Asynchronous version

#include <iostream>
#include <opencv2/opencv_modules.hpp>

//if defined(HAVE_OPENCV_GPU)

#include <opencv2/opencv.hpp>
#include <opencv2/core/opengl_interop.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

#define RESIZEF 0.5

using namespace std;
using namespace cv;


#include "kernel.h"

//nvcc kernel.cu simple_kernelopencv.cu -o mygpu.ex `pkg-config --cflags --libs opencv` -arch=sm_13


uchar3 value_red[2] = {make_uchar3(170,110,90),make_uchar3(179,255,255)};


void usage(int argc,char **argv){
    if(argc != 2){
        cout << "Usage : "<<argv[0]<<" namefile"<<endl;
        exit(-1);
    }
}

int main(int argc, char** argv)
{
    usage(argc,argv);
    setUseOptimized(true);

    gpu::setGlDevice();

    //Get the video on GPU (gpu video reader)
    gpu::GpuMat frame;
    gpu::VideoReader_GPU d_reader(argv[1]);
    gpu::GpuMat resized;
    gpu::GpuMat hsv;
    int counter = 0;

    //Gpu matrix allocation
    gpu::GpuMat mask_red(0,0,CV_8U);

    while(true){
        //Read frame
        if (!d_reader.read(frame))
            break;

        //Resize the image on the GPU
        gpu::resize(frame, resized, Size(0,0),RESIZEF,RESIZEF);

        //Allocate GPU memory if necessary
        gpu::ensureSizeIsEnough(resized.rows,resized.cols,CV_8U,mask_red);  

        //Convert the RGB image in HSV values
        gpu::cvtColor(resized,hsv,CV_BGR2HSV);
        
        //Launch CUDA kernel
        mygpuinrange(hsv,mask_red,value_red[0],value_red[1]);

        //Convert back to RGB
        gpu::GpuMat maskrgb(mask_red);        
        gpu::cvtColor( mask_red, maskrgb, CV_GRAY2BGR );

        //Get image from GPU to CPU
        Mat resframe(maskrgb);

        imshow("GPU",resframe);
        counter++;
        if(waitKey(30)>0 )
            break;
    }

    return 0;
}

//#endif
