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

//nvcc kernel.cu testgpu.cu -o mygpu.ex `pkg-config --cflags --libs opencv`


uchar3 value_red[2] = {make_uchar3(170,110,90),make_uchar3(179,255,255)};
uchar3 value_blue[2] = {make_uchar3(100,120,50),make_uchar3(130,255,255)};
uchar3 value_green[2] =  {make_uchar3(35,120,50),make_uchar3(80,255,255)};
uchar3 value_yellow[2] = {make_uchar3(20,120,70),make_uchar3(35,255,255)};
uchar3 value_white[2] = { make_uchar3(0,0,150),make_uchar3(179,30,255)};
uchar3 value_black[2] = {make_uchar3(0,0,0),make_uchar3(179,225,70)};


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

    //namedWindow("GPU",WINDOW_OPENGL);

    gpu::setGlDevice();

    Mat frame;
    VideoCapture capture(argv[1]);
    int counter = 0;

    if(!capture.isOpened()){
        cerr<<"Error while reading file";
        exit(-1);
    }
    while(true){
        capture>>frame;
        if(frame.empty())
            break;
        gpu::GpuMat gpuframe(frame);
        gpu::GpuMat resized;
        gpu::GpuMat hsv;
        gpu::resize(gpuframe, resized, Size(0,0),RESIZEF,RESIZEF);
        gpu::GpuMat mask_red(resized.rows,resized.cols,resized.type());
        
        //mask_red.setTo(Scalar::all(0));
        //if(counter>=50){
            
            gpu::cvtColor(resized,hsv,CV_BGR2HSV);
            
            mygpuinrange(resized.cols,resized.rows,gpu::PtrStep<uchar3>(resized),
                            gpu::PtrStep<uchar3>(mask_red),value_red[0],value_red[1]);

            Mat resframe(mask_red);
        //}

        imshow("GPU",resframe);
        counter++;
        if(waitKey(1)>0 )
            break;
    }

    return 0;
}

//#endif
