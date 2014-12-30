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

#define uchar unsigned char

void mygpuinrange(int width,int height,gpu::PtrStep<uchar3> src,gpu::PtrStep<uchar3> dst,uchar3 min, uchar3 max);
__global__ void GpuinRange(gpu::PtrStep<uchar3> src,gpu::PtrStep<uchar3> dst,int width,int height,uchar3 valmin, uchar3 valmax);
