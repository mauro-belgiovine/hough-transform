// **********************************************************************************
//
// BSD License.
// This file is part of a Hough Transformation tutorial,
// see: http://www.keymolen.com/2013/05/hough-transformation-c-implementation.html
//
// Copyright (c) 2013, Bruno Keymolen, email: bruno.keymolen@gmail.com
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or other
// materials provided with the distribution.
// Neither the name of "Bruno Keymolen" nor the names of its contributors may be
// used to endorse or promote products derived from this software without specific
// prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
// IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
// NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// **********************************************************************************

#include "hough.h"
#include "simpleIndexing.cu"
#include <cmath>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>
// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper utility functions 


#define DEG2RAD 0.017453293f

#define SET_GRID_DIM(npoints, threadsPerBlock) ((npoints+(threadsPerBlock-1))/threadsPerBlock)
#define BLOCK_DIM 32

using namespace std;

/* CUDA functions definitions */

// CUDA timer macros
cudaEvent_t c_start, c_stop;

inline void start_time() {
    cudaEventCreate(&c_start);
    cudaEventCreate(&c_stop);
    cudaEventRecord(c_start, 0);
}

inline float stop_time(const char *msg) {
  float elapsedTime = 0;
  cudaEventRecord(c_stop, 0);
  cudaEventSynchronize(c_stop);
  cudaEventElapsedTime(&elapsedTime, c_start, c_stop);
  //if ( VERBOSE )
  printf("Time to %s: %.3f ms\n", msg, elapsedTime);
  cudaEventDestroy(c_start);
  cudaEventDestroy(c_stop);
  return elapsedTime;
}

//every CUDA Thread works processes a point of the input image
__global__ void CudaTransform(unsigned char* dev_img, unsigned int *dev_accu, int w, int h){
  
  //calculate params
  float hough_h = ((sqrt(2.0) * (float)(h>w?h:w)) / 2.0);
	  
  float center_x = w/2;
  float center_y = h/2;
  
  //calculate index which this thread have to process
  unsigned int index = getGlobalIdx_2D_2D();
  //calculate coordinates for corresponding index in entire image
  int x = index % w;
  int y = index / w;
  
  if( dev_img[index] > 250 ){ //se il punto è bianco (val in scala di grigio > 250)
    for(int t=0;t<180;t++){ //plot dello spazio dei parametri da 0° a 180° (sist. polare)
      
      float r = ( ((float)x - center_x) * cos((float)t * DEG2RAD)) + (((float)y - center_y) * sin((float)t * DEG2RAD));
      
      //dev_accu[ (int)((round(r + hough_h) * 180.0)) + t]++;
      atomicAdd(&(dev_accu[ (int)((round(r + hough_h) * 180.0)) + t]), 1);
      
    }
  }
  
  
}

namespace keymolen {

	Hough::Hough():_accu(0), _accu_w(0), _accu_h(0), _img_w(0), _img_h(0)
	{

	}

	Hough::~Hough() {
		if(_accu)
			free(_accu);
	}


	int Hough::Transform(unsigned char* img_data, int w, int h)
	{
	  
		
		_img_w = w;
		_img_h = h;

		//Create the accu
		double hough_h = ((sqrt(2.0) * (double)(h>w?h:w)) / 2.0);
		_accu_h = hough_h * 2.0; // -r -> +r
		_accu_w = 180;

		_accu = (unsigned int*)calloc(_accu_h * _accu_w, sizeof(unsigned int));

		double center_x = w/2;
		double center_y = h/2;

		start_time();

		for(int y=0;y<h;y++)
		{
			for(int x=0;x<w;x++)
			{
				if( img_data[ (y*w) + x] > 250 )
				{
					for(int t=0;t<180;t++)
					{
						double r = ( ((double)x - center_x) * cos((double)t * DEG2RAD)) + (((double)y - center_y) * sin((double)t * DEG2RAD));
						_accu[ (int)((round(r + hough_h) * 180.0)) + t]++; 
					}
				}
			}
		}
		stop_time("CPU Transform");
		return 0;
	}
	
	int Hough::Transform_GPU(unsigned char* img_data, int w, int h){
	  
	  
	  _img_w = w;
	  _img_h = h;

	  //Create the accu
	  double hough_h = ((sqrt(2.0) * (double)(h>w?h:w)) / 2.0);
	  _accu_h = hough_h * 2.0; // -r -> +r
	  _accu_w = 180;
	  _accu = (unsigned int*)calloc(_accu_h * _accu_w, sizeof(unsigned int));
	  
	  unsigned char *dev_img;
	  unsigned int *dev_accu;
	  
	  
	  
	  checkCudaErrors(cudaMalloc((void **) &dev_img, (sizeof(char)*w*h)));
	  checkCudaErrors(cudaMalloc((void **) &dev_accu, (sizeof(unsigned int) * _accu_w * _accu_h)));
	  checkCudaErrors(cudaMemset(dev_accu, 0, (sizeof(unsigned int) * _accu_w * _accu_h)));
	  
	  //copy data on device
	  checkCudaErrors(cudaMemcpy(dev_img, img_data, (sizeof(char)*w*h), cudaMemcpyHostToDevice));
	  
	  start_time();
	  //launch kernel
	  dim3 block(BLOCK_DIM, BLOCK_DIM);
	  dim3 grid(SET_GRID_DIM(w,BLOCK_DIM), SET_GRID_DIM(h,BLOCK_DIM));
	  
	  CudaTransform <<< grid, block >>> (dev_img, dev_accu, w, h);
	  stop_time("GPU Transform");
	  
	  //copy back results
	  checkCudaErrors(cudaMemcpy(_accu, dev_accu, (sizeof(unsigned int) * _accu_w * _accu_h), cudaMemcpyDeviceToHost));
	  
	  cudaFree(dev_img);
	  cudaFree(dev_accu);
	  return 0;
	}

	std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > > Hough::GetLines(int threshold)
	{
		std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > > lines;

		if(_accu == 0)
			return lines;

		for(int r=0;r<_accu_h;r++)
		{
			for(int t=0;t<_accu_w;t++)
			{
				if((int)_accu[(r*_accu_w) + t] >= threshold)
				{
					//Is this point a local maxima (9x9)
					int max = _accu[(r*_accu_w) + t];
					for(int ly=-4;ly<=4;ly++)
					{
						for(int lx=-4;lx<=4;lx++)
						{
							if( (ly+r>=0 && ly+r<_accu_h) && (lx+t>=0 && lx+t<_accu_w)  )
							{
								if( (int)_accu[( (r+ly)*_accu_w) + (t+lx)] > max )
								{
									max = _accu[( (r+ly)*_accu_w) + (t+lx)];
									ly = lx = 5;
								}
							}
						}
					}
					if(max > (int)_accu[(r*_accu_w) + t])
						continue;


					int x1, y1, x2, y2;
					x1 = y1 = x2 = y2 = 0;

					if(t >= 45 && t <= 135)
					{
						//y = (r - x cos(t)) / sin(t)
						x1 = 0;
						y1 = ((double)(r-(_accu_h/2)) - ((x1 - (_img_w/2) ) * cos(t * DEG2RAD))) / sin(t * DEG2RAD) + (_img_h / 2);
						x2 = _img_w - 0;
						y2 = ((double)(r-(_accu_h/2)) - ((x2 - (_img_w/2) ) * cos(t * DEG2RAD))) / sin(t * DEG2RAD) + (_img_h / 2);
					}
					else
					{
						//x = (r - y sin(t)) / cos(t);
						y1 = 0;
						x1 = ((double)(r-(_accu_h/2)) - ((y1 - (_img_h/2) ) * sin(t * DEG2RAD))) / cos(t * DEG2RAD) + (_img_w / 2);
						y2 = _img_h - 0;
						x2 = ((double)(r-(_accu_h/2)) - ((y2 - (_img_h/2) ) * sin(t * DEG2RAD))) / cos(t * DEG2RAD) + (_img_w / 2);
					}

					lines.push_back(std::pair< std::pair<int, int>, std::pair<int, int> >(std::pair<int, int>(x1,y1), std::pair<int, int>(x2,y2)));

				}
			}
		}

		std::cout << "lines: " << lines.size() << " " << threshold << std::endl;
		return lines;
	}

	const unsigned int* Hough::GetAccu(int *w, int *h)
	{
		*w = _accu_w;
		*h = _accu_h;

		return _accu;
	}
}
