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

#define HOUGHSPACE_DEG 180
#define HOUGHSPACE_STEP 10

#define SHARED_MEM_BANKS 32
#define SET_GRID_DIM(npoints, threadsPerBlock) ceil((npoints+((threadsPerBlock)-1))/(threadsPerBlock))
#define BLOCK_DIM 8
#define BLOCK_DIM_Y (SHARED_MEM_BANKS/BLOCK_DIM)

extern dim3 block;
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


/* getPixels(): kernel used to making an array of all pixels that need to be processed

NOTE: dovendo accedere allo stesso indirizzo di memoria __shared__, dobbiamo garantire un accesso
privo di bank-conflict, per evitare perdita di informazioni dovute all'accesso concorrente: 
per la memoria __shared__, ciò è garantito solo in un warp (32 thread) per device 2.x (16 thread per 1.x) che accede allo stesso bank (32 bit) di memoria
grazie alle primitive di "broadcast"

The fast case:
- If all threads of a half-warp access different banks, there is no bank conflict
- ** If all threads of a half-warp read the identical address, there is no bank conflict (broadcast) **
The slow case:
- Bank Conflict: multiple threads in the same half-warp access the same bank
- Must serialize the accesses
- Cost = max # of simultaneous accesses to a single bank

FROM -> http://on-demand.gputechconf.com/gtc-express/2011/presentations/NVIDIA_GPU_Computing_Webinars_CUDA_Memory_Optimization.pdf

TODO: provare a raddoppiare la dimensione del blocco (64) utilizzando 2 diverse __shared__ memory per contare i pixel,
in modo da garantire accesso a 1 warp alla volta ("broadcast") per evitare bank-conflict

PSEUDO CODE:
1 pixel_value = image[x,y]
2 if(pixel_value > threshold) {
3 	do {
4 		index++
5 		SMEM_index = index
6 		SMEM_array[index] = (x,y)
7 	} while(SMEM_array[index] != (x,y))
8 }
9 index = SMEM_index
*/

__global__ void getPixels(unsigned char* dev_img, unsigned int *dev_globalPixelArray, unsigned int *dev_globalPixelCount,  int w, int h, unsigned int *globalPixelCounter){
  
  //calculate index which this thread have to process
  unsigned int index = getGlobalIdx_2D_2D();
  unsigned int blockIndex = (threadIdx.y * blockDim.x) + threadIdx.x;
  unsigned int pixel_count = 0;
  __shared__ unsigned int sh_pixel_count;
  __shared__ unsigned int sh_pixel_array[BLOCK_DIM*BLOCK_DIM_Y];
  
  
  if(blockIndex == 0) sh_pixel_count = 0;
  
  
  //check index is in image bounds
  if(index < (w*h)){
    
    if( dev_img[index] > 250 ){ //se il punto è bianco (val in scala di grigio > 250)
      
      
      do{
	pixel_count++;
	sh_pixel_count = pixel_count;
	sh_pixel_array[pixel_count-1] = index;
	__syncthreads();
      }while(sh_pixel_array[pixel_count-1] != index );
    }
    
    pixel_count = sh_pixel_count;
  
    //save the list of interesting points
    unsigned int blockId = (blockIdx.y * gridDim.x) + blockIdx.x;
    
    //First one thread in each thread block
    if((threadIdx.x == 0) && (threadIdx.y == 0)){
      //add the sum of all pixels collected in this thread-block
      dev_globalPixelCount[blockId] = pixel_count;    
      atomicAdd(globalPixelCounter, pixel_count);
      
      //copy in the global array each pixel to be processed
      for(unsigned int i = 0; i < pixel_count; i++){
	dev_globalPixelArray[index+i] = sh_pixel_array[i];
      }
    }
  }
}

__global__ void voteHough(unsigned char* dev_img, unsigned int *dev_accu, unsigned int *dev_globalPixelArray, int w, int h, unsigned int *pixelCount){
  
  //block-shared Hough accumulator: each block fills HOUGHSPACE_STEP degrees (of 180) of the Hough space
  extern __shared__ int block_accum[];
  
  unsigned int index_accum;
  //calculate indexes: we are interested in blockIDx.y == 0 for the image pixel-index
  unsigned int index = getGlobalIdx_1D_2D();
  //this is the local (block) thread index
  unsigned int blockIndex = (threadIdx.y * blockDim.x) + threadIdx.x;
  
  
  if(index < *pixelCount)  {
    
    //calculate params
    float hough_h = ((sqrt(2.0) * (float)(h>w?h:w)) / 2.0);
    
    unsigned int sh_mem_size = (hough_h*2* HOUGHSPACE_STEP );
    unsigned int sh_thread_portion = sh_mem_size /(blockDim.x * blockDim.y);
    //each thread initializes a portion of the shared memory
    for(unsigned int i = 0; i < sh_thread_portion; i++){
      block_accum[(blockIndex*sh_thread_portion)+i] = 0;
    }
    //last thread has to init the remaining sh. memory space
    if(blockIndex == ((blockDim.x * blockDim.y)-1)){
      for(unsigned int i = (blockIndex*sh_thread_portion)+(sh_thread_portion-1); i < sh_mem_size; i++){
	block_accum[i] = 0;
      }
    }
    
    __syncthreads();
    
    unsigned int indexToProcess = dev_globalPixelArray[index];	//get the point-index that this thread has to process
    
    if(indexToProcess != 0){	//if it has to be processed
      
      float center_x = w/2;
      float center_y = h/2;
	
      //calculate coordinates for corresponding index in entire image
      int x = indexToProcess % w;
      int y = indexToProcess / w;
            
      unsigned int start_deg = blockIdx.y * HOUGHSPACE_STEP;
      unsigned int stop_deg = start_deg + (HOUGHSPACE_STEP);
      
      for(int t = start_deg; t < stop_deg; t++){ //plot dello spazio dei parametri da 0° a 180° (sist. polare)
	  
	  float r = ( ((float)x - center_x) * cos((float)t * DEG2RAD)) + (((float)y - center_y) * sin((float)t * DEG2RAD));
	  //index_accum = (int)((round(r + hough_h) * HOUGHSPACE_DEG)) + t;
	  //atomicAdd(&(dev_accu[index_accum]), 1);
	  
	  //TODO: check 2 linee qui sotto, dovrebbero essere ok
	  index_accum = (int)((round(r + hough_h) * HOUGHSPACE_STEP)) + t;
	  atomicAdd(&(block_accum[index_accum]), 1);
      }
      
      unsigned int row_counter = 0;
      unsigned int global_index = 0;
      //TODO: riportare correttamente nell'array globale (evitare operazioni atomiche)
      
      if(blockIndex == 0){
	for(unsigned int i = 0; i < sh_mem_size; i++){
	  
	  global_index = (blockIdx.y * HOUGHSPACE_STEP ) + //proper column in hough space (eg. 0 - 10°)
			 (row_counter * HOUGHSPACE_DEG ) + //proper row in hough space
			 (i % HOUGHSPACE_STEP) ;	   //index relative to block-hough space
	  
	  atomicAdd(&(dev_accu[global_index]), block_accum[i]);
	  //dev_accu[global_index] += block_accum[i];
	  
	  if((i % (HOUGHSPACE_STEP)) == (HOUGHSPACE_STEP - 1)) row_counter++;
	  
	}
      }
      
    }
    
  }
}


//every CUDA Thread works processes a point of the input image
__global__ void CudaTransform(unsigned char* dev_img, unsigned int *dev_accu, int w, int h){
  
  
  //calculate index which this thread have to process
  unsigned int index = getGlobalIdx_2D_2D();
  
  //check index is in image bounds
  if(index < (w*h)){
    //calculate params
    float hough_h = ((sqrt(2.0) * (float)(h>w?h:w)) / 2.0);
	    
    float center_x = w/2;
    float center_y = h/2;
      
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
		
		unsigned int total_processed_pixels = 0;

		for(int y=0;y<h;y++)
		{
			for(int x=0;x<w;x++)
			{
				if( img_data[ (y*w) + x] > 250 )
				{
				  total_processed_pixels++;
					for(int t=0;t<180;t++)
					{
						double r = ( ((double)x - center_x) * cos((double)t * DEG2RAD)) + (((double)y - center_y) * sin((double)t * DEG2RAD));
						_accu[ (int)((round(r + hough_h) * 180.0)) + t]++; 
						//if((total_processed_pixels < 10) && (t < 10)) cout << ((round(r + hough_h) * 180.0)) + t << " ";
					}
				}
			}
		}
		
		cout << "Total processed pixels " << total_processed_pixels << endl;
		
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
	  
	  
	  //launch kernel
	  dim3 block(BLOCK_DIM, 4);
	  dim3 grid(SET_GRID_DIM(w,BLOCK_DIM), SET_GRID_DIM(h,4));
	  start_time();
	  CudaTransform <<< grid, block >>> (dev_img, dev_accu, w, h);
	  stop_time("GPU Transform");
	  
	  //copy back results
	  checkCudaErrors(cudaMemcpy(_accu, dev_accu, (sizeof(unsigned int) * _accu_w * _accu_h), cudaMemcpyDeviceToHost));
	  
	  cudaFree(dev_img);
	  cudaFree(dev_accu);
	  return 0;
	}
	
	int Hough::Transform_GPUFast(unsigned char* img_data, int w, int h){
	  
	  _img_w = w;
	  _img_h = h;

	  //Create the accu
	  double hough_h = ((sqrt(2.0) * (double)(h>w?h:w)) / 2.0);
	  _accu_h = hough_h * 2.0; // -r -> +r
	  _accu_w = 180;
	  _accu = (unsigned int*)calloc(_accu_h * _accu_w, sizeof(unsigned int));
	  
	  unsigned char *dev_img;
	  unsigned int *dev_globalPixelArray;	//it will contain only pixels that have to be processed
	  unsigned int *dev_globalPixelCount;	//it will hold number of pixels that have to be processed per each thread-BLOCK
	  unsigned int *dev_accu;
	  
	  unsigned int *dev_pixelCount;
	  
	  checkCudaErrors(cudaMalloc((void **) &dev_img, (sizeof(char) * w * h)));
	  //copy data on device
	  checkCudaErrors(cudaMemcpy(dev_img, img_data, (sizeof(char)*w*h), cudaMemcpyHostToDevice));
	  
	  checkCudaErrors(cudaMalloc((void **) &dev_globalPixelArray, (sizeof(unsigned int) * w * h)));
	  checkCudaErrors(cudaMalloc((void **) &dev_globalPixelCount, (sizeof(unsigned int) * SET_GRID_DIM(w,BLOCK_DIM) * SET_GRID_DIM(h,BLOCK_DIM_Y))));
	  checkCudaErrors(cudaMalloc((void **) &dev_pixelCount, sizeof(unsigned int)));
	  
	  checkCudaErrors(cudaMemset(dev_globalPixelCount, 0 , (sizeof(unsigned int) * SET_GRID_DIM(w,BLOCK_DIM) * SET_GRID_DIM(h,BLOCK_DIM_Y))));
	  checkCudaErrors(cudaMemset(dev_globalPixelArray, 0 , (sizeof(unsigned int) * w * h)));
	  checkCudaErrors(cudaMemset(dev_pixelCount, 0 , sizeof(unsigned int)));

	  checkCudaErrors(cudaMalloc((void **) &dev_accu, (sizeof(unsigned int) * _accu_w * _accu_h)));
	  
	  
	  //TODO:con l'uso della memoria shared, il prossimo passaggio si può saltare
	  checkCudaErrors(cudaMemset(dev_accu, 0, (sizeof(unsigned int) * _accu_w * _accu_h)));
	  
	  //BEGIN Hough Transform on GPU
	  
	  dim3 block_getPixels(BLOCK_DIM, BLOCK_DIM_Y);
	  dim3 grid_getPixels(SET_GRID_DIM(w,BLOCK_DIM), SET_GRID_DIM(h,BLOCK_DIM_Y));
	  start_time();
	  getPixels <<<grid_getPixels, block_getPixels>>> (dev_img, dev_globalPixelArray, dev_globalPixelCount, w, h, dev_pixelCount);
	  stop_time("getPixels");
	  
	  start_time();
	  unsigned int *PixelArray = (unsigned int *) malloc(sizeof(unsigned int) * w * h);
	  unsigned int *PixelCount = (unsigned int *) malloc(sizeof(unsigned int) * SET_GRID_DIM(w,BLOCK_DIM) * SET_GRID_DIM(h,BLOCK_DIM_Y));
	  unsigned int pixelCount = 0;
	  checkCudaErrors(cudaMemcpy(PixelCount, dev_globalPixelCount, (sizeof(unsigned int) * SET_GRID_DIM(w,BLOCK_DIM) * SET_GRID_DIM(h,BLOCK_DIM_Y)), cudaMemcpyDeviceToHost));
	  checkCudaErrors(cudaMemcpy(PixelArray, dev_globalPixelArray, (sizeof(unsigned int) * w * h), cudaMemcpyDeviceToHost));
	  checkCudaErrors(cudaMemcpy(&pixelCount, dev_pixelCount, (sizeof(unsigned int)), cudaMemcpyDeviceToHost));
	  
	  //Realloc pixel array upon number of pixels that have been collected
	  
	  unsigned int *new_pixelArray = (unsigned int *) malloc(sizeof(unsigned int) * pixelCount);
	  cudaFree(dev_globalPixelArray);
	  checkCudaErrors(cudaMalloc((void **) &dev_globalPixelArray, (sizeof(unsigned int) * pixelCount)));
	  
	  unsigned int value = 0;
	  unsigned int blockID = 0;
	  unsigned int new_arrayIndex = 0;
	  
	  for(unsigned int i = 0; i < (w*h); i += (BLOCK_DIM * BLOCK_DIM_Y)){
	    blockID = i / (BLOCK_DIM * BLOCK_DIM_Y);
	    if(PixelCount[blockID] != 0){
	      for(unsigned int y = 0; y < PixelCount[blockID]; y++){
		value = PixelArray[i+y];
		new_pixelArray[new_arrayIndex++] = value;
	      }
	    }
	  }
	  
	  checkCudaErrors(cudaMemcpy(dev_globalPixelArray, new_pixelArray, (sizeof(unsigned int) * pixelCount), cudaMemcpyHostToDevice));
	  stop_time("realloc pixel array");
	  
	  dim3 block_voteHough(BLOCK_DIM*4, BLOCK_DIM*4);
	  dim3 grid_voteHough(SET_GRID_DIM(pixelCount, BLOCK_DIM*4 * BLOCK_DIM*4), (HOUGHSPACE_DEG/HOUGHSPACE_STEP));
	  size_t shared_mem_size = sizeof(int) * _accu_h * HOUGHSPACE_STEP;
	  
	  start_time();
	  voteHough <<<grid_voteHough, block_voteHough, shared_mem_size>>> (dev_img, dev_accu, dev_globalPixelArray, w, h, dev_pixelCount);
	  stop_time("voteHough");
	  
	  
	  //END Hough Transform on GPU
	  
	  //copy back results
	  checkCudaErrors(cudaMemcpy(_accu, dev_accu, (sizeof(unsigned int) * _accu_w * _accu_h), cudaMemcpyDeviceToHost));

	  free(PixelCount);
	  free(PixelArray);
	  cudaFree(dev_img);
	  cudaFree(dev_accu);
	  cudaFree(dev_globalPixelCount);
	  cudaFree(dev_globalPixelArray);
	  
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

		std::cout << "lines: " << lines.size() << " " << threshold << "; img dim: w=" << _img_w << " h=" << _img_h << std::endl;
		return lines;
	}

	const unsigned int* Hough::GetAccu(int *w, int *h)
	{
		*w = _accu_w;
		*h = _accu_h;

		return _accu;
	}
}
