//
//  ht_helix.cpp
//  
//
//  Created by Lorenzo Rinaldi on 29/04/14.
//
//
// compile:
// nvcc -I/usr/local/cuda-5.5/samples/common/inc -I/usr/local/cuda-5.5/targets/x86_64-linux/include -gencode arch=compute_20,code=sm_21 -o ht_rhophi ht_rhophi.cu

#include <cuda_runtime.h>
// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper utility functions 

#include "simpleIndexing.cu"

#include <string.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unistd.h>

using namespace std;

#define NHMAX 300
#define Nsec 4 // Numero settori in piano trasverso
#define Ntheta 16 // Numero settori in piano longitudinale
#define Nphi 1024 // Numero bin angolo polare
#define Nrho 1024 // Numero bin distanza radiale

#define rhomin 500.f // mm
#define rhomax 100000.f // mm
#define phimin 0.f // rad
#define phimax 2*M_PI // rad
#define thetamin 0.f // rad
#define thetamax M_PI // rad

#define ac_soglia 4 // soglia nella matrice di accumulazione

/* --- DEFINE TO ALTER EXECUTION --- */
//#define PARALLEL_REDUX_MAX
//#define VERBOSE_DUMP
#define CUDA_MALLOCHOST_OUTPUT
#define CUDA_MANAGED_TRANSFER

#define max_tracks_out 100

int acc_Mat [ Nsec ][ Ntheta ][Nphi ] [Nrho ];
//int Max_rel [ Nsec ][ Ntheta ][Nphi ] [Nrho ];
int debug_accMat[ Nsec ][ Ntheta ][ Nphi ] [ Nrho ];

float dtheta= M_PI/Ntheta;
float drho= (rhomax-rhomin)/Nrho;
float dphi= (phimax-phimin)/Nphi;

vector<float> x_values;
vector<float> y_values;
vector<float> z_values;

#ifndef PARALLEL_REDUX_MAX

struct track_param{
      int acc;
      /*unsigned int isec;
      unsigned int ith;
      unsigned int iphi;
      unsigned int irho;*/
    };
    
#ifndef CUDA_MALLOCHOST_OUTPUT
struct track_param host_out_tracks[ Nsec * Ntheta * Nphi * Nrho ];
#endif

#endif

//lock definition
#ifndef __LOCK_H__
#define __LOCK_H__

struct Lock {
    int *mutex;
    Lock( void ) {
         cudaMalloc( (void**)&mutex, sizeof(int) ) ;
         cudaMemset( mutex, 0, sizeof(int) );
    }

    ~Lock( void ) {
        cudaFree( mutex );
    }

    __device__ void lock( void ) {
        while( atomicCAS( mutex, 0, 1 ) != 0 );
    }

    __device__ void unlock( void ) {
        atomicExch( mutex, 0 );
    }
};

#endif
//end lock


void read_inputFile(string file_path, unsigned int num_hits);

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
  //printf("Time to %s: %.3f ms\n", msg, elapsedTime);
  cudaEventDestroy(c_start);
  cudaEventDestroy(c_stop);
  return elapsedTime;
}

__global__ void voteHoughSpace(float *dev_x_values, float *dev_y_values, float *dev_z_values, int *dev_accMat, float dtheta, float drho, float dphi){
  
  __shared__ float x_val;
  __shared__ float y_val;
  __shared__ float z_val;
   
  if(threadIdx.x == 0){
    x_val = dev_x_values[blockIdx.x];
    y_val = dev_y_values[blockIdx.x];
    z_val = dev_z_values[blockIdx.x];
  }
  
  __syncthreads();
  
  float R2 = x_val*x_val + y_val*y_val;
  float theta=acos(z_val/sqrt(R2+z_val*z_val));
  
  int ith=(int) (theta/dtheta)+0.5f;
  
  float sec=atan2(y_val,x_val);
  if (sec<0.f)
  {
    sec=2*M_PI+sec;
  }
  int isec=int(sec/2/M_PI*Nsec);
  
  int iphi = threadIdx.x;
  float phi=phimin+iphi*dphi;
  float rho=R2/2.f/(x_val*cos(phi)+y_val*sin(phi));
  int irho=(int)((rho-rhomin)/drho)+0.5f;
  
  int accu_index = (isec*(Ntheta*Nphi*Nrho))+((ith*Nphi*Nrho) +((iphi*Nrho)+irho));
  
  if (rho<=rhomax && rho>rhomin)
  {
    atomicAdd(&(dev_accMat[accu_index]),1);
  }
}

#ifndef PARALLEL_REDUX_MAX

#define get4DIndex(x,y,z,r) ((x)*(Ntheta*Nphi*Nrho))+(((y)*Nphi*Nrho) +(((z)*Nrho)+(r)))

__global__ void findRelativeMax(int *dev_accMat, struct track_param *dev_output, Lock mutex, unsigned int *NMrel){
  
  
  unsigned int isec = blockIdx.x;
  unsigned int ith = blockIdx.y / (Nrho/blockDim.y);
  unsigned int iphi = threadIdx.x;
  unsigned int irho = blockIdx.y % (Nrho/blockDim.y);
  
  unsigned int globalIndex = getGlobalIdx_2D_2D();
  unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
  
  /*__shared__ unsigned int local_NMrel;
  
  if(threadIdx.x == 0) local_NMrel = 0;
  __syncthreads();*/
  
  //check if it is a local maxima by verifying that it is greater then (>=) its neighboors
  
  //we must check from isec >= 0, ith >= 1, iphi >= 1, irho >= 1
  if((ith > 0) && (iphi > 0) && (irho > 0)){
    
    //each thread is assigned to one point of the accum. matrix:
    int acc= dev_accMat[get4DIndex(isec, ith, iphi, irho)];
    
    if (acc >= ac_soglia){
      
      if(acc > dev_accMat[get4DIndex(isec, ith,iphi, irho-1)] && acc >= dev_accMat[get4DIndex(isec, ith, iphi, irho+1)]){
	
	if(acc > dev_accMat[get4DIndex(isec, ith, iphi-1, irho)] && acc >= dev_accMat[get4DIndex(isec, ith, iphi+1, irho)]){
      
		/*atomicAdd(&local_NMrel, 1);
		
		if(threadIdx.x == 0){
		  mutex.lock();
		  *NMrel += local_NMrel;
		  mutex.unlock();
		}*/
		atomicAdd(NMrel, 1);
		
		//mutex.lock();
		dev_output[globalIndex].acc = acc;
		/*dev_output[globalIndex].isec = isec;
		dev_output[globalIndex].ith = ith;
		dev_output[globalIndex].iphi = iphi;
		dev_output[globalIndex].irho = irho;*/
		//mutex.unlock();
	}

      }
    }
    
    
  }               
}

#else

//NOTE: wrong approach to solve this problem
//TODO: improve as on slides
__global__ void reduceParallelMax(int *dev_accMat, int *dev_output, int *dev_maxRelOutput, unsigned int N){
  
  
  extern __shared__ int sdata[];
  
  int* max_sdata = (int *) sdata;
  int* relMax_sdata = (int *) &sdata[blockDim.x];
  
  //each thread loads one element from global to shared mem
  unsigned int tid = threadIdx.x; //local index
  //unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; //global index (1D grid - 1D block)
  unsigned int i = getGlobalIdx_2D_1D();
  
  if(i < N){ //check if thread is in data bounds
  
    max_sdata[tid] = dev_accMat[i];
    relMax_sdata[tid] = dev_accMat[i];
    __syncthreads();
    
    //do reduction in shared memory
    for(unsigned int s=1; s < blockDim.x; s*=2){
      if(tid % (2*s) == 0){ //it is for a different stride
	//atomicMax(&(max_sdata[tid]),max_sdata[tid+s]); //TODO: change without atomic
	max_sdata[tid] = (max_sdata[tid] > max_sdata[tid+s]) ? max_sdata[tid] : max_sdata[tid+s];
	__syncthreads();
      }
      __syncthreads();
      
    }
    
    //write results (now found in the first element of the array) for this block to global memory 
    //if(tid == 0) dev_output[blockIdx.x] = sdata[0];
    
    if(tid == 0) dev_output[blockIdx.x] = max_sdata[0]; //at sdata[0], we found the maximum
    
    if(relMax_sdata[tid] >= ac_soglia){ 
      dev_maxRelOutput[i] = relMax_sdata[tid];
    }else{
      dev_maxRelOutput[i] = 0;
    }

  }
}
#endif

void help(char* prog) {

  printf("Use %s [-l #loops] [-n #hitsToRead] [-h] \n\n", prog);
  printf("  -l loops        Number of executions (Default: 1).\n");
  printf("  -n hits         Number of hits to read from input file (Default: 236).\n");
  printf("  -h              This help.\n");

}

int main(int argc, char* argv[]){
  
  
    unsigned int N_LOOPS = 1;
    unsigned int N_HITS = 236;
    int c;
    
    //getting command line options
    while ( (c = getopt(argc, argv, "l:n:h")) != -1 ) {
      switch(c) {
	
	case 'n':
	  N_HITS = atoi(optarg);
	  break;
	  
	case 'l':
	  N_LOOPS = atoi(optarg);
	  break;
	case 'h':
	  help(argv[0]);
	  return 0;
	  break;
	default:
	  printf("Unkown option!\n");
	  help(argv[0]);
	  return 0;
      }
    }
    
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    
#ifdef CUDA_MALLOCHOST_OUTPUT
      struct track_param *host_out_tracks;
      start_time();
      checkCudaErrors(cudaMallocHost((void **) &host_out_tracks, (sizeof(struct track_param)*(Nsec * Ntheta * Nphi * Nrho))));
      float init_outputMatrix = stop_time("init output matrix with cudaMallocHost");
      cout << "time to init output matrix (once): " << init_outputMatrix << endl;
#endif
  
    int *dev_accMat;
    float *dev_x_values;
    float *dev_y_values;
    float *dev_z_values;
    
    float *x_values_temp;
    float *y_values_temp;
    float *z_values_temp;
    
    //executions loop
    for(unsigned int loop = 0; loop < N_LOOPS; loop++){
      
      float timing[5];
      //float R = 0.f;
            
      // Inizializzo a zero le matrici
      memset(&acc_Mat, 0, (sizeof(int)*(Nsec*Ntheta*Nphi*Nrho)) );
      memset(&debug_accMat, 0, (sizeof(int)*(Nsec*Ntheta*Nphi*Nrho)) );
      //memset(&Max_rel, 0, (sizeof(int)*(Nsec*Ntheta*Nphi*Nrho)) );
      
      //alloc accumulator matrix on GPU
      start_time();
      checkCudaErrors(cudaMalloc((void **) &dev_accMat, (sizeof(int)* (Nsec * Ntheta * Nphi * Nrho)) ));
      checkCudaErrors(cudaMemset(dev_accMat, 0, (sizeof(int)*(Nsec*Ntheta*Nphi*Nrho))));
      timing[1] = stop_time("malloc dev_accMat and memset(0)");
      
      //riempi i valori dentro x_values , y_values , z_values
      read_inputFile("hits-5000.txt", N_HITS);
  //    read_inputFile("../datafiles/hits-1.txt");
      
      x_values_temp = (float*) malloc(sizeof(float)*x_values.size());
      y_values_temp =  (float*) malloc(sizeof(float)*y_values.size());
      z_values_temp = (float*)  malloc( sizeof(float)*z_values.size());
      
      for(unsigned int i = 0; i < x_values.size(); i++){
	x_values_temp[i] = x_values.at(i);
	y_values_temp[i] = y_values.at(i);
	z_values_temp[i] = z_values.at(i);
      }
      
      start_time();
      checkCudaErrors(cudaMalloc((void **) &dev_x_values, sizeof(float)*x_values.size()));
      checkCudaErrors(cudaMalloc((void **) &dev_y_values, sizeof(float)*y_values.size()));
      checkCudaErrors(cudaMalloc((void **) &dev_z_values, sizeof(float)*z_values.size()));
      checkCudaErrors(cudaMemcpy(dev_x_values, x_values_temp, sizeof(float)*x_values.size(), cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(dev_y_values, y_values_temp, sizeof(float)*y_values.size(), cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(dev_z_values, z_values_temp, sizeof(float)*z_values.size(), cudaMemcpyHostToDevice));
      timing[0] = stop_time("Input malloc and copy HtoD");
      
      start_time();
      voteHoughSpace <<<x_values.size(), Nphi>>> (dev_x_values, dev_y_values, dev_z_values, dev_accMat, dtheta, drho, dphi); //assumes that Nphi == Nrho
      timing[2] = stop_time("Vote");
#ifdef VERBOSE_DUMP     
      checkCudaErrors(cudaMemcpy((void *) &debug_accMat, dev_accMat, (sizeof(int)*(Nsec*Ntheta*Nphi*Nrho)), cudaMemcpyDeviceToHost));
#endif
      for(unsigned int i = 0; i < x_values.size(); i++){
	  //cout << x_values.at(i) << " - ";
	  //cout << y_values.at(i) << endl;
	  
	  float R2=x_values.at(i)*x_values.at(i)+y_values.at(i)*y_values.at(i);
	  float theta=acos(z_values.at(i)/sqrt(R2+z_values.at(i)*z_values.at(i)));
	  int ith=(int) (theta/dtheta)+0.5f;
	  
	  float sec=atan2(y_values.at(i),x_values.at(i));
	  if (sec<0.f)
	  {
	      sec=2*M_PI+sec;
	  }
	  int isec=int(sec/2/M_PI*Nsec);
	  
	  for(int iphi = 0; iphi < Nphi; iphi++){
	      float phi=phimin+iphi*dphi;
	      float rho=R2/2.f/(x_values.at(i)*cos(phi)+y_values.at(i)*sin(phi));
	      int irho=(int)((rho-rhomin)/drho)+0.5f;
	      if (rho<=rhomax && rho>rhomin)
	      {
		  acc_Mat[isec][ith][iphi][irho]++;
	      }
	  }
      }
      
#ifdef VERBOSE_DUMP
      //check
      unsigned int corretto = 0;
      unsigned int errore = 0;
      unsigned int letto = 0;
      for(unsigned int isec = 0; isec < Nsec; isec++){
	  
	  for(unsigned int ith = 0; ith < Ntheta; ith++){
	      
	      for(unsigned int iphi = 0; iphi < Nphi; iphi++){
		  
		  for(unsigned int irho = 0; irho < Nrho; irho++){
		    
		    if(acc_Mat[isec][ith][iphi][irho] != debug_accMat[isec][ith][iphi][irho]){
		    printf("diverso acc_Mat[%d][%d][%d][%d] %d - debug_accMat[%d][%d][%d][%d] %d \n", isec, ith, iphi, irho, acc_Mat[isec][ith][iphi][irho],
		      isec, ith, iphi, irho, debug_accMat[isec][ith][iphi][irho]);
		      errore++;
		    }else corretto++;
		    letto++;
		  }
	      }
	  }
      }
      printf("corretti %d sbaglati %d; letti %d\n", corretto, errore, letto);
#endif
      
      checkCudaErrors(cudaFree(dev_x_values));
      checkCudaErrors(cudaFree(dev_y_values));
      checkCudaErrors(cudaFree(dev_z_values));
      
      free(x_values_temp);
      free(y_values_temp);
      free(z_values_temp);
      
      x_values.clear();
      y_values.clear();
      z_values.clear();
      
      //trova il massimo relativo
      unsigned int host_NMrel = 0;
      
      // --- Prendiamo le informazioni specifiche della GPU per la divisione del lavoro appropriata
      unsigned int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;

#ifndef PARALLEL_REDUX_MAX
      
      struct track_param *dev_indexOutput;
      Lock my_lock;
      
      unsigned int *NMrel;
      
      start_time();   
      checkCudaErrors(cudaMalloc((void **) &NMrel, (sizeof(unsigned int))));
      checkCudaErrors(cudaMemset(NMrel, 0, sizeof(unsigned int)));
      
#ifdef CUDA_MANAGED_TRANSFER
      int cudaVer = 0;
      cudaRuntimeGetVersion(&cudaVer);
      if(cudaVer >= 6000){
	checkCudaErrors(cudaMallocManaged(&dev_indexOutput,(sizeof(struct track_param)* (Nsec * Ntheta * Nphi * Nrho)) ));
      }else{
#endif
	
      checkCudaErrors(cudaMalloc((void **) &dev_indexOutput, (sizeof(struct track_param)* (Nsec * Ntheta * Nphi * Nrho)) ));
      
#ifdef CUDA_MANAGED_TRANSFER
      }
#endif
      
      checkCudaErrors(cudaMemset(dev_indexOutput, -1, (sizeof(struct track_param)* (Nsec * Ntheta * Nphi * Nrho))));
      
      timing[1] += stop_time("malloc dev_indexOutput+NMrel and memset");
      
      // dividiamo adeguatamente il lavoro
      // in base al numero massimo di thread disponibili in un singolo thread-block
      unsigned int dim_x_block = Nphi;
      unsigned int dim_y_block = maxThreadsPerBlock/dim_x_block;
      unsigned int dim_x_grid = Nsec;
      unsigned int dim_y_grid = Ntheta * (Nrho/dim_y_block);
      
      dim3 grid(dim_x_grid, dim_y_grid);
      dim3 block(dim_x_block, dim_y_block);
      
      start_time();
      findRelativeMax <<<grid, block>>> (dev_accMat, dev_indexOutput, my_lock, NMrel);
      timing[3] = stop_time("Max. Relative");
      
      start_time();
#ifdef CUDA_MALLOCHOST_OUTPUT
      checkCudaErrors(cudaMemcpy((void *) host_out_tracks, dev_indexOutput, (sizeof(int)* (Nsec * Ntheta * Nphi * Nrho)), cudaMemcpyDeviceToHost));
#else
      checkCudaErrors(cudaMemcpy((void *) &host_out_tracks, dev_indexOutput, (sizeof(int)* (Nsec * Ntheta * Nphi * Nrho)), cudaMemcpyDeviceToHost));
#endif
      checkCudaErrors(cudaMemcpy((void *) &host_NMrel, NMrel, (sizeof(int)), cudaMemcpyDeviceToHost));
      timing[4] = stop_time("Copy results DtoH");

#ifdef VERBOSE_DUMP
      cout << "NMrel from GPU "<< host_NMrel << endl;

      unsigned int ntracks = 0;
      
      for(unsigned int i = 0; ((i < (Nsec * Ntheta * Nphi * Nrho)) && (ntracks < host_NMrel)); i++){
	
	if(host_out_tracks[i].acc > -1){
	  cout << "track " << ntracks << " acc value = " << host_out_tracks[i].acc << " [" << i << "]" << endl;
	  ntracks++;    
	}
      }
#endif
      //free mem
      checkCudaErrors(cudaFree(dev_indexOutput));
      checkCudaErrors(cudaFree(NMrel));

      //print timing results with this format:
      // NHIT HtoD_input MEMSET_cumulative VOTE MAX_REL DtoH_output
      cout << N_HITS << " " << timing[0] << " " << timing[1] << " " << timing[2] << " " << timing[3] << " " << timing[4] << endl; 
      
      
#else
      
#define SET_GRID_DIM(npoints, threadsPerBlock) ceil((npoints+((threadsPerBlock)-1))/(threadsPerBlock))
      
      unsigned int half_grid = SET_GRID_DIM((Nsec*Ntheta*Nphi*Nrho), maxThreadsPerBlock)/2;
      
      dim3 grid(half_grid, 2);
      
      unsigned int n_blocks = half_grid * 2;
      
      int * dev_maxBlockOutput;
      checkCudaErrors(cudaMalloc((void **) &dev_maxBlockOutput, (sizeof(int) * n_blocks)));
      int * dev_maxRelOutput;
      checkCudaErrors(cudaMalloc((void **) &dev_maxRelOutput, (sizeof(int) * (Nsec*Ntheta*Nphi*Nrho))));
      
      reduceParallelMax<<<grid, maxThreadsPerBlock, 2*(maxThreadsPerBlock*sizeof(int))>>>(dev_accMat, dev_maxBlockOutput, dev_maxRelOutput, (Nsec*Ntheta*Nphi*Nrho));
      
      int *host_maxBlockOutput = (int *) malloc((sizeof(int)* n_blocks));
      checkCudaErrors(cudaMemcpy(host_maxBlockOutput, dev_maxBlockOutput, (sizeof(int) * n_blocks), cudaMemcpyDeviceToHost));
      
      int *host_maxRelOutput = (int *) malloc((sizeof(int)* (Nsec*Ntheta*Nphi*Nrho)));
      checkCudaErrors(cudaMemcpy(host_maxRelOutput, dev_maxRelOutput, (sizeof(int) * (Nsec*Ntheta*Nphi*Nrho)), cudaMemcpyDeviceToHost));
      
      unsigned int debug = 0;
      
      for(unsigned int i = 0; i < n_blocks; i++){
	
	if(host_maxBlockOutput[i] != 0){
	  cout << "block " << i << " max: " << host_maxBlockOutput[i] << " [" << i*maxThreadsPerBlock << "]" << endl;
	  host_NMrel++;
	}
	
	unsigned int found = 0;
	
	for(unsigned int y = 0; y < maxThreadsPerBlock; y++){
	  unsigned int globalIndex = (y+(i*maxThreadsPerBlock));
	  if((host_maxRelOutput[globalIndex] != 0)) {
	    cout << "out["<< globalIndex << "]="<< host_maxRelOutput[globalIndex]<< " ";
	    found++; debug++;
	  }
	}
	if(found > 0) cout << " (block "<< i << ")" << endl << endl;
	
      }
      
      
      /*for(unsigned int i = 0; i < (Nsec*Ntheta*Nphi*Nrho); i += maxThreadsPerBlock){
	
	if(host_maxBlockOutput[i] != 0) cout << "block" << i/maxThreadsPerBlock << " max: " << host_maxBlockOutput[i] << " [" << i << "]" << endl;
	
	unsigned int found = 0;
	
	for(unsigned int y = 0; y < (maxThreadsPerBlock); y++){ // check relative maxima
	  if((host_maxRelOutput[i+y] != 0)){ cout << "out["<< i+y << "]="<< host_maxRelOutput[i+y]<< " "; found++; host_NMrel++;}
	}
	if(found > 0) cout << endl << endl;
      }*/
      
      cout << "NMrel from GPU "<< host_NMrel << " " << debug << endl;
      
      cudaFree(dev_maxBlockOutput);
      cudaFree(dev_maxRelOutput);
      
      free(host_maxBlockOutput);
      free(host_maxRelOutput);
      
#endif  
      
      host_NMrel = 0;
      
      int accumax = -1;
      int iphiMax = 0;
      int irhoMax = 0;
      int ithMax = 0;
      int isecMax = 0;
      
      
      for(unsigned int isec = 0; isec < Nsec; isec++){
	  
	  for(unsigned int ith = 1; ith < Ntheta; ith++){
	      
	      for(unsigned int iphi = 1; iphi < Nphi; iphi++){
		  
		  for(unsigned int irho = 1; irho < Nrho; irho++){
		      
		      float acc=acc_Mat[isec][ith][iphi][irho];
		      if (acc >= ac_soglia){
			  if (acc > accumax){
			      accumax=acc;
			  }
			  /*if (acc>acc_Mat[isec][ith-1][iphi][irho] && acc >= acc_Mat[isec][ith+1][iphi][irho]){
			      if (acc>acc_Mat[isec][ith][iphi-1][irho-1] && acc >= acc_Mat[isec][ith][iphi-1][irho+1]){ //TODO: chiedi a Lorenzo perché [iphi+1][irho+1] invece di [iphi-1][irho+1]
				  if (acc>acc_Mat[isec][ith][iphi][irho-1] && acc >= acc_Mat[isec][ith][iphi][irho+1]){
				      if (acc>acc_Mat[isec][ith][iphi+1][irho-1] && acc >= acc_Mat[isec][ith][iphi+1][irho+1]){*/
			  
			  if(acc > acc_Mat[isec][ith][iphi][irho-1] && acc >= acc_Mat[isec][ith][iphi][irho+1]){
			    if(acc > acc_Mat[isec][ith][iphi-1][irho] && acc >= acc_Mat[isec][ith][iphi+1][irho]){
					  //if (acc>=acc_Mat[isec][ith][iphi+1][irho] ){
					      accumax = acc_Mat[isec][ith][iphi+1][irho];
					      //Max_rel[isec][ith][iphi+1][irho]=1;
					      host_NMrel++;
					      ithMax=ith;
					      irhoMax=irho;
					      iphiMax=iphi;
					      isecMax=isec+1;
					      float t_th=(thetamin+ithMax*dtheta)*360.f/M_PI;
					      float t_rho=rhomin+irhoMax*drho;
					      float t_phi=phimin+iphiMax*dphi;
					      //float q=t_rho/sin(t_phi);
					      //float xm=-1/tan(t_phi);
					      //cout << acc <<" "<< t_rho <<" "<< t_phi << " " << isecMax << endl;
					      
					  //}
				      //}
				  //}
			      }
			  }
		      }
		  }
	      }
	  }
      }
#ifdef VERBOSE_DUMP
      cout << "NMrel from CPU "<< host_NMrel << endl;
#endif
      checkCudaErrors(cudaFree(dev_accMat));
      
      
  }
  
#ifdef CUDA_MALLOCHOST_OUTPUT      
  checkCudaErrors(cudaFreeHost(host_out_tracks));
#endif
    
    return 0;
}

/*****************************
 * file opener
 *****************************/


void read_inputFile(string file_path, unsigned int num_hits)
{
    
    ifstream input_f;
    
    string line;
    string value;
    
    stringstream ss;
    unsigned int val_iter;
    
    unsigned int line_read = 0;
    
    input_f.open(file_path.c_str());
    
    if (input_f.is_open())
    {
        while ( getline (input_f,line) && (line_read < num_hits) )
        {
            val_iter = 0;
            ss.str(line);
            //prendiamo dati direttamente dal file ASCII in input
            while(ss >> value){
                //i valori che ci interessano sono X, Y e Z
                if (val_iter == 0) x_values.push_back(atof(value.c_str()));
                else if (val_iter == 1) y_values.push_back(atof(value.c_str()));
                else if (val_iter == 2) z_values.push_back(atof(value.c_str()));
                val_iter++;
                
            }
            ss.clear();
	    line_read++;
        }
        input_f.close();
    }
    
    
    
    
}

