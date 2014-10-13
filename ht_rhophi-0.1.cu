//
//  ht_helix.cpp
//  
//
//  Created by Lorenzo Rinaldi on 29/04/14.
//
//
// compile:
// nvcc -I/usr/local/cuda-5.5/samples/common/inc -I/usr/local/cuda-5.5/targets/x86_64-linux/include -gencode arch=compute_20,code=sm_21 -o ht_rhophi ht_rhophi.cuda_runtime


//NOTE: INVERTITE DIMENSIONI NRHO-NPHI PER ACCESSO MATRICE
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

//#define CUDA_MANAGED_TRANSFER
#define VERBOSE_OUTPUT

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

#define max_tracks_out 100

int acc_Mat [ Nsec ][ Ntheta ][Nrho ][Nphi ] ;
//int Max_rel [ Nsec ][ Ntheta ] [Nrho ][Nphi ];


float dtheta= M_PI/Ntheta;
float drho= (rhomax-rhomin)/Nrho;
float dphi= (phimax-phimin)/Nphi;

vector<float> x_values;
vector<float> y_values;
vector<float> z_values;

struct track_param{
	int acc;
      /*unsigned int isec;
      unsigned int ith;
      unsigned int iphi;
      unsigned int irho;*/
  };

#define get4DIndex(s,t,r,p) ((s)*(Ntheta*Nrho*Nphi))+(((t)*Nrho*Nphi) +(((r)*Nphi)+(p)))
#define get3DIndex(t,r,p) (((t)*Nphi*Nrho) +(((r)*Nphi)+(p)))

__global__ void voteHoughSpace(float *dev_x_values, float *dev_y_values, float *dev_z_values, int *dev_accMat, float dtheta, float dphi, float drho){

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

	//float sec=atan2(y_val,x_val);
  	/*In trigonometria la funzione a due argomenti atan2 
  	rappresenta una variazione dell'arcotangente. 
  	Comunque presi gli argomenti reali x e y non nulli, 
  	atan2(y,x) indica l'angolo in radianti tra l'asse positivo 
  	delle X in un piano e un punto di coordinate (x,y) giacente 
  	su di esso. L'angolo è positivo se antiorario 
  	(semipiano delle ordinate positive, y>0) e negativo 
  	se in verso orario (semipiano delle ordinate negative, y < 0).

	Questa funzione quindi restituisce un valore compreso 
	nell'intervallo (-\pi, \pi). La funzione è definita per 
	tutte le coppie di valori reali (x, y) eccetto la coppia (0, 0).*/

	/*if (sec<0.f) //angle sec < 0.f : lower planes (y < 0) 
	{
		sec=2*M_PI+sec; //make it positive
	}
	int isec=int(sec/2/M_PI*Nsec);*/

	int iphi = threadIdx.x;
	float phi=phimin+iphi*dphi;
	float rho=R2/2.f/(x_val*cos(phi)+y_val*sin(phi));
	int irho=(int)((rho-rhomin)/drho)+0.5f;

	int accu_index = get3DIndex(ith, irho, iphi);

	if (rho<=rhomax && rho>rhomin)
	{
		atomicAdd(&(dev_accMat[accu_index]),1);
	}
}

__global__ void findRelativeMax(int *dev_accMat, struct track_param *dev_output, unsigned int *NMrel){


	//unsigned int isec = blockIdx.x;
	unsigned int ith = blockIdx.x ; //instead of blockIdx.y / (...etc...)
	unsigned int iphi = threadIdx.x;
	unsigned int irho = blockIdx.y;

	unsigned int globalIndex = getGlobalIdx_1D_2D(); //instead of ...2D_2D()

  //check if it is a local maxima by verifying that it is greater then (>=) its neighboors

  //we must check from ith >= 1, iphi >= 1, irho >= 1
	if(((iphi > 0) && (irho > 0)) && ((iphi < Nphi-1) && (irho < Nrho-1))){

    //each thread is assigned to one point of the accum. matrix:
		int acc= dev_accMat[get3DIndex(ith, irho, iphi)];

		if (acc >= ac_soglia){

			if(acc > dev_accMat[get3DIndex(ith, irho-1, iphi)] && acc >= dev_accMat[get3DIndex(ith, irho+1, iphi)]){

				if(acc > dev_accMat[get3DIndex( ith, irho, iphi-1)] && acc >= dev_accMat[get3DIndex(ith, irho , iphi+1)]){

					atomicAdd(NMrel, 1);

					dev_output[globalIndex].acc = acc;
					/*dev_output[globalIndex].isec = isec;
					dev_output[globalIndex].ith = ith;
					dev_output[globalIndex].iphi = iphi;
					dev_output[globalIndex].irho = irho;*/

				}

			}
		}
	}
}

//class used to hold each sector of ht_rhophi data, to be assigned to some GPU
class ht_gpu_data{

	public:
		int id; //IMPORTANT: this is the GPU-id which this computation is assigned to
		cudaStream_t stream;
	  	//cudaEvent_t event;
	  	cudaDeviceProp dev_prop;
		int* accMat; //single sector portion of the accumulator matrix
		struct track_param *indexOutput; //output tracks
		unsigned int *NMrel;
#ifndef CUDA_MANAGED_TRANSFER
		struct track_param *host_indexOutput; //HOST output tracks
		unsigned int host_NMrel;
#endif
		float* x_values;
		float* y_values;
		float* z_values;
		unsigned int hit_n; //num of hit for this computation




		void init(int dev_id){
			if(dev_id >= 0){
				id = dev_id;
				//first, set proper GPU device id
				checkCudaErrors(cudaSetDevice(id));
				//checkCudaErrors(cudaEventCreate(&event));
				checkCudaErrors(cudaStreamCreate(&stream));
				checkCudaErrors(cudaGetDeviceProperties(&dev_prop, dev_id));
				hit_n = 0;
			}else{
				cout << "GPU Init Error: GPU ID must be >= 0, " << dev_id << " given. Stop init()." << endl;
			}
		}

		void destroy(){
			//first, set proper GPU device id
			checkCudaErrors(cudaSetDevice(id));
			checkCudaErrors(cudaStreamDestroy(stream));
			//checkCudaErrors(cudaEventDestroy(event));

		}
		
		void accMat_init(){
			size_t memsize = Ntheta*Nrho*Nphi;
			//first, set proper GPU device id
			checkCudaErrors(cudaSetDevice(id));

#ifdef CUDA_MANAGED_TRANSFER
			checkCudaErrors(cudaMallocManaged((void **) &accMat, (sizeof(int)* memsize) ));
#else
			checkCudaErrors(cudaMalloc((void **) &accMat, (sizeof(int)* memsize) ));
#endif

			checkCudaErrors(cudaMemset(accMat, 0, (sizeof(int)*memsize)));
		}

		void indexOutput_init(){
			size_t memsize = Ntheta*Nrho*Nphi;
			//first, set proper GPU device id
			checkCudaErrors(cudaSetDevice(id));

#ifdef CUDA_MANAGED_TRANSFER
			checkCudaErrors(cudaMallocManaged(&indexOutput,(sizeof(struct track_param)* (memsize)) ));
			checkCudaErrors(cudaMallocManaged(&NMrel,sizeof(unsigned int) ));
			*NMrel = 0;
#else
			checkCudaErrors(cudaMalloc(&indexOutput,(sizeof(struct track_param)* (memsize)) ));
			checkCudaErrors(cudaMalloc(&NMrel,sizeof(unsigned int) ));
			checkCudaErrors(cudaMemset(NMrel, 0, (sizeof(unsigned int))));
			host_NMrel = 0;
#endif
			checkCudaErrors(cudaMemset(indexOutput, -1, (sizeof(struct track_param)* (memsize))));

		}

#ifndef CUDA_MANAGED_TRANSFER
		void host_indexOutput_init(){
			size_t memsize = Ntheta*Nrho*Nphi;
			//first, set proper GPU device id
			checkCudaErrors(cudaSetDevice(id));
			checkCudaErrors(cudaMallocHost((void **) &host_indexOutput, (sizeof(struct track_param)*(memsize))));
		}

		void host_indexOutput_memcpy(){
			size_t memsize = Ntheta*Nrho*Nphi;
			//first, set proper GPU device id
			checkCudaErrors(cudaSetDevice(id));
			checkCudaErrors(cudaMemcpy((void *) host_indexOutput, indexOutput, (sizeof(struct track_param)* (memsize)), cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaMemcpy((void *) &host_NMrel, NMrel, (sizeof(int)), cudaMemcpyDeviceToHost));
		}
#endif

		void inputVal_init(vector<float> x_vec, vector<float> y_vec, vector<float> z_vec){
			//first, set proper GPU device id
			checkCudaErrors(cudaSetDevice(id));
			
			//check x_vec, y_vec and z_vec are the same size
			if(!((x_vec.size() == y_vec.size()) && (x_vec.size() == z_vec.size()))){
				cout << "Error during input values read. Aborting..." << endl;
				exit(1);
			}

			//read from x_vec, y_vec, z_vec size() and assign to inputVal_num
			unsigned int inputVal_num = x_vec.size(); 
			//alloc and init each GPU array

#ifdef CUDA_MANAGED_TRANSFER
			checkCudaErrors(cudaMallocManaged((void **) &x_values, (sizeof(float) * inputVal_num)));
			checkCudaErrors(cudaMallocManaged((void **) &y_values, (sizeof(float) * inputVal_num)));
			checkCudaErrors(cudaMallocManaged((void **) &z_values, (sizeof(float) * inputVal_num)));
#else
			checkCudaErrors(cudaMalloc((void **) &x_values, (sizeof(float) * inputVal_num)));
			checkCudaErrors(cudaMalloc((void **) &y_values, (sizeof(float) * inputVal_num)));
			checkCudaErrors(cudaMalloc((void **) &z_values, (sizeof(float) * inputVal_num)));

#endif

			checkCudaErrors(cudaMemset(x_values, 0.f, (sizeof(float) * inputVal_num)));
			checkCudaErrors(cudaMemset(y_values, 0.f, (sizeof(float) * inputVal_num)));
			checkCudaErrors(cudaMemset(z_values, 0.f, (sizeof(float) * inputVal_num)));

#ifdef CUDA_MANAGED_TRANSFER
			for(unsigned int i = 0; i < inputVal_num; i++){
				x_values[i] = x_vec.at(i);
				y_values[i] = y_vec.at(i);
				z_values[i] = z_vec.at(i);
			}
#else

			float *x_values_temp;
			float *y_values_temp;
			float *z_values_temp;

			x_values_temp = (float*) malloc(sizeof(float)*inputVal_num);
			y_values_temp =  (float*) malloc(sizeof(float)*inputVal_num);
			z_values_temp = (float*)  malloc( sizeof(float)*inputVal_num);

			for(unsigned int i = 0; i < inputVal_num; i++){
				x_values_temp[i] = x_vec.at(i);
				y_values_temp[i] = y_vec.at(i);
				z_values_temp[i] = z_vec.at(i);
			}

			checkCudaErrors(cudaMemcpy(x_values, x_values_temp, sizeof(float)*inputVal_num, cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(y_values, y_values_temp, sizeof(float)*inputVal_num, cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(z_values, z_values_temp, sizeof(float)*inputVal_num, cudaMemcpyHostToDevice));

			free(x_values_temp);
			free(y_values_temp);
			free(z_values_temp);

#endif

			hit_n = inputVal_num;

		}

		void vote_HoughSpace(){
			//first, set proper GPU device id
			checkCudaErrors(cudaSetDevice(id));
			voteHoughSpace <<<hit_n, Nphi, 0, stream>>> (x_values, y_values, z_values, accMat, dtheta, dphi, drho); 
			//assumes that Nphi == Nrho
		}

		void find_RelativeMax(){
			//first, set proper GPU device id
			checkCudaErrors(cudaSetDevice(id));
			//find adeguate worksize, relative to this GPU
			unsigned int dim_x_block = Nphi;
		    unsigned int dim_y_block = dev_prop.maxThreadsPerBlock/dim_x_block;
		    //working on a single sector, thus we don't need grid-Y dimension
		    unsigned int dim_x_grid = Ntheta;
		    unsigned int dim_y_grid = (Nrho/dim_y_block);

		    dim3 grid(dim_x_grid,dim_y_grid);
		    dim3 block(dim_x_block, dim_y_block);

		    findRelativeMax <<<grid, block, 0, stream>>> (accMat, indexOutput, NMrel);



		}

		void dev_sync(){
			//first, set proper GPU device id
			checkCudaErrors(cudaSetDevice(id));
			checkCudaErrors(cudaDeviceSynchronize());
		}
		
		void accMat_free(){
			//first, set proper GPU device id
			checkCudaErrors(cudaSetDevice(id));
			checkCudaErrors(cudaFree(accMat));
			checkCudaErrors(cudaFree(x_values));
			checkCudaErrors(cudaFree(y_values));
			checkCudaErrors(cudaFree(z_values));
			hit_n = 0;
#ifdef CUDA_MANAGED_TRANSFER
			*NMrel = 0;
#endif
		}

		void indexOutput_free(){
			//first, set proper GPU device id
			checkCudaErrors(cudaSetDevice(id));
			checkCudaErrors(cudaFree(indexOutput));
			checkCudaErrors(cudaFree(NMrel));
#ifndef CUDA_MANAGED_TRANSFER
			checkCudaErrors(cudaFreeHost( host_indexOutput));
#endif
		}
};

#define MAX_GPUS 4

vector<ht_gpu_data> sector_data;

//function headers
void read_inputFile(string file_path, unsigned int num_hits);
void init_inputVal();

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
			default:
			printf("Unkown option!\n");
			help(argv[0]);
			return 0;
		}
	}

	int GPU_N = 0;
	checkCudaErrors(cudaGetDeviceCount(&GPU_N));


	if (GPU_N > 0){

		//...
    	//	inside loop: allocate accMat for each GPU found, using sectors as main dimensions
    	//	E.G.: 1 GPU -> all 4 sector are assigned to default GPU
    	//  	  2 GPUs -> 2 sectors for GPU0 and other 2 for GPU1
    	//		  3 GPUs -> 2 sectors for GPU0 (best) and others for GPU1 and GPU2
    	//		  4 GPUs -> each sector is assigned to a different GPU
		//
    	//	wehen creating sector data structure, we use 
		//  modulus operand to assign sectors to available GPUs: 
		//    sec_n % GPU_N
    	//...

		//for each sector, we create a different dataset 
		//to be assigned to	available GPUs
		for(unsigned int i = 0; i < Nsec; i++){

			ht_gpu_data new_data;
			sector_data.push_back(new_data);

			//define wich GPU has to be assigned to this sector: sec_n % GPU_N
			unsigned int gpu_id = i % GPU_N;

    		//init each data class
			sector_data.at(i).init(gpu_id);

			cout << "sector " << i << " assigned to GPU" << sector_data.at(i).id << " : " << sector_data.at(i).dev_prop.name << endl;
		}
		
		cout << "nhits, host_output init, input malloc +  copy HtoD, malloc+memset accMat & indexOutput, vote, rel. max, result DtoH" << endl; 

    	//executions loop
		for(unsigned int loop = 0; loop < N_LOOPS; loop++){

			float timing[6];
		  	//float R = 0.f;

		  	// init matrix to zeros
			memset(&acc_Mat, 0, (sizeof(int)*(Nsec*Ntheta*Nrho*Nphi)) );

		  	//alloc accumulator matrix on GPU
			start_time();

			for(unsigned int i = 0; i < Nsec; i++){
				sector_data.at(i).accMat_init(); //alloc accumulator matrix
				sector_data.at(i).indexOutput_init(); //alloc output tracks structure
			}

			timing[1] = stop_time("malloc accMat+indexOutput and memset(0)");

#ifndef CUDA_MANAGED_TRANSFER
			start_time();
			for(unsigned int i = 0; i < Nsec; i++){
				sector_data.at(i).host_indexOutput_init();
			}
			timing[5] = stop_time("mallochost output struct");
#endif

		  	//end alloc

		  	//set values inside global x_values , y_values , z_values vectors
			read_inputFile("hits-5000.txt", N_HITS);

			//then assign each hit to its sector_data
			start_time();
			init_inputVal();
			timing[0] = stop_time("Input malloc and copy HtoD");

			//let each GPU begin vote computation for the assigned sector
			start_time();
			for(unsigned int i = 0; i < Nsec; i++){
      	  		sector_data.at(i).vote_HoughSpace();
      		}
      		//host waits for GPUs while processing...
      		for(unsigned int i = 0; i < Nsec; i++){
      			sector_data.at(i).dev_sync();//sync
      		}
      		timing[2] = stop_time("Vote");

#ifdef VERBOSE_OUTPUT
      		/*for(unsigned int i = 0; i < Nsec; i++){
      			cout << "sec " << i << ":" << endl;
				for(unsigned int ith = 0; ith < Ntheta; ith++){
		      
		    		for(unsigned int iphi = 0; iphi < Nphi; iphi++){
			  
						for(unsigned int irho = 0; irho < Nrho; irho++){

							if(sector_data.at(i).accMat[get3DIndex(ith, iphi, irho)] != 0) 
								cout << "accMat[get3DIndex(" << ith << ", " << iphi << ", " << irho << ") = " << sector_data.at(i).accMat[get3DIndex(ith, iphi, irho)] << endl;

						}
					}
				}
			}*/
#endif

      		//start relative Maxima finding
      		start_time();
      		for(unsigned int i = 0; i < Nsec; i++){
      	  		sector_data.at(i).find_RelativeMax();
      		}
      		//host waits for GPUs while processing...
      		for(unsigned int g = 0; g < GPU_N; g++){
      			checkCudaErrors(cudaSetDevice(g));
      			checkCudaErrors(cudaDeviceSynchronize()); //sync
      		}
			timing[3] = stop_time("Find Rel. Maxima");

#ifndef CUDA_MANAGED_TRANSFER
			start_time();
			for(unsigned int i = 0; i < Nsec; i++){
				sector_data.at(i).host_indexOutput_memcpy();
			}

			timing[4] = stop_time("Copy results DtoH");

#endif


			//-----------------------------------------
			//TODO: merge every sector results
			//-----------------------------------------

			//free GPU memory
			for(unsigned int i = 0; i < Nsec; i++){

#ifdef VERBOSE_OUTPUT


#ifdef CUDA_MANAGED_TRANSFER
				unsigned int *mrel = sector_data.at(i).NMrel;
#else
				unsigned int *mrel = &(sector_data.at(i).host_NMrel);
#endif
				cout << "NMrel found on GPU" << sector_data.at(i).id <<
				" : " << *(mrel) << endl;
#endif
				sector_data.at(i).accMat_free();
	      		sector_data.at(i).indexOutput_free();

			x_values.clear();
			y_values.clear();
			z_values.clear();
	      	}
#ifdef CUDA_MANAGED_TRANSFER
	      	cout << N_HITS << " " << timing[0] << " " << timing[1] << " " << timing[2] << " " << timing[3] << /*" " << timing[4] <<*/ endl; 
#else
	      	cout << N_HITS << " " << timing[5] << " " << timing[0] << " " << timing[1] << " " << timing[2] << " " << timing[3] << " " << timing[4] << endl;
#endif
      	}//end exec loop

      	//clean all gpu_data class objects
      	for(unsigned int i = 0; i < Nsec; i++){
      		sector_data.at(i).destroy();
      	}

      	sector_data.clear();

      }else{
      	cout << "ERROR: NO GPUs found. Aborting..." << endl;
		exit(1);
      }

	cudaDeviceReset();
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

/**************************
* init input values
***************************/
void init_inputVal(){

	vector<float> temp_x_val[Nsec];
	vector<float> temp_y_val[Nsec];
	vector<float> temp_z_val[Nsec];

	//check x_values, y_values and z_values are the same size
	if(!((x_values.size() == y_values.size()) && (x_values.size() == z_values.size()))){
		cout << "Error during input values read. Aborting..." << endl;
		exit(1);
	}

	unsigned int inputVal_num = x_values.size(); 

	for(unsigned int i = 0; i < inputVal_num; i++){
		//setup each GPU input value based on the sector where it is found
		float sec=atan2(y_values.at(i),x_values.at(i));
		if (sec<0.f)
		{
			sec=2*M_PI+sec;
		}
		int isec=int(sec/2/M_PI*Nsec);
		temp_x_val[isec].push_back(x_values.at(i));
		temp_y_val[isec].push_back(y_values.at(i));
		temp_z_val[isec].push_back(z_values.at(i));
	}

	for(unsigned int i = 0; i < Nsec; i++){ //fill data with each GPU temp vector data
		sector_data.at(i).inputVal_init(temp_x_val[i], temp_y_val[i], temp_z_val[i]);
	}
}
