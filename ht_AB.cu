#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <cmath>

using namespace std;

#define Nphi 4 // Number of sectors (bins) in the transverse plane (azimuthal angle, phi)
#define Ntheta 16 // Number of bins for the polar angle (z-plane)
#define NA 1024 // number of bins in the x-plane
#define NB 1024 // number of bins in the y-plane

#define Amin -50000.f // mm
#define Amax 50000.f // mm
#define Bmin -50000.f // mm
#define Bmax 50000.f // mm
#define rhomin 2000.f // mm
#define rhomax 2000.f // mm
#define thetamin 0.f // rad
#define thetamax M_PI // rad

#define acc_thresh 4 // vote threshold

int h_accMat[Nphi][Ntheta][NB][NA]; //host-device accumulation matrix

bool read_hits(char *filename, float *x_val, float *y_val, float *z_val, int *n_hits){

	FILE *f = fopen(filename, "r");

	if(f == NULL) return false;

	char buf[256];

	size_t lines = 0;

	while(fgets(buf, sizeof(buf), f) != NULL){
		lines++;
	}

	rewind(f);

	x_val = (float *) malloc(sizeof(float)*lines);
	y_val = (float *) malloc(sizeof(float)*lines);
	z_val = (float *) malloc(sizeof(float)*lines);

	*n_hits = lines;

	lines = 0;

	while(fgets(buf, sizeof(buf), f) != NULL){
		sscanf(buf,"%f %f %f", &(x_val[lines]), &(y_val[lines]), &(z_val[lines]));
		lines++;
	}

	return true;


}

#define get4DIndex(phi,theta,B,A) (((phi)*Ntheta*NB*NA)+((theta)*NB*NA)+((B)*NA)+(A))
#define get3DIndex(theta,B,A) (((theta)*NB*NA)+((B)*NA)+(A))
#define get2DIndex(B,A) (((B)*NA)+(A))

__global__ void HoughVote(float *x_val, float *y_val, float *z_val, float *acc_mat, float d_phi, float d_theta, float dA, float dB){

	__shared__ float x, y, z;

	// each thread-block processes a single hit:
	// the first thread loads from the global memory into the shared memory
	// of each block the x,y,z values referred to the hit
	if(threadIdx.x == 0){
		x = x_val[blockIdx.x];
		y = y_val[blockIdx.x];
		z = z_val[blockIdx.x];
	}
	__syncthreads(); // all the other threads in the block wait for it to finish

	// compute the polar angle theta of the hit with respect to the z-axis
	float r = sqrt(x*x + y*y + z*z);
	float theta = acos(z/r);
	// then compute the relative bin index
	float i_theta = floor(theta/d_theta);

	//compute the azimuthal angle phi = atan(y/x)
	float phi = atan2(y,x); //NOTE: we must use atan2 (see notes)
	// and adjust the angle (since the output is between [-M_PI, M_PI])
	// to retrieve the correct bin-index / sector ( between 0 and 3)
	if(phi < 0){
		phi += 2*M_PI; //get the corresponding positive radiant
	}
	// compute the bin
	int i_phi = floor(phi / d_phi);

	int iA = threadIdx.x; // each thread in a thread block is associated to an A bin
	//compute real A for this thread  based on the bin size
	float A = Amin + (iA * d_A);

	//compute B based on this A
	float B = ((x*x)+(y*y)-2*A*x)/(2*y);
	// and the relative bin
	int iB = floor((B-Bmin)/dB);

	//if iB is in the correct range (so that satisfies the equation)
	if(iB > 0 && iB < NB){
		// compute the index in the accumulator matrix
		int acc_index = get4DIndex(i_phi,i_theta,iB,iA);
		// vote in the hough space
		//(we use atomic add cause more than one thread could vote in the same area)
		atomicAdd(&(acc_mat[acc_index]), 1);
	}


}

int main(int argc, char *argv[]){

	memset(h_accMat, 0, sizeof(int)*Nphi*Ntheta*NB*NA); //set the host accumulator matrix to 0

	float *h_x_val; //this will be allocated while reading the file
	float *h_y_val;
	float *h_z_val;

	float dphi = (2*M_PI)/Nphi;
	float dtheta= M_PI/Ntheta;	//delta-theta (based on the bin size and max-min values)
	//NOTE theta (polar angle) is defined between 0 and M_PI
	float dA=(Amax-Amin)/NA;	// delta-A
	float dB=(Bmax-Bmin)/NB;	// delta-B

	// gather GPU informations
	int GPU_N;
	cudaGetDeviceCount(&GPU_N);
	cudaDeviceProp deviceProp[GPU_N];
	for(int i = 0; i < GPU_N; i++){
		cudaGetDeviceProperties(&(deviceProp[i]),i);
		cout << "Device " << i << " " << deviceProp[i].name << endl;
		cout << "Max threads per block " << deviceProp[i].maxThreadsPerBlock << endl;
		cout << "Max thread per dimension (" << deviceProp[i].maxThreadsDim[0] << ","
				<< deviceProp[i].maxThreadsDim[1] << "," << deviceProp[i].maxThreadsDim[2] << ")"
				<< endl;
		cout << "Max thread per SM " << deviceProp[i].maxThreadsPerMultiProcessor << endl;
		cout << "Warp size per SM " << deviceProp[i].warpSize << endl;
	}

	int *d_accMat;
	float *d_x_val;
	float *d_y_val;
	float *d_z_val;

	//allocate and initialize accumulator matrix on the GPU

	cudaMalloc((void **) &d_accMat, (sizeof(int)*Nphi*Ntheta*NB*NA));
	cudaMemset(d_accMat, 0, (sizeof(int)*Nphi*Ntheta*NB*NA));

	int N_HITS;

	if(!read_hits("hits-5000.txt", h_x_val, h_y_val, h_z_val, &N_HITS)) return -1;

	//allocate memory on the device for input values
	cudaMalloc((void **) &d_x_val, sizeof(float)*N_HITS);
	cudaMalloc((void **) &d_y_val, sizeof(float)*N_HITS);
	cudaMalloc((void **) &d_z_val, sizeof(float)*N_HITS);

	//copy values on the GPU memory from host to device
	cudaMemcpy(d_x_val, h_x_val, sizeof(float)*N_HITS, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y_val, h_y_val, sizeof(float)*N_HITS, cudaMemcpyHostToDevice);
	cudaMemcpy(d_z_val, h_z_val, sizeof(float)*N_HITS, cudaMemcpyHostToDevice);

	dim3 grid(N_HITS); // 1D-grid over the number of hits
	dim3 block(NA);	//1D-threadBlock over the bin size of A parameter (in this case is equal to the maximum num. of thread in a threadblock)

	HoughVote<<<grid, block>>>(d_x_val, d_y_val, d_z_val, dphi, dtheta, dB, dA);





















}
