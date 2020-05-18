#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <string.h>
#include <time.h>
#include <omp.h>
#include <math.h>

#include "NBody.h"
#include "NBodyVisualiser.h"

#define USER_NAME "aca15a"		//username

#define THREADS_PER_BLOCK 256
#define SQRT_THREADS_PER_BLOCK sqrt(THREADS_PER_BLOCK)

//FUNCTION DEFINITIONS
void checkCUDAErrors(const char*);
void print_help();
//step function for OMP mode

__global__ void kernelStep_SOA(nbody_soa* d_body, float* g, float* Fx, float*Fy, unsigned int d_D, unsigned int d_N);
__global__ void grid_SOA(nbody_soa* d_body, float* g, float* Fx, float* Fy, unsigned int d_D, unsigned int d_N);
__global__ void kernelStep_AOS(nbody* d_body, float* g, float* Fx, float* Fy, unsigned int d_D, unsigned int d_N);
__global__ void grid_AOS(nbody* d_body, float* g, float* Fx, float* Fy, unsigned int d_D, unsigned int d_N);

__global__ void kernelStep_SOA_Interactions(nbody_soa* d_body, float* g, float* Fx, float* Fy, unsigned int d_D, unsigned int d_N);

void CUDAstep();

float* Fx;
float* Fy;


//GLOBAL VARIABLES
//create pointer to nbody structure to store bodies
struct nbody* body;
nbody_soa* d_body;


__device__ unsigned int d_N;
__device__ unsigned int d_D;
//__constant__ unsigned int d_N;
//__constant__ unsigned int d_D;

//nbody* d_body;
float* grid;
float* d_grid;

//initialise N D I as global to be used by step()
unsigned int N;
unsigned int D;
static unsigned int I;

//1D heatmap grid
//Define grid as d_body global variable

//MAIN//

int main(int argc, char* argv[]) {

	//use time to create d_body random seed
	srand((unsigned int)time(NULL)); // randomness initialization

	if (argc < 4) {
		exit(1);
	}

	//Checks if N is an int and also if it is positive
	//atoi will return 0 if it is not an int
	if (atoi(argv[1]) <= 0) {
		printf("The value for N is not valid, please enter d_body positive integer \n");
		exit(1);
	}
	else {
		N = atoi(argv[1]);
	}

	//assign memmory needed for number of bodies
	body = (struct nbody*)malloc(sizeof(struct nbody) * N);

	if (body == NULL) {
		fprintf(stderr, "malloc failed\n");
		return -1;
	}

	printf("N = %d \n", N);

	//Checks if D is an int and also if it is positive
	if (atoi(argv[2]) <= 0) {
		printf("The value for D is not valid, please enter d_body positive integer \n");
		exit(1);
	}
	else {
		D = atoi(argv[2]);
	}

	//assign memmory needed for dimensions of 1D grid
	grid = (float*)malloc(sizeof(float) * D * D);

	printf("D = %d \n", D);

	enum MODE mode;
	char* str;
	str = argv[3];

	if (strcmp(str, "CPU") == 0) {
		mode = CPU;
	}
	else if (strcmp(str, "OPENMP") == 0) {
		mode = OPENMP;
	}
	else if (strcmp(str, "CUDA") == 0) {
		mode = CUDA;
	}
	else {
		printf("The value for M is not valid, please enter CPU or OPENMP or CUDA \n");
		exit(1);
	}

	printf("Mode = %s \n", str);

	I = NULL;
	unsigned int n = 0;

	//check for the optional arguments
	for (int i = 4; i < argc; i += 2) {

		str = argv[i];

		if (strcmp(str, "-i") == 0) {
			I = atoi(argv[i + 1]);
			printf("I = %d \n", I);
		}


		if (strcmp(str, "-f") == 0) {
			FILE* f = NULL;
			str = argv[i + 1];

			f = fopen(str, "r");
			if (f == NULL) {
				fprintf(stderr, "Error: Could not find file \n");
				exit(1);
			}

			char buffer[200];
			//read single body data
			//run loop until end of input or nbodys
			while (!feof(f) && !(N == n)) {

				//get the input using fgets
				fgets(buffer, 200, f);

				//if line starts with # ignore
				if (buffer[0] == '#' || isspace(buffer[0])) {
					continue;
				}

				//printf("\n %s",buffer);

				//read 5 floats and skip "," and whitespace characters with %*c
				sscanf(buffer, "%f %*c%*c %f %*c%*c %f %*c%*c %f %*c%*c %f", &body[n].x, &body[n].y, &body[n].vx, &body[n].vy, &body[n].m);

				if (body[n].x < 0) {
					body[n].x = ((float)rand()) / RAND_MAX;       // returns d_body random integer between 0 and 1 for any invalid input
				}

				if (body[n].x < 0) {
					body[n].y = ((float)rand()) / RAND_MAX;      // returns d_body random integer between 0 and 1 for any invalid input
				}

				if (body[n].vx < 0) {
					body[n].vx = 0.0f;      // returns 0 for vx for any invalid input

				}

				if (body[n].vy < 0) {
					body[n].vy = 0.0f;      // returns 0 for vy for any invalid input
				}

				if (body[n].m <= 0) {				//mass can not be 0
					body[n].m = (float)1 / N;       // returns m as 1/n for any invalid input
				}

				//printf("\n body %d: x=%f, y=%f, vx=%f, vy=%f, m=%f", n, body[n].x, body[n].y, body[n].vx, body[n].vy, body[n].m);
				n++;
			}

			fclose(f);
		}
	}

	//if more bodies are needed
	for (unsigned int i = n; i < N; i++) {
		body[i].x = ((float)rand()) / RAND_MAX;      // Returns d_body random integer between 0 and 1
		body[i].y = ((float)rand()) / RAND_MAX;     // Returns d_body random integer between 0 and 1
		body[i].vx = 0.0f;      // Returns 0 for vx
		body[i].vy = 0.0f;      // Returns 0 for vy
		body[i].m = (float)1 / N;      // Returns m as 1/N
		//printf("\n body %d: x=%f, y=%f, vx=%f, vy=%f, m=%f", i, body[i].x, body[i].y, body[i].vx, body[i].vy, body[i].m);
	}

	if (mode == CUDA) {
		
		//Size of each float array
		unsigned int size = N * sizeof(float);

		//CPU soa
		nbody_soa* n_body = (nbody_soa*)malloc(sizeof(nbody_soa));


		//allocate memmory for CPU soa
		n_body->x = (float*)malloc(size);
		n_body->y = (float*)malloc(size);
		n_body->vx = (float*)malloc(size);
		n_body->vy = (float*)malloc(size);
		n_body->m = (float*)malloc(size);

		//populate soa
		for (int i = 0; i < N; i++) {
			n_body->x[i] = body[i].x;
			n_body->y[i] = body[i].y;
			n_body->vx[i] = body[i].vx;
			n_body->vy[i] = body[i].vy;
			n_body->m[i] = body[i].m;
		}
		
		//Intermediate soa
		nbody_soa* h_body = (nbody_soa*)malloc(sizeof(nbody_soa));

		//Allocate memory on GPU for intermediate float pointers and copy the information
		cudaMalloc((void**)&h_body->x, size);
		cudaMemcpy(h_body->x, n_body->x, size, cudaMemcpyHostToDevice);
		cudaMalloc((void**)&h_body->y, size);
		cudaMemcpy(h_body->y, n_body->y, size, cudaMemcpyHostToDevice);
		cudaMalloc((void**)&h_body->vx, size);
		cudaMemcpy(h_body->vx, n_body->vx, size, cudaMemcpyHostToDevice);
		cudaMalloc((void**)&h_body->vy, size);
		cudaMemcpy(h_body->vy, n_body->vy, size, cudaMemcpyHostToDevice);
		cudaMalloc((void**)&h_body->m, size);
		cudaMemcpy(h_body->m, n_body->m, size, cudaMemcpyHostToDevice);

		
		//Allocate memory on GPU for device soa and copy the information
		cudaMalloc(&(d_body), sizeof(nbody_soa));
		cudaMemcpy(d_body, h_body, sizeof(nbody_soa), cudaMemcpyHostToDevice);
		

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		/*NBODY AOS*/
		//cudaMalloc((void**)&(d_body), N * sizeof(nbody));
		//cudaMemcpy(d_body, body, N * sizeof(nbody), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&(d_grid), sizeof(grid));
		cudaMemcpy(d_grid, grid, sizeof(grid), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&Fx, size);
		cudaMalloc((void**)&Fy, size);

		d_D = D;
		d_N = N;
		//cudaMemcpyToSymbol(d_D, &D, sizeof(unsigned int));
		//cudaMemcpyToSymbol(d_N, &N, sizeof(unsigned int));

		cudaEvent_t start, stop;
		float milliseconds = 0;

		int errors;

		//checkCUDAErrors("CUDA malloc");
		//checkCUDAErrors("CUDA memcpy");

		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start);

		if (I == NULL) {
			setActivityMapData(d_grid);
			initViewer(N, D, mode, &CUDAstep);
			setActivityMapData(d_grid);
			//setNBodyPositions(d_body);
			setNBodyPositions2f(h_body->x, h_body->y);
			startVisualisationLoop();
		}
		else {
			for (unsigned int k = 1; k < I; k++) {
				CUDAstep();
			}
		}
		

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&milliseconds, start, stop);

		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		//stop timer
		int seconds = milliseconds / 1000;
		int mils = (milliseconds-(seconds*1000)) * 100;
		printf("Execution time is %d sec %d0.3 ms\n",seconds, mils);


		// Cleanup
		cudaFree(h_body->x);
		cudaFree(h_body->y);
		cudaFree(h_body->vx);
		cudaFree(h_body->vy);
		cudaFree(h_body->m);

		free(h_body);
		free(n_body);
	}
	
	// Cleanup
	cudaFree(d_body);
	cudaFree(d_grid);
	cudaFree(Fx);
	cudaFree(Fy);
	checkCUDAErrors("CUDA cleanup");

	//free any allocated memmory
	free(body);
	free(grid);

	cudaDeviceReset();

	return 0;

}

void CUDAstep(void) {

	dim3 blocksPerGrid(N/32, 1);
	dim3 threadsPerBlock(32, 1);

	kernelStep_SOA <<<blocksPerGrid, threadsPerBlock >>> (d_body, d_grid,Fx, Fy, d_D, d_N);
	//kernelStep_SOA_Interactions << <blocksPerGrid, threadsPerBlock >> > (d_body, d_grid, Fx, Fy, d_D, d_N);
	cudaDeviceSynchronize();
	grid_SOA << < blocksPerGrid, threadsPerBlock >> > (d_body, d_grid,Fx,Fy,d_D,d_N);
	

	//kernelStep_AOS << <blocksPerGrid, threadsPerBlock >> > (d_body, d_grid, d_D, d_N);
	//grid_AOS << < blocksPerGrid, threadsPerBlock >> > (d_body, d_grid);
	//cudaDeviceSynchronize();


}

__global__ void kernelStep_SOA_Interactions(nbody_soa* d_body, float* g, float* Fx, float* Fy, unsigned int d_D, unsigned int d_N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i != j && i<d_N && j<d_N){

		Fx[i] = 0.0f;
		Fy[i] = 0.0f;


		float xDiff = d_body->x[j] - d_body->x[i];
		float yDiff = d_body->y[j] - d_body->y[i];

		float denominator = powf(xDiff * xDiff + yDiff * yDiff + SOFTENING * SOFTENING, (3 / 2));

		Fx[i] += d_body->m[j] * xDiff / denominator;
		Fy[i] += d_body->m[j] * yDiff / denominator;

		
		Fx[i] *= G * d_body->m[i];
		Fy[i] *= G * d_body->m[i];

	}
	if (i < d_D * d_D) {
		g[i] = 0;
	}
}

__global__ void kernelStep_SOA(nbody_soa* d_body, float* g, float* Fx, float* Fy, unsigned int d_D, unsigned int d_N) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i<d_N) {		
		Fx[i] = 0.0f;
		Fy[i] = 0.0f;
//#pragma unroll
		for (int j = 0; j < d_N; j++) {
			if (i == j) continue;
			
			float xDiff = d_body->x[j] - d_body->x[i];

			float yDiff = d_body->y[j] - d_body->y[i];

			float denominator = powf(xDiff * xDiff + yDiff * yDiff + SOFTENING * SOFTENING, (3 / 2));

			Fx[i] += d_body->m[j] * xDiff / denominator;
			Fy[i] += d_body->m[j] * yDiff / denominator;
		}

		Fx[i] *= G * d_body->m[i];
		Fy[i] *= G * d_body->m[i];
	}
	if (i < d_D * d_D) {
		g[i] = 0;
	}
}

__global__ void grid_SOA(nbody_soa* d_body, float* g, float* Fx, float* Fy, unsigned int d_D, unsigned int d_N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	//get x,y positions on the activity heatmap
	if (i < d_N) {

		//update velocities
		d_body->vx[i] += Fx[i] / d_body->m[i] * dt;
		d_body->vy[i] += Fy[i] / d_body->m[i] * dt;

		//update positions
		d_body->x[i] += d_body->vx[i] * dt;
		if (d_body->x[i] > 1) d_body->x[i] = 1; //clamp to 1
		d_body->y[i] += d_body->vy[i] * dt;
		if (d_body->y[i] > 1) d_body->y[i] = 1;

		//get grid positions
		int pos, posx, posy;
		posx = (int)floor((d_body->x[i]) / ((float)1 / d_D));
		posy = (int)floor((d_body->y[i]) / ((float)1 / d_D));

		if ((posx <= d_D * d_D) && (posx >= 0) && (posy <= d_D * d_D) && (posy >= 0)) {

			//convert from 2D positions to 1D
			pos = posy + (d_D * posx);

			//local atomic add to the grid
			atomicAdd(&g[pos], (1.0/(d_N*0.3)));

		}
	}
}

__global__ void kernelStep_AOS(nbody* d_body, float* g, float* Fx, float* Fy, unsigned int d_D, unsigned int d_N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < d_N) {
		Fx[i] = 0.0f;
		Fy[i] = 0.0f;
//#pragma unroll
		for (int j = 0; j < d_N; j++) {
			float xDiff = d_body[j].x - d_body[i].x;

			float yDiff = d_body[j].y - d_body[i].y;

			float denominator = powf(xDiff * xDiff + yDiff * yDiff + SOFTENING * SOFTENING, 3 / 2);

			Fx[i] += d_body[j].m * xDiff / denominator;
			Fy[i] += d_body[j].m * yDiff / denominator;
		}

		Fx[i] *= G * d_body[i].m;
		Fy[i] *= G * d_body[i].m;
	}
	if (i < d_D * d_D) {
		g[i] = 0;
	}

}

__global__ void grid_AOS(nbody* d_body, float* g, float* Fx, float* Fy, unsigned int d_D, unsigned int d_N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	d_body[i].vx += (Fx[i] / d_body[i].m) * dt;
	d_body[i].vy += (Fy[i] / d_body[i].m) * dt;

	d_body[i].x += d_body[i].vx * dt;
	d_body[i].y += d_body[i].vy * dt;

	//get x,y positions on the activity heatmap
	int pos, posx, posy;
	posx = (int)floor((d_body[i].x) / ((float)1 / d_D));
	posy = (int)floor((d_body[i].y) / ((float)1 / d_D));

	if ((posx <= d_D * d_D) && (posx >= 0) && (posy <= d_D * d_D) && (posy >= 0)) {

		//convert from 2D positions to 1D
		pos = posy + (d_D * posx);

		//local atomic add to the grid
		atomicAdd(&g[pos], (1.0 / (d_N * 0.3)));

	}
}

void checkCUDAErrors(const char* msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void print_help() {
	printf("nbody_%s N D M [-i I] [-i input_file]\n", USER_NAME);

	printf("where:\n");
	printf("\tN                Is the number of bodies to simulate.\n");
	printf("\tD                Is the integer dimension of the activity grid. The Grid has D*D locations.\n");
	printf("\tM                Is the operation mode, either  'CPU' or 'OPENMP'\n");
	printf("\t[-i I]           Optionally specifies the number of simulation iterations 'I' to perform. Specifying no value will use visualisation mode. \n");
	printf("\t[-f input_file]  Optionally specifies an input file with an initial N bodies of data. If not specified random data will be created.\n");
}