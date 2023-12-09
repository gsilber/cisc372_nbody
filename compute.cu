#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"

#include<stdio.h>

#undef DEBUG
//#define DEBUG 1

#define THREADS_PER_BLOCK 256
#define VECTOR_SIZE 3
#define BLOCK_WIDTH 16

__global__ void gpu_compute_accels(vector3* accels, vector3* hPos, double* mass) {

    // TODO: shared memory hopefully speeds things up
    __shared__ double shared_mass[BLOCK_WIDTH];
    __shared__ vector3 shared_hPos_x[BLOCK_WIDTH];
    __shared__ vector3 shared_hPos_y[BLOCK_WIDTH];

    // the entity being accelerated
    int entityIndex = blockIdx.x * BLOCK_WIDTH + threadIdx.x;

    // the entity causing the acceleration
    int otherEntityIndex = blockIdx.y * BLOCK_WIDTH + threadIdx.y;

    if (threadIdx.x == 0) {
        shared_mass[threadIdx.y] = mass[otherEntityIndex];
        shared_hPos_x[threadIdx.y][0] = hPos[otherEntityIndex][0];
        shared_hPos_x[threadIdx.y][1] = hPos[otherEntityIndex][1];
        shared_hPos_x[threadIdx.y][2] = hPos[otherEntityIndex][2];
    }
    if (threadIdx.y == 0) {
        shared_hPos_y[threadIdx.x][0] = hPos[entityIndex][0];
        shared_hPos_y[threadIdx.x][1] = hPos[entityIndex][1];
        shared_hPos_y[threadIdx.x][2] = hPos[entityIndex][2];
    }

    __syncthreads();

    if(entityIndex < NUMENTITIES && otherEntityIndex < NUMENTITIES) {

        if (entityIndex == otherEntityIndex) {
				FILL_VECTOR(accels[entityIndex * NUMENTITIES + otherEntityIndex],0,0,0);
			}
        else{
            vector3 distance;
            for (int k=0;k<3;k++) distance[k]=shared_hPos_y[threadIdx.x][k]-shared_hPos_x[threadIdx.y][k];
            double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
            double magnitude=sqrt(magnitude_sq);
            double accelmag=-1*GRAV_CONSTANT*shared_mass[threadIdx.y]/magnitude_sq;
            FILL_VECTOR(accels[entityIndex * NUMENTITIES + otherEntityIndex],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
        }
    }
}

void compute_accels(vector3* h_values, vector3* d_values) {

    dim3 blockSize = dim3(BLOCK_WIDTH, BLOCK_WIDTH, 1);
    dim3 gridSize = dim3(ceil((float)NUMENTITIES / (float)BLOCK_WIDTH), ceil((float)NUMENTITIES / (float)BLOCK_WIDTH), 1);

    // call the kernel
    gpu_compute_accels<<<gridSize, blockSize>>>(d_values, d_hPos, d_mass);

    #ifdef DEBUG
    // copy the gpu acceleration matrix back to the host acceleration matrix
    cudaMemcpy(h_values, d_values, sizeof(vector3) * NUMENTITIES * NUMENTITIES, cudaMemcpyDeviceToHost);

    // print out the matrix after GPU operation
    for (int i = 0; i < NUMENTITIES * NUMENTITIES; i++) {
        printf("%32.32f\n", h_values[i][0]);
        if(i % NUMENTITIES == 0)
            printf("\n");        
    }
    printf("\n");
    #endif
}

__global__ void gpu_advance_simulation(vector3** accels, vector3* hPos, double* mass) {

}

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){

	// make an acceleration matrix which is NUMENTITIES squared in size
	vector3* h_values=(vector3*)malloc(sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	vector3** h_accels=(vector3**)malloc(sizeof(vector3*)*NUMENTITIES);
	for (int i = 0; i < NUMENTITIES; i++)
		h_accels[i]=&h_values[i*NUMENTITIES];

    for (int i = 0; i < NUMENTITIES; i++) {
        for(int j = 0; j < NUMENTITIES; j++) {
            for(int k = 0; k < VECTOR_SIZE; k++) {
                h_accels[i][j][k] = 2.0;
            }
        }
    }

    // create that accleration "matrix" on the GPU
    vector3* d_values;
    cudaMalloc((void **)&d_values, sizeof(vector3) * NUMENTITIES * NUMENTITIES);

    // create the positions array on the GPU
    cudaMalloc((void **)&d_hPos, sizeof(vector3) * NUMENTITIES);
    cudaMemcpy(d_hPos, hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);

    // create the mass array on the GPU
    cudaMalloc((void **)&d_mass, sizeof(double) * NUMENTITIES);
    cudaMemcpy(d_mass, mass, sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice);

    // create the velocity array on the GPU
	cudaMalloc((void **)&d_hVel, sizeof(vector3) * NUMENTITIES);
    cudaMemcpy(d_hVel, hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);

    compute_accels(h_values, d_values);

    //sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	for (int i = 0; i < NUMENTITIES; i++){
		vector3 accel_sum={0,0,0};
		for (int j = 0; j < NUMENTITIES; j++){
			for (int k = 0; k < 3; k++)
				accel_sum[k]+=h_accels[i][j][k];
		}
		//compute the new velocity based on the acceleration and time interval
		//compute the new position based on the velocity and time interval
		for (int k = 0; k < 3; k++){
			hVel[i][k]+=accel_sum[k]*INTERVAL;
			hPos[i][k]+=hVel[i][k]*INTERVAL;
		}
	}
	
    free(h_accels);
	free(h_values);

    cudaFree(d_hVel);
    cudaFree(d_hPos);
    cudaFree(d_mass);
}
