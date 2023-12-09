#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"

#include<stdio.h>

//#undef DEBUG
#define DEBUG 1

#define THREADS_PER_BLOCK 256
#define VECTOR_SIZE 3
#define BLOCK_WIDTH 16

__global__ void gpu_compute_accels(vector3** accels, vector3* hPos, double* mass) {

    // the entity being accelerated
    int entityIndex = blockIdx.x * BLOCK_WIDTH + threadIdx.x;

    // the entity causing the acceleration
    int otherEntityIndex = blockIdx.y * BLOCK_WIDTH + threadIdx.y;

    if(entityIndex < NUMENTITIES && otherEntityIndex < NUMENTITIES) {
        if (entityIndex == otherEntityIndex) {
				FILL_VECTOR(accels[entityIndex][otherEntityIndex],0,0,0);
			}
        else{
            vector3 distance;
            for (int k=0;k<3;k++) distance[k]=hPos[entityIndex][k]-hPos[otherEntityIndex][k];
            double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
            double magnitude=sqrt(magnitude_sq);
            double accelmag=-1*GRAV_CONSTANT*mass[otherEntityIndex]/magnitude_sq;
            FILL_VECTOR(accels[entityIndex][otherEntityIndex],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
        }
    }
}

void compute_accels(vector3** h_accels) {

    // create that accleration matrix on the GPU
    vector3** d_accels;
    cudaMalloc((void **)&d_accels, sizeof(vector3*) * NUMENTITIES); // array of rows (arrays)

    for (int i = 0; i < NUMENTITIES; i++) {
        vector3* d_accel_row;

        // allocate space for the row on gpu
        cudaMalloc((void **)&d_accel_row, sizeof(vector3) * NUMENTITIES);

        // copy values from host row to that row in gpu
        cudaMemcpy(d_accel_row, h_accels[i], sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);

        // put that row into the matrix on the gpu
        cudaMemcpy(d_accels+i, &d_accel_row, sizeof(vector3*), cudaMemcpyHostToDevice);
    }

    // create the positions array on the GPU
    cudaMalloc((void **)&d_hPos, sizeof(vector3) * NUMENTITIES);
    cudaMemcpy(d_hPos, hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);

    // create the mass array on the GPU
    cudaMalloc((void **)&d_mass, sizeof(double) * NUMENTITIES);
    cudaMemcpy(d_mass, mass, sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice);

    dim3 blockSize = dim3(BLOCK_WIDTH, BLOCK_WIDTH, 1);
    dim3 gridSize = dim3(ceil((float)NUMENTITIES / (float)BLOCK_WIDTH), ceil((float)NUMENTITIES / (float)BLOCK_WIDTH), 1);

    printf("grid size x:%d y:%d z:%d\n", gridSize.x, gridSize.y, gridSize.z);
    printf("block size x:%d y:%d z:%d\n", blockSize.x, blockSize.y, blockSize.z);

    // call the kernel
    gpu_compute_accels<<<gridSize, blockSize>>>(d_accels, d_hPos, d_mass);

    // copy the gpu acceleration matrix back to the host acceleration matrix
    for (int i = 0; i < NUMENTITIES; i++) {
        vector3* d_accel_row;

        // copy the address of the row on the GPU back to CPU
        cudaMemcpy(&d_accel_row, d_accels+i, sizeof(vector3*), cudaMemcpyDeviceToHost);
        
        // copy the data in the row from the GPU to the row on CPU matrix
        cudaMemcpy(h_accels[i], d_accel_row, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
    }

    // print out the matrix after GPU operation
    #ifdef DEBUG
    for (int i = 0; i < NUMENTITIES; i++) {
        for(int j = 0; j < NUMENTITIES; j++) {
            printf("%32.32f\n", h_accels[i][j][0]);
        }
        printf("\n");
    }
    printf("\n");
    #endif
}

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){

    printf("In compute.cu\n");

	// make an acceleration matrix which is NUMENTITIES squared in size
	vector3* h_values=(vector3*)malloc(sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	vector3** h_accels=(vector3**)malloc(sizeof(vector3*)*NUMENTITIES);
	for (int i = 0; i < NUMENTITIES; i++)
		h_accels[i]=&h_values[i*NUMENTITIES];

    for (int i = 0; i < NUMENTITIES; i++) {
        for(int j = 0; j < NUMENTITIES; j++) {
            for(int k = 0; k < VECTOR_SIZE; k++) {
                h_accels[i][j][k] = 0.0;
            }
        }
    }

    // allocate device memory of velocity, position, and mass
	cudaMalloc((void **)&d_hVel, sizeof(vector3) * NUMENTITIES);
    cudaMemcpy(d_hVel, hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);

    compute_accels(h_accels);

/*     for (int i = 0; i < NUMENTITIES; i++) {
        for(int j = 0; j < NUMENTITIES; j++) {
            for(int k = 0; k < VECTOR_SIZE; k++) {
                printf("%32.32f\n", h_accels[i][j][0]);
            }
        }
    } */

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
