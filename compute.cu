#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"

#include<stdio.h>

#define THREADS_PER_BLOCK 256

__global__ void gpu_compute_accels(vector3** accels, vector3* positions, double* mass) {

    // the entity being accelerated
    int entityIndex = blockIdx.x;

    // the entity causing the acceleration
    int otherEntityIndex = blockIdx.y * THREADS_PER_BLOCK + threadIdx.x;

    // if there are more threads in a block than there are entities, it would be possible for them to go out of bounds
    if(otherEntityIndex < NUMENTITIES) {
        accels[entityIndex][otherEntityIndex][0] = blockIdx.y * THREADS_PER_BLOCK + threadIdx.x;

        // the old serial calculations with indexes plugged in in place of i and j
        // further parallelizing to split the k loop among 3 threads would be cool but 
        // I already have 2 other programming assignments on the same scale as this one due in three days!
        if (entityIndex == otherEntityIndex) {
            FILL_VECTOR(accels[entityIndex][otherEntityIndex],0,0,0);
        }
        else{
            vector3 distance;
            for (int k = 0; k < 3; k++) distance[k]=positions[entityIndex][k]-positions[otherEntityIndex][k];
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
    cudaMemcpy(d_mass, mass, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);

    // each entity needs to compute a vector for each entity, so we have to set up our grid like this
    // imagine we have 80 entities to compute accels for. By nature of the accel calculation we need to compute the accel of each entity on each entity
    // now imagine that we only can do 25 accels for each block. Each entity needs to compute accels for 80/25 rounded up.
    // so we have an 80 x ceil(80x25) grid of blocks
    // 80 = NUM ENTITIES | 25 = THREADS PER BLOCK
    dim3 gridSize = dim3(NUMENTITIES, ceil((float)NUMENTITIES / (float)THREADS_PER_BLOCK), 1);

    // call the kernel
    gpu_compute_accels<<<gridSize, NUMENTITIES>>>(d_accels, d_hPos, d_mass);

    //printf("grid x: %d | grid y: %d | grid z: %d | TPB: %d\n", gridSize.x, gridSize.y, gridSize.z, THREADS_PER_BLOCK);

    // copy the gpu acceleration matrix back to the host acceleration matrix
    for (int i = 0; i < NUMENTITIES; i++) {
        vector3* d_accel_row;

        // copy the address of the row on the GPU back to CPU
        cudaMemcpy(&d_accel_row, d_accels+i, sizeof(vector3*), cudaMemcpyDeviceToHost);
        
        // copy the data in the row from the GPU to the row on CPU matrix
        cudaMemcpy(h_accels[i], d_accel_row, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
    }

    // print out the matrix after GPU operation
    for (int i = 0; i < NUMENTITIES; i++) {
        for(int j = 0; j < NUMENTITIES; j++) {
            printf("%1.1f ", h_accels[i][j][0]);
        }
        printf("\n");
    }
    printf("\n");
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
            h_accels[i][j][0] = 0.0;
        }
    }

    // allocate device memory of velocity, position, and mass
	cudaMalloc((void **)&d_hVel, sizeof(vector3) * NUMENTITIES);
    cudaMemcpy(d_hVel, hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);

    compute_accels(h_accels);

    //first compute the pairwise accelerations.  Effect is on the first argument.
/* 	for (int i = 0; i < NUMENTITIES; i++){
		for (int j = 0; j < NUMENTITIES; j++){
			if (i==j) {
				FILL_VECTOR(h_accels[i][j],0,0,0);
			}
			else{
				vector3 distance;
				for (int k = 0; k < 3; k++) distance[k]=hPos[i][k]-hPos[j][k];
				double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
				double magnitude=sqrt(magnitude_sq);
				double accelmag=-1*GRAV_CONSTANT*mass[j]/magnitude_sq;
				FILL_VECTOR(h_accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
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
