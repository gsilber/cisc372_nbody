#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"

#include <stdio.h>

#define BLOCKWIDTH 8

// DO NOT CHANGE THE VECTOR SIZE
#define VECTORSIZE 3

__global__ void test_kernel(vector3 *accels, vector3* pos, double* mass) {

    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;

    __shared__ double distances[BLOCKWIDTH][BLOCKWIDTH][VECTORSIZE];

    if(i < NUMENTITIES && j < NUMENTITIES) {
        distances[threadIdx.x][threadIdx.y][threadIdx.z] = pos[i][threadIdx.z] - pos[j][threadIdx.z];
    }
    __syncthreads();

    if(i < NUMENTITIES && j < NUMENTITIES) {
        //accels[i * NUMENTITIES + j][threadIdx.z] = i;

        if (i == j) {
            accels[i * NUMENTITIES + j][threadIdx.z] = 0.0;
        }
        else{
  /*           double dx = pos[i][0] - pos[j][0];
            double dy = pos[i][1] - pos[j][1];
            double dz = pos[i][2] - pos[j][2]; */

            //double magnitude_sq = dx * dx + dy * dy + dz * dz;
            double magnitude_sq = ( 
                distances[threadIdx.x][threadIdx.y][0] * distances[threadIdx.x][threadIdx.y][0] + 
                distances[threadIdx.x][threadIdx.y][1] * distances[threadIdx.x][threadIdx.y][1] + 
                distances[threadIdx.x][threadIdx.y][2] * distances[threadIdx.x][threadIdx.y][2]
            );

            double magnitude = sqrt(magnitude_sq);
            double accelmag =- 1 * GRAV_CONSTANT * mass[j] / magnitude_sq;
            //accels[i * NUMENTITIES + j][threadIdx.z] = accelmag * dx / magnitude; //I SHOULD NOT BE DX FOR ALL threadIdx.z
            accels[i * NUMENTITIES + j][threadIdx.z] = accelmag * distances[threadIdx.x][threadIdx.y][threadIdx.z] / magnitude;
            //accels[i * NUMENTITIES + j][threadIdx.z] = distances[threadIdx.x][threadIdx.y][threadIdx.z];
        }
    }

/*     vel[threadIdx.x][0] = vel[threadIdx.x][0] + 1.0;
    pos[threadIdx.x][0] = pos[threadIdx.x][0] + 2.0;
    mass[threadIdx.x] = mass[threadIdx.x] + 3.0; */
}

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){

    dim3 blockSize = dim3(BLOCKWIDTH, BLOCKWIDTH, 3);
    dim3 gridSize = dim3(ceil((double)NUMENTITIES / (double)blockSize.x), ceil((double)NUMENTITIES / (double)blockSize.y), 1);

    printf("NUMENTITIES = %d | blockSize.x = %d | NUMENTITIES/BLOCKSIZE = %f | CEIL = %f\n", NUMENTITIES, blockSize.x, (double) NUMENTITIES / (double) blockSize.x, ceil((double) NUMENTITIES / (double) blockSize.x));
    printf("gridSize.x: %d | gridSize.y %d\n", gridSize.x, gridSize.y);

    test_kernel<<<gridSize, blockSize>>>(d_hAccels, d_hPos, d_hmass);

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA API call failed: %s\n", cudaGetErrorString(cudaStatus));
        // Handle the error, throw an exception, or take appropriate action
    }

    cudaMemcpy(hAccels, d_hAccels, sizeof(vector3) * NUMENTITIES * NUMENTITIES, cudaMemcpyDeviceToHost);

    for(int i = 0; i < NUMENTITIES * NUMENTITIES; i ++) {
        
        if(i % NUMENTITIES == 0) {
            printf("\n");
        }
        printf("%.32f\n", hAccels[i][0]);
    }
    printf("\n");
/*     printf("%.2f ", hAccels[9][0]);
    printf("%.2f ", hAccels[9][1]);
    printf("%.2f ", hAccels[9][2]); */

/*     cudaMemcpy(hVel, d_hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
    cudaMemcpy(hPos, d_hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
    cudaMemcpy(hmass, d_hmass, sizeof(double) * NUMENTITIES, cudaMemcpyDeviceToHost);

    printf("hVel: %.1f | hPos: %.1f | hmass: %.1f\n\n", hVel[1][0], hPos[1][0], hmass[1]); */


/* 	//make an acceleration matrix which is NUMENTITIES squared in size;
	int i,j,k;
	vector3* values=(vector3*)malloc(sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	vector3** accels=(vector3**)malloc(sizeof(vector3*)*NUMENTITIES);
	for (i=0;i<NUMENTITIES;i++)
		accels[i]=&values[i*NUMENTITIES];
	//first compute the pairwise accelerations.  Effect is on the first argument.
	for (i=0;i<NUMENTITIES;i++){
		for (j=0;j<NUMENTITIES;j++){
			if (i==j) {
				FILL_VECTOR(accels[i][j],0,0,0);
			}
			else{
				vector3 distance;
				for (k=0;k<3;k++) distance[k]=hPos[i][k]-hPos[j][k];
				double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
				double magnitude=sqrt(magnitude_sq);
				double accelmag=-1*GRAV_CONSTANT*hmass[j]/magnitude_sq;
				FILL_VECTOR(accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
			}
		}
	}
	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	for (i=0;i<NUMENTITIES;i++){
		vector3 accel_sum={0,0,0};
		for (j=0;j<NUMENTITIES;j++){
			for (k=0;k<3;k++)
				accel_sum[k]+=accels[i][j][k];
		}
		//compute the new velocity based on the acceleration and time interval
		//compute the new position based on the velocity and time interval
		for (k=0;k<3;k++){
			hVel[i][k]+=accel_sum[k]*INTERVAL;
			hPos[i][k]+=hVel[i][k]*INTERVAL;
		}
	}
	free(accels);
	free(values); */
}