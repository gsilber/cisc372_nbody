#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"

#include <stdio.h>

__global__ void test_kernel(vector3 *accels, vector3* pos, double* mass) {
    //int threadIndex = threadIdx.x;

    accels[threadIdx.x * NUMENTITIES + threadIdx.y][threadIdx.z] = threadIdx.z;

/*     vel[threadIdx.x][0] = vel[threadIdx.x][0] + 1.0;
    pos[threadIdx.x][0] = pos[threadIdx.x][0] + 2.0;
    mass[threadIdx.x] = mass[threadIdx.x] + 3.0; */
}

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){

    dim3 blockSize = dim3(16, 16, 3);
    dim3 gridSize = dim3(1, 1, 1);

    test_kernel<<<gridSize, blockSize>>>(d_hAccels, d_hPos, d_hmass);

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA API call failed: %s\n", cudaGetErrorString(cudaStatus));
        // Handle the error, throw an exception, or take appropriate action
    }

    cudaMemcpy(hAccels, d_hAccels, sizeof(vector3) * NUMENTITIES * NUMENTITIES, cudaMemcpyDeviceToHost);

    for(int i = 0; i < NUMENTITIES * NUMENTITIES; i ++) {
        
        if(i % NUMENTITIES == 0)
            printf("\n");

        printf("%.1f ", hAccels[i][2]);
    }
    printf("\n");

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