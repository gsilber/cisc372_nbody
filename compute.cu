#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"

#include<stdio.h>

/* __global__ void compute_accels(vector3* accels) {

    for(int i = 0; i < NUMENTITIES * NUMENTITIES; i++) {
        accels[i][0] += 1.0;
    }
} */

__global__ void compute_accels(vector3** accels) {

    for(int i = 0; i < NUMENTITIES; i++) {
        for(int j = 0; j < NUMENTITIES; j++) {
            accels[i][j][0] += 3.0;
        }
    }
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
            h_accels[i][j][0] = 2.0;
        }
    }

    // create that accleration matrix on the GPU
    vector3* d_values;
    cudaMalloc((void **)&d_values, sizeof(vector3) * NUMENTITIES * NUMENTITIES);

    vector3** d_accels;
    cudaMalloc((void **)&d_accels, sizeof(vector3*) * NUMENTITIES);

    for (int i = 0; i < NUMENTITIES; i++) {
        vector3* d_accel_row;
        cudaMalloc((void **)&d_accel_row, sizeof(vector3) * NUMENTITIES);
        cudaMemcpy(d_accels+i, &d_accel_row, sizeof(vector3*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_accel_row, h_accels[i], sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
    }

    //cudaMemcpy(d_values, h_values, sizeof(vector3) * NUMENTITIES * NUMENTITIES, cudaMemcpyHostToDevice);

    //compute_accels<<<1,1>>>(d_values);
    compute_accels<<<1,1>>>(d_accels);

    printf("got through kernel\n");

    for (int i = 0; i < NUMENTITIES; i++) {
        vector3* d_accel_row;

        cudaMemcpy(&d_accel_row, d_accels+i, sizeof(vector3*), cudaMemcpyDeviceToHost);
        printf("copied the row address back\n");
        cudaMemcpy(h_accels[i], d_accel_row, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
        printf("copied the row data back\n");
    }

//    cudaMemcpy(h_values, d_values, sizeof(vector3) * NUMENTITIES * NUMENTITIES, cudaMemcpyDeviceToHost);
    
/*     for(int i = 0; i < NUMENTITIES * NUMENTITIES; i++) {
        if(i % NUMENTITIES == 0) printf("\n");
        printf("%f ", h_values[i][0]);
    } */


    for (int i = 0; i < NUMENTITIES; i++) {
        for(int j = 0; j < NUMENTITIES; j++) {
            printf("%1.1f ", h_accels[i][j][0]);
        }
        printf("\n");
    }
    printf("\n");

    // allocate device memory of velocity, position, and mass
	cudaMalloc((void **)&d_hVel, sizeof(vector3) * NUMENTITIES);
    cudaMemcpy(d_hVel, hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_hPos, sizeof(vector3) * NUMENTITIES);
    cudaMemcpy(d_hPos, hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_mass, sizeof(double) * NUMENTITIES);
    cudaMemcpy(d_mass, mass, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);

    //compute_accels<<<1,1>>>(d_hPos);
    //cudaMemcpy(hPos, d_hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
	
    //first compute the pairwise accelerations.  Effect is on the first argument.
	for (int i = 0; i < NUMENTITIES; i++){
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
	}

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
