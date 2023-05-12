#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda.h>
#include <stdio.h>

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
//__global__ void compute(vector3 *hVel,vector3 *hPos,double *mass)
__global__ void compute1(vector3 *d_hVel,vector3 *d_hPos,double *d_mass,vector3* values,vector3** accels){
	//make an acceleration matrix which is NUMENTITIES squared in size;
	int j,k;
	int i =  blockDim.x * blockIdx.x +  threadIdx.x;
	//int i = blockDim.x * blockIdx.x +blockIdx.x + (16*threadIdx.y);
	//vector3* values=(vector3*)malloc(sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	//vector3** accels=(vector3**)malloc(sizeof(vector3*)*NUMENTITIES);
	//printf("bx:%d   by: %d  tx: %d ty: %d\n",blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y);
	//printf("test:%d\n",i);
	if (i < NUMENTITIES){
		accels[i]=&values[i*NUMENTITIES];
	}

	//free(accels);
	//free(values);
}

__global__ void compute2(vector3 *d_hVel,vector3 *d_hPos,double *d_mass,vector3* values,vector3** accels){
	int j,k;
	int i =  blockDim.x * blockIdx.x +  threadIdx.x;
	//first compute the pairwise accelerations.  Effect is on the first argument.
	if (i < NUMENTITIES){
		for (j=0;j<NUMENTITIES;j++){
			//printf("test: %d \n",j);
			if (i==j) {
				FILL_VECTOR(accels[i][j],0,0,0);
			}
			else{
				vector3 distance;
				for (k=0;k<3;k++) distance[k]=d_hPos[i][k]-d_hPos[j][k];
				double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
				double magnitude=sqrt(magnitude_sq);
				double accelmag=-1*GRAV_CONSTANT* d_mass[j]/magnitude_sq;
				FILL_VECTOR(accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);

			}

		}
	}

}


__global__ void compute3(vector3 *d_hVel,vector3 *d_hPos,double *d_mass,vector3* values,vector3** accels){
	int j,k;
	int i =  blockDim.x * blockIdx.x +  threadIdx.x;
		//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	if (i < NUMENTITIES){
		vector3 accel_sum={0,0,0};
		for (j=0;j<NUMENTITIES;j++){
			for (k=0;k<3;k++)
				accel_sum[k]+=accels[i][j][k];
		}
		//compute the new velocity based on the acceleration and time interval
		//compute the new position based on the velocity and time interval
		for (k=0;k<3;k++){
			d_hVel[i][k]+=accel_sum[k]*INTERVAL;
			d_hPos[i][k]=d_hVel[i][k]*INTERVAL;
		}
	}

}