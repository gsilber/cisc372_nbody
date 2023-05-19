#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda_runtime.h>

__global__ void pAccComp(vector3 *hPos, vector3 *accels, double *mass){
	int col = (blockDim.x * blockIdx.x) + threadIdx.x;
	int row = (blockDim.y * blockIdx.y) + threadIdx.y;
	int ind = (NUMENTITIES * row) + col; 
	int i = row; 
	int j = col; 
	if(ind < NUMENTITIES * NUMENTITIES){
		if (i == j){
			FILL_VECTOR(accels[ind], 0, 0, 0);
		} else{
			vector3 dist; 
			for (int k = 0; k < 3; k++){
				dist[k] = hPos[i][k] - hPos[j][k];
			}
			double mag_sq = dist[0] * dist[0] + dist[1] * dist[1] + dist[2] * dist[2];
			double mag = sqrt(mag_sq);
			double acc = -1 * GRAV_CONSTANT * mass[j] /mag_sq;
			FILL_VECTOR(accels[ind], acc * dist[0] / mag, acc * dist[1] / mag, acc * dist[2]/ mag);
		}
	}
} 

__global__ void sum(vector3 *accels, vector3 *accel_sum, vector3 *hPos, vector3 *hVel){
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int i = row;
	if (i < NUMENTITIES){
		FILL_VECTOR(accel_sum[i], 0, 0, 0);
		for (int j = 0; j < NUMENTITIES; j++){
			for (int k = 0; k < 3; k++){
				accel_sum[i][k] += accels[(i * NUMENTITIES) + j][k];
			}
		}
		for (int k = 0; k < 3; k++){
			hVel[i][k] += accel_sum[i][k] * INTERVAL;
			hPos[i][k] = hVel[i][k] * INTERVAL;
		}

	}
}

void compute(){
	vector3 *dhPos, *dhVel, *dacc, *dsum;
	double *dmass;
	int block = ceilf(NUMENTITIES / 16.0f);
	int thread = ceilf(NUMENTITIES / (float) block);
	dim3 gridDim(block, block, 1);
	dim3 blockDim(thread, thread, 1);
	
	cudaMalloc((void**) &dhPos, sizeof(vector3) * NUMENTITIES);
	cudaMalloc((void**) &dhVel, sizeof(vector3) * NUMENTITIES);
	cudaMalloc((void**) &dacc, sizeof(vector3) * NUMENTITIES);
	cudaMalloc((void**) &dsum, sizeof(vector3) * NUMENTITIES);
	cudaMalloc((void**) &dmass, sizeof(double) * NUMENTITIES);

	cudaMemcpy(dhPos, hPos, sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(dhVel, hVel, sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(dmass, mass, sizeof(double)*NUMENTITIES, cudaMemcpyHostToDevice);
	
	pAccComp<<<gridDim, blockDim>>>(dhPos, dacc, dmass);
	cudaDeviceSynchronize();

	sum<<<gridDim.x, blockDim.x>>>(dacc, dsum, dhPos, dhVel);
	cudaDeviceSynchronize();

	cudaMemcpy(hPos, dhPos, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);
	cudaMemcpy(hVel, dhVel, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);

	cudaFree(dhPos);
	cudaFree(dhVel);
	cudaFree(dmass);
	cudaFree(dacc);

}
