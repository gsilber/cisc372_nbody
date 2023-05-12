/*
	CISC372 Assignment 4: Lost in space 
	Contributors: Patrick Harris, Robert Reardon
	File: cuda.c
*/

#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda_runtime.h>

__global__ void pairwise_acceleration(int num_entities, double* pos_x, double* pos_y, double* pos_z, double* mass, double* accels)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < num_entities && j < num_entities && i != j)
    {
        double distance_x = pos_x[i] - pos_x[j];
        double distance_y = pos_y[i] - pos_y[j];
        double distance_z = pos_z[i] - pos_z[j];
        double magnitude_sq = distance_x * distance_x + distance_y * distance_y + distance_z * distance_z;
        double magnitude = sqrt(magnitude_sq);
        double accelmag = -1 * GRAV_CONSTANT * mass[j] / magnitude_sq;
        accels[i * num_entities + j] = accelmag * distance_x / magnitude;
        accels[i * num_entities + j + num_entities * num_entities] = accelmag * distance_y / magnitude;
        accels[i * num_entities + j + 2 * num_entities * num_entities] = accelmag * distance_z / magnitude;
    }
}

__global__ void row_summation(int num_entities, double* accels, double* vel_x, double* vel_y, double* vel_z, double* pos_x, double* pos_y, double* pos_z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_entities)
    {
        double accel_sum_x = 0;
        double accel_sum_y = 0;
        double accel_sum_z = 0;
        for (int j = 0; j < num_entities; j++)
        {
            accel_sum_x += accels[i * num_entities + j];
            accel_sum_y += accels[i * num_entities + j + num_entities * num_entities];
            accel_sum_z += accels[i * num_entities + j + 2 * num_entities * num_entities];
        }
        vel_x[i] += accel_sum_x * INTERVAL;
        vel_y[i] += accel_sum_y * INTERVAL;
        vel_z[i] += accel_sum_z * INTERVAL;
        pos_x[i] += vel_x[i] * INTERVAL;
        pos_y[i] += vel_y[i] * INTERVAL;
        pos_z[i] += vel_z[i] * INTERVAL;
    }
}

void compute()
{
    double* d_pos_x;
    double* d_pos_y;
    double* d_pos_z;
    double* d_mass;
    double* d_accels;
    double* d_vel_x;
    double* d_vel_y;
    double* d_vel_z;
    int size = NUMENTITIES * sizeof(double);

    // Allocate device memory
    cudaMalloc((void**)&d_pos_x, size);
    cudaMalloc((void**)&d_pos_y, size);
    cudaMalloc((void**)&d_pos_z, size);
    cudaMalloc((void**)&d_mass, size);
    cudaMalloc((void**)&d_accels, size * NUMENTITIES * 3);
    cudaMalloc((void**)&d_vel_x, size);
    cudaMalloc((void**)&d_vel_y, size);
    cudaMalloc((void**)&d_vel_z, size);
    // Copy host memory to device memory
	cudaMemcpy(d_pos_x, hPos[0], size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pos_y, hPos[1], size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pos_z, hPos[2], size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mass, mass, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vel_x, hVel[0], size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vel_y, hVel[1], size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vel_z, hVel[2], size, cudaMemcpyHostToDevice);

	// Define grid and block dimensions for pairwise acceleration computation
	dim3 blockDim(16, 16);
	dim3 gridDim((NUMENTITIES + blockDim.x - 1) / blockDim.x, (NUMENTITIES + blockDim.y - 1) / blockDim.y);

	// Compute pairwise accelerations
	pairwise_acceleration<<<gridDim, blockDim>>>(NUMENTITIES, d_pos_x, d_pos_y, d_pos_z, d_mass, d_accels);

	// Define grid dimensions for row summation
	dim3 gridDim2((NUMENTITIES + blockDim.x - 1) / blockDim.x);

	// Sum up rows to get effect on each entity, then update velocity and position
	row_summation<<<gridDim2, blockDim.x>>>(NUMENTITIES, d_accels, d_vel_x, d_vel_y, d_vel_z, d_pos_x, d_pos_y, d_pos_z);

	// Copy device memory back to host memory
	cudaMemcpy(hPos[0], d_pos_x, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(hPos[1], d_pos_y, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(hPos[2], d_pos_z, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(hVel[0], d_vel_x, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(hVel[1], d_vel_y, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(hVel[2], d_vel_z, size, cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(d_pos_x);
	cudaFree(d_pos_y);
	cudaFree(d_pos_z);
	cudaFree(d_mass);
	cudaFree(d_accels);
	cudaFree(d_vel_x);
	cudaFree(d_vel_y);
	cudaFree(d_vel_z);
}