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

