/*
	CISC372 Assignment 4: Lost in space 
	Contributors: Patrick Harris, Robert Reardon
	File: cuda.h
*/

#ifndef CUDA_H
#define CUDA_H

#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Constants
#define GRAV_CONSTANT 6.67430e-11
#define NUMENTITIES 100
#define INTERVAL 0.01

// Functions
__global__ void pairwise_acceleration(int num_entities, double* pos_x, double* pos_y, double* pos_z, double* mass, double* accels);
__global__ void row_summation(int num_entities, double* accels, double* vel_x, double* vel_y, double* vel_z, double* pos_x, double* pos_y, double* pos_z);
//void compute();

#endif /* CUDA_H */