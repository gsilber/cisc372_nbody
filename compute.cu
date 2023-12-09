#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"

#include<stdio.h>

#undef DEBUG
//#define DEBUG 1

#define THREADS_PER_BLOCK 256
#define VECTOR_SIZE 3
#define BLOCK_WIDTH 16

__global__ void gpu_compute_accels(vector3* accels, vector3* hPos, double* mass) {

    // shared memory hopefully speeds things up
    __shared__ double shared_mass[BLOCK_WIDTH];
    __shared__ vector3 shared_hPos_x[BLOCK_WIDTH];
    __shared__ vector3 shared_hPos_y[BLOCK_WIDTH];

    // the entity being accelerated
    int entityIndex = blockIdx.x * BLOCK_WIDTH + threadIdx.x;

    // the entity causing the acceleration
    int otherEntityIndex = blockIdx.y * BLOCK_WIDTH + threadIdx.y;

    if (threadIdx.x == 0) {
        shared_mass[threadIdx.y] = mass[otherEntityIndex];
        shared_hPos_x[threadIdx.y][0] = hPos[otherEntityIndex][0];
        shared_hPos_x[threadIdx.y][1] = hPos[otherEntityIndex][1];
        shared_hPos_x[threadIdx.y][2] = hPos[otherEntityIndex][2];
    }
    if (threadIdx.y == 0) {
        shared_hPos_y[threadIdx.x][0] = hPos[entityIndex][0];
        shared_hPos_y[threadIdx.x][1] = hPos[entityIndex][1];
        shared_hPos_y[threadIdx.x][2] = hPos[entityIndex][2];
    }

    __syncthreads();

    if(entityIndex < NUMENTITIES && otherEntityIndex < NUMENTITIES) {

        if (entityIndex == otherEntityIndex) {
				FILL_VECTOR(accels[entityIndex * NUMENTITIES + otherEntityIndex],0,0,0);
			}
        else{
            vector3 distance;
            for (int k=0;k<3;k++) distance[k]=shared_hPos_y[threadIdx.x][k]-shared_hPos_x[threadIdx.y][k];
            double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
            double magnitude=sqrt(magnitude_sq);
            double accelmag=-1*GRAV_CONSTANT*shared_mass[threadIdx.y]/magnitude_sq;
            FILL_VECTOR(accels[entityIndex * NUMENTITIES + otherEntityIndex],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
        }
    }
}

void compute_accels(vector3* h_values, vector3* d_values) {

    dim3 blockSize = dim3(BLOCK_WIDTH, BLOCK_WIDTH, 1);
    dim3 gridSize = dim3(ceil((float)NUMENTITIES / (float)BLOCK_WIDTH), ceil((float)NUMENTITIES / (float)BLOCK_WIDTH), 1);

    // call the kernel
    gpu_compute_accels<<<gridSize, blockSize>>>(d_values, d_hPos, d_mass);

    #ifdef DEBUG
    // copy the gpu acceleration matrix back to the host acceleration matrix
    cudaMemcpy(h_values, d_values, sizeof(vector3) * NUMENTITIES * NUMENTITIES, cudaMemcpyDeviceToHost);

    printf("GRID SIZE: %d\n", gridSize.x);

    // print out the matrix after GPU operation
    for (int i = 0; i < NUMENTITIES * NUMENTITIES; i++) {
        printf("%32.32f\n", h_values[i][0]);
        if(i % NUMENTITIES == 0)
            printf("\n");        
    }
    printf("\n");
    #endif
}

// from lecture slides could be useful
__global__ void sumNoncommSingleBlock(int *gArr, int *out, int arraySize) {
    int thIdx = threadIdx.x;
    __shared__ int shArr[THREADS_PER_BLOCK*2];
    __shared__ int offset;

    shArr[thIdx] = thIdx < arraySize ? gArr[thIdx] : 0;

    if(thIdx == 0)
        offset = THREADS_PER_BLOCK;
    __syncthreads();

    while(offset < arraySize) {
        shArr[thIdx + THREADS_PER_BLOCK] = thIdx + offset < arraySize ? gArr[thIdx] : 0;
        __syncthreads();
        if(thIdx == 0)
            offset += THREADS_PER_BLOCK;
        
        int sum = shArr[2*thIdx] + shArr[2*thIdx+1];
        __syncthreads();
        shArr[thIdx] = sum;
    }
    __syncthreads();

    for(int stride = 1; stride < THREADS_PER_BLOCK; stride *= 2) {
        int arrIdx = thIdx*stride*2;
        if(arrIdx + stride < THREADS_PER_BLOCK)
            shArr[arrIdx] += shArr[arrIdx + stride];
        __syncthreads();
    }
    if(thIdx == 0)
        *out = shArr[0];
}

__global__ void sumOneVectorPerBlock(vector3 *gArr, vector3 *out, int arraySize) {
    int thIdx = threadIdx.x;
    int bIdx = blockIdx.x;

    __shared__ vector3 shArr[THREADS_PER_BLOCK * 2];
    __shared__ int offset;

    shArr[thIdx][0] = thIdx < arraySize ? gArr[bIdx * arraySize + thIdx][0] : 0;
    shArr[thIdx][1] = thIdx < arraySize ? gArr[bIdx * arraySize + thIdx][1] : 0;
    shArr[thIdx][2] = thIdx < arraySize ? gArr[bIdx * arraySize + thIdx][2] : 0;

    if (thIdx == 0)
        offset = blockDim.x;
    __syncthreads();

    while (offset < arraySize) {

        shArr[thIdx + THREADS_PER_BLOCK][0] = thIdx + offset < arraySize ? gArr[bIdx * arraySize + thIdx + offset][0] : 0;
        shArr[thIdx + THREADS_PER_BLOCK][1] = thIdx + offset < arraySize ? gArr[bIdx * arraySize + thIdx + offset][1] : 0;
        shArr[thIdx + THREADS_PER_BLOCK][2] = thIdx + offset < arraySize ? gArr[bIdx * arraySize + thIdx + offset][2] : 0;
        __syncthreads();

        if (thIdx == 0)
            offset += THREADS_PER_BLOCK;

        vector3 sum; 
        sum[0] = shArr[2 * thIdx][0] + shArr[2 * thIdx + 1][0];
        sum[1] = shArr[2 * thIdx][1] + shArr[2 * thIdx + 1][1];
        sum[2] = shArr[2 * thIdx][2] + shArr[2 * thIdx + 1][2];
        
        __syncthreads();
        shArr[thIdx][0] = sum[0];
        shArr[thIdx][1] = sum[1];
        shArr[thIdx][2] = sum[2];
    }
    __syncthreads();

    for (int stride = 1; stride < THREADS_PER_BLOCK; stride *= 2) {
        int arrIdx = thIdx * stride * 2;
        if (arrIdx + stride < THREADS_PER_BLOCK) {
            shArr[arrIdx][0] += shArr[arrIdx + stride][0];
            shArr[arrIdx][1] += shArr[arrIdx + stride][1];
            shArr[arrIdx][2] += shArr[arrIdx + stride][2];
        }
        __syncthreads();
    }

    if (thIdx == 0) {
        out[bIdx][0] = shArr[0][0];
        out[bIdx][1] = shArr[0][1];
        out[bIdx][2] = shArr[0][2];
    }
}

void sumAccelerations(vector3 *d_input, vector3 *d_output, int arraySize, int numVectors) {

    dim3 gridSize = dim3(numVectors, 1, 1);
    dim3 blockSize = dim3(THREADS_PER_BLOCK, 1, 1);

    sumOneVectorPerBlock<<<gridSize, blockSize>>>(d_input, d_output, arraySize);

    cudaDeviceSynchronize();
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaError));
    }
}

__global__ void gpu_advance_simulation(vector3* d_values, vector3* hVel, vector3* hPos) {
    
    int entityIndex = (blockIdx.x * blockDim.x + threadIdx.x) * NUMENTITIES;

    for(int i = entityIndex + 1; i < entityIndex + NUMENTITIES; i++) {
        d_values[entityIndex][0] += d_values[i][0];
        d_values[entityIndex][1] += d_values[i][1];
        d_values[entityIndex][2] += d_values[i][2];
    }

    for (int k = 0; k < 3; k++){
        hVel[blockIdx.x * blockDim.x + threadIdx.x][k]+=d_values[entityIndex][k]*INTERVAL;
        hPos[blockIdx.x * blockDim.x + threadIdx.x][k]+=hVel[blockIdx.x * blockDim.x + threadIdx.x][k]*INTERVAL;
    }
}

void advance_simulation(vector3* h_values, vector3* d_values, vector3* d_hVel, vector3* d_hPos) {

    dim3 blockSize = dim3(256, 1, 1);
    dim3 gridSize = dim3(ceil((float)NUMENTITIES / 256), 1, 1);

    gpu_advance_simulation<<<gridSize, blockSize>>>(d_values, d_hVel, d_hPos);

    cudaMemcpy(hPos, d_hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, d_hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);

    #ifdef DEBUG
    printf("GRID SIZE: %d\n", gridSize.x);
    // copy the gpu acceleration matrix back to the host acceleration matrix
    cudaMemcpy(h_values, d_values, sizeof(vector3) * NUMENTITIES * NUMENTITIES, cudaMemcpyDeviceToHost);

    // print out the matrix after GPU operation
    for (int i = 0; i < NUMENTITIES; i+=1) {
        printf("%10.32f\n",hVel[i][0]);
        if (i % NUMENTITIES == 0) {
            //printf("%32.32f\n", h_values[i][0]);
            //printf("%32.1f %32.1f %32.32f\n", h_values[i][0], h_values[i][1], h_values[i][2]);
        }
    }
    printf("\n");
    #endif
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
            for(int k = 0; k < VECTOR_SIZE; k++) {
                h_accels[i][j][k] = 2.0;
            }
        }
    }

    // create that accleration "matrix" on the GPU
    vector3* d_values;
    cudaMalloc((void **)&d_values, sizeof(vector3) * NUMENTITIES * NUMENTITIES);

    // create the positions array on the GPU
    cudaMalloc((void **)&d_hPos, sizeof(vector3) * NUMENTITIES);
    cudaMemcpy(d_hPos, hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);

    // create the mass array on the GPU
    cudaMalloc((void **)&d_mass, sizeof(double) * NUMENTITIES);
    cudaMemcpy(d_mass, mass, sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice);

    // create the velocity array on the GPU
	cudaMalloc((void **)&d_hVel, sizeof(vector3) * NUMENTITIES);
    cudaMemcpy(d_hVel, hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);

    //compute_accels(h_values, d_values);
    //advance_simulation(h_values, d_values, d_hVel, d_hPos);

    // START TEST ZONE
    int size = 6;
    
    vector3 *h_input = (vector3*)malloc(sizeof(vector3) * size * size);
    for(int i = 0; i < size * size; i++) {
        h_input[i][0] = (double)i;
        h_input[i][1] = (double)i;
        h_input[i][2] = (double)i;
    }

    vector3 *h_output = (vector3*)malloc(sizeof(vector3) * size);
    for(int i = 0; i < size; i++) {
        h_output[i][0] = 0.0;
        h_output[i][1] = 0.0;
        h_output[i][2] = 0.0;
    }

    vector3 *d_input;
    cudaMalloc((void **)&d_input, sizeof(vector3) * size * size);
    cudaMemcpy(d_input, h_input, sizeof(vector3) * size * size, cudaMemcpyHostToDevice);

    vector3 *d_output;
    cudaMalloc((void **)&d_output, sizeof(vector3) * size);

    sumAccelerations(d_input, d_output, size, size*size);

    cudaMemcpy(h_output, d_output, sizeof(vector3) * size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; i++) {
        printf("%1.1f %1.1f %1.1f\n", h_output[i][0], h_output[i][1], h_output[i][2]);
    }

    // END TEST ZONE

    free(h_accels);
	free(h_values);

    cudaFree(d_values);
    cudaFree(d_hVel);
    cudaFree(d_hPos);
    cudaFree(d_mass);
}
