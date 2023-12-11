#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"

#include <stdio.h>
#define SUMDEBUG 1


#define BLOCK_WIDTH_ACCELS 16

#define SUM_TOTAL_THREADS 1

// DO NOT CHANGE THE VECTOR SIZE
#define VECTORSIZE 3

__global__ void compute_accels(vector3 *accels, vector3* pos, double* mass) {

    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;

    __shared__ double distances[BLOCK_WIDTH_ACCELS][BLOCK_WIDTH_ACCELS][VECTORSIZE];

    if(i < NUMENTITIES && j < NUMENTITIES) {
        distances[threadIdx.x][threadIdx.y][threadIdx.z] = pos[i][threadIdx.z] - pos[j][threadIdx.z];
    }
    __syncthreads();

    if(i < NUMENTITIES && j < NUMENTITIES) {

        if (i == j) {
            accels[i * NUMENTITIES + j][threadIdx.z] = 0.0;
        }
        else{
            double magnitude_sq = ( 
                distances[threadIdx.x][threadIdx.y][0] * distances[threadIdx.x][threadIdx.y][0] + 
                distances[threadIdx.x][threadIdx.y][1] * distances[threadIdx.x][threadIdx.y][1] + 
                distances[threadIdx.x][threadIdx.y][2] * distances[threadIdx.x][threadIdx.y][2]
            );

            double magnitude = sqrt(magnitude_sq);
            double accelmag =- 1 * GRAV_CONSTANT * mass[j] / magnitude_sq;
            accels[i * NUMENTITIES + j][threadIdx.z] = accelmag * distances[threadIdx.x][threadIdx.y][threadIdx.z] / magnitude;
        }
    }
}

__global__ void sumOneVectorComponentPerBlock(vector3* gArr, vector3* out) {
    
    int thIdx = threadIdx.x;
    int blIdx = blockIdx.x;
    int vIdx = blockIdx.y;

    int gIdx = (blIdx * NUMENTITIES) + thIdx;



    __shared__ double shArr[SUM_TOTAL_THREADS * 2];
    __shared__ int offset;

    shArr[thIdx] = thIdx < NUMENTITIES ? gArr[gIdx][vIdx] : 0;

    if(thIdx == 0) {
        offset = SUM_TOTAL_THREADS;
    }
    __syncthreads();

    while (offset < NUMENTITIES) {
        //fill the 2nd half with new data from global (0s if we run out of new data)
        shArr[thIdx + NUMENTITIES] = thIdx + offset < NUMENTITIES ? gArr[gIdx + offset][vIdx] : 0;
        __syncthreads();

        if(thIdx == 0) {
            offset += SUM_TOTAL_THREADS;
        }

        double sum = shArr[2 * thIdx] + shArr[(2 * thIdx) + 1];
        __syncthreads();

        shArr[thIdx] = sum;
    }
    __syncthreads();

    for(int stride = 1; stride < SUM_TOTAL_THREADS; stride*=2) {
        __syncthreads();
        int arrIdx = thIdx * stride * 2;
        if(arrIdx + stride < SUM_TOTAL_THREADS) {
            shArr[arrIdx] += shArr[arrIdx + stride];
        }
    }

    if (thIdx == 0) {
        //out[blIdx][vIdx] = shArr[0];
        out[blockIdx.x][blockIdx.y] = blockDim.x;
    }

    return;

    if (thIdx == 0) {
        out[blIdx][vIdx] = shArr[0];
        out[blockIdx.x][blockIdx.y] = blockDim.x;
    }
}

__global__ void sumOneVectorPerBlock(vector3 *gArr, vector3 *out, int arraySize) {

    __shared__ vector3 shArr[SUM_TOTAL_THREADS * 2];
    __shared__ int offset;

    shArr[threadIdx.x][blockIdx.y] = threadIdx.x < arraySize ? gArr[blockIdx.x * arraySize + threadIdx.x][blockIdx.y] : 0;

    if (threadIdx.x == 0)
        offset = blockDim.x;
    __syncthreads();

    while (offset < arraySize) {

        shArr[threadIdx.x + SUM_TOTAL_THREADS][blockIdx.y] = threadIdx.x + offset < arraySize ? gArr[blockIdx.x * arraySize + threadIdx.x + offset][blockIdx.y] : 0;
        __syncthreads();

        if (threadIdx.x == 0)
            offset += SUM_TOTAL_THREADS;
 
        double sum = shArr[2 * threadIdx.x][blockIdx.y] + shArr[2 * threadIdx.x + 1][blockIdx.y];
        __syncthreads();

        shArr[threadIdx.x][blockIdx.y] = sum;
    }
    __syncthreads();

    for (int stride = 1; stride < SUM_TOTAL_THREADS; stride *= 2) {
        __syncthreads();
        int arrIdx = threadIdx.x * stride * 2;
        if (arrIdx + stride < SUM_TOTAL_THREADS) {
            shArr[arrIdx][blockIdx.y] += shArr[arrIdx + stride][blockIdx.y];
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        out[blockIdx.x][blockIdx.y] = shArr[0][blockIdx.y];
        out[blockIdx.x][blockIdx.y] = blockDim.x;
    }
}

__global__ void advance_time(vector3* accel, vector3* vel, vector3* pos) {
    vel[blockIdx.x][threadIdx.x] += accel[blockIdx.x][threadIdx.x]*INTERVAL;
    pos[blockIdx.x][threadIdx.x] += vel[blockIdx.x][threadIdx.x]*INTERVAL;
}

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){

    dim3 blockSize = dim3(16, 16, 3);
    dim3 gridSize = dim3(ceil((double)NUMENTITIES / (double)blockSize.x), ceil((double)NUMENTITIES / (double)blockSize.y), 1);

    compute_accels<<<gridSize, blockSize>>>(d_hAccels, d_hPos, d_hmass);

    /* 
    cudaMemcpy(hAccels, d_hAccels, sizeof(vector3) * NUMENTITIES * NUMENTITIES, cudaMemcpyDeviceToHost);
    for(int i = 0; i < NUMENTITIES * NUMENTITIES; i ++) {
        
        if(i % NUMENTITIES == 0) {
            printf("\n");
        }
        printf("%.32f\n", hAccels[i][0]);
    }
    printf("\n"); 
    */

    blockSize = dim3(SUM_TOTAL_THREADS, 1, 1);
    gridSize = dim3(NUMENTITIES, VECTORSIZE, 1);

    cudaDeviceSynchronize();
    //sumOneVectorPerBlock<<<gridSize, blockSize>>>(d_hAccels, d_output, NUMENTITIES);
    sumOneVectorComponentPerBlock<<<gridSize, blockSize>>>(d_hAccels, d_output);
    cudaDeviceSynchronize();

    #ifdef SUMDEBUG

    vector3 *h_output = (vector3*)malloc(sizeof(vector3) * NUMENTITIES);

    for(int i = 0; i < NUMENTITIES; i++) {
        h_output[i][0] = -1.0;
        h_output[i][1] = -1.0;
        h_output[i][2] = -1.0;
    }
    cudaMemcpy(h_output, d_output, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
    for(int i = 0; i < NUMENTITIES; i++) {
        printf("%.32f %.32f %.32f\n", h_output[i][0], h_output[i][1], h_output[i][2]);
    }

    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cudaError));
    }
    #endif

    blockSize = dim3(VECTORSIZE, 1, 1);
    gridSize = dim3(NUMENTITIES, 1, 1);

    cudaDeviceSynchronize();
    advance_time<<<gridSize, blockSize>>> (d_output, d_hVel, d_hPos);
}