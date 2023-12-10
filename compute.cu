#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"

#include <stdio.h>

#define BLOCKWIDTH 16
#define THREADS_PER_BLOCK BLOCKWIDTH*BLOCKWIDTH

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

__global__ void sumOneVectorPerBlock(vector3 *gArr, vector3 *out, int arraySize) {

    __shared__ vector3 shArr[THREADS_PER_BLOCK * 2];
    __shared__ int offset;

    shArr[threadIdx.x][blockIdx.y] = threadIdx.x < arraySize ? gArr[blockIdx.x * arraySize + threadIdx.x][blockIdx.y] : 0;

    if (threadIdx.x == 0)
        offset = blockDim.x;
    __syncthreads();

    while (offset < arraySize) {

        shArr[threadIdx.x + THREADS_PER_BLOCK][blockIdx.y] = threadIdx.x + offset < arraySize ? gArr[blockIdx.x * arraySize + threadIdx.x + offset][blockIdx.y] : 0;
        __syncthreads();

        if (threadIdx.x == 0)
            offset += THREADS_PER_BLOCK;

        vector3 sum; 
        sum[blockIdx.y] = shArr[2 * threadIdx.x][blockIdx.y] + shArr[2 * threadIdx.x + 1][blockIdx.y];
        
        __syncthreads();
        shArr[threadIdx.x][blockIdx.y] = sum[blockIdx.y];
        shArr[threadIdx.x][1] = sum[1];
        shArr[threadIdx.x][2] = sum[2];
        __syncthreads();
    }
    __syncthreads();

    for (int stride = 1; stride < THREADS_PER_BLOCK; stride *= 2) {
        __syncthreads();
        int arrIdx = threadIdx.x * stride * 2;
        if (arrIdx + stride < THREADS_PER_BLOCK) {
            shArr[arrIdx][blockIdx.y] += shArr[arrIdx + stride][blockIdx.y];
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        out[blockIdx.x][blockIdx.y] = shArr[0][blockIdx.y];
    }
}

// d_input is our array of values to sum (IT'S A 2D ARRAY IN 1D FORM)
// d_output is our array of sums
// arraySize is both the number of sums we have to compute and the number of values that will be summed 
// it's both because it's a square
void sumAccelerations(vector3 *d_input, vector3 *d_output, int arraySize) {

    dim3 gridSize = dim3(arraySize, VECTORSIZE, 1);
    dim3 blockSize = dim3(THREADS_PER_BLOCK, 1, 1);

    sumOneVectorPerBlock<<<gridSize, blockSize>>>(d_input, d_output, arraySize);

    cudaDeviceSynchronize();
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaError));
    }
}

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){

    dim3 blockSize = dim3(BLOCKWIDTH, BLOCKWIDTH, 3);
    dim3 gridSize = dim3(ceil((double)NUMENTITIES / (double)blockSize.x), ceil((double)NUMENTITIES / (double)blockSize.y), 1);

    //printf("NUMENTITIES = %d | blockSize.x = %d | NUMENTITIES/BLOCKSIZE = %f | CEIL = %f\n", NUMENTITIES, blockSize.x, (double) NUMENTITIES / (double) blockSize.x, ceil((double) NUMENTITIES / (double) blockSize.x));
    //printf("gridSize.x: %d | gridSize.y %d\n", gridSize.x, gridSize.y);

    test_kernel<<<gridSize, blockSize>>>(d_hAccels, d_hPos, d_hmass);

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA API call failed: %s\n", cudaGetErrorString(cudaStatus));
        // Handle the error, throw an exception, or take appropriate action
    }

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

    /* 
    blockSize = dim3(BLOCKWIDTH, BLOCKWIDTH, 3);

    int testArrSize = 5;

    vector3 testOutput;
    vector3 testArr[testArrSize];

    for(int i = 0; i < testArrSize; i++) {
        testArr[i][0] = i * 1.0;
        testArr[i][1] = i * 2.0;
        testArr[i][2] = i * 3.0;
    }

    sumOneVectorPerBlock<<<1, blockSize>>>(testArr, testArrSize); */

    // START TEST ZONE
    /* int size = 300;
    
    vector3 *h_input = (vector3*)malloc(sizeof(vector3) * size * size);
    for(int i = 0; i < size * size; i++) {
        h_input[i][0] = (double)i * 0.00000000001;
        h_input[i][1] = (double)i*0.00000000002;
        h_input[i][2] = (double)i*0.00000000003;
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

    sumAccelerations(d_input, d_output, size);

    cudaMemcpy(h_output, d_output, sizeof(vector3) * size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; i++) {
        printf("%32.32f %32.32f %32.32f\n", h_output[i][0], h_output[i][1], h_output[i][2]);
    } 
    */

    /*     
    vector3 *d_output;
    cudaMalloc((void **)&d_output, sizeof(vector3) * NUMENTITIES);

    vector3 *h_output = (vector3*)malloc(sizeof(vector3) * NUMENTITIES);
    for(int i = 0; i < NUMENTITIES; i++) {
        h_output[i][0] = 0.0;
        h_output[i][1] = 0.0;
        h_output[i][2] = 0.0;
    }

    sumAccelerations(d_hAccels, d_output, NUMENTITIES);
    cudaMemcpy(h_output, d_output, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
    for (int i = 0; i < NUMENTITIES; i++) {
        printf("%32.32f %32.32f %32.32f\n", h_output[i][0], h_output[i][1], h_output[i][2]);
    } 
    */
    // END TEST ZONE

    //printf("before cudaMemcpy\n");
    cudaMemcpy(hAccels, d_hAccels, sizeof(vector3) * NUMENTITIES * NUMENTITIES, cudaMemcpyDeviceToHost);
   //printf("after cudaMemcpy\n");
	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	for (int i=0;i<NUMENTITIES;i++){
        //printf("i is %d\n", i);
		vector3 accel_sum={0,0,0};
		for (int j=0;j<NUMENTITIES;j++){
			for (int k=0;k<3;k++)
				accel_sum[k]+=hAccels[i*NUMENTITIES + j][k];
		}
		//compute the new velocity based on the acceleration and time interval
		//compute the new position based on the velocity and time interval
		for (int k=0;k<3;k++){
            //printf("k is %d\n", k);
			hVel[i][k]+=accel_sum[k]*INTERVAL;
			hPos[i][k]+=hVel[i][k]*INTERVAL;
		}
	}
    //printf("after the loops?\n");
	//free(hAccels);
	//free(values);
}