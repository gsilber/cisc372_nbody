// THIS CODE IS TAKEN FROM THIS TUTORIAL: https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial01/
// ALL CREDIT GOES TO THIS TUTORIAL

#define N 5000

#include <time.h>
#include <stdio.h>

__global__ void vector_add(float *out, float *a, float *b) {
    
    int threadIndex = threadIdx.x;
    int stride = blockDim.x;
    
    __shared__ float shared_a[N];
    __shared__ float shared_b[N];

    for (int i = threadIndex; i < N; i += stride) {
        shared_a[i] = a[i];
        shared_b[i] = b[i];
    }

    __syncthreads();

    out[threadIndex] = 0.0;

    for(int i = threadIndex; i < N; i += stride) {
        for(int j = 0; j < N; j+= 1) {
            out[i] += shared_a[i] * shared_b[j];
        }
    }
}

int main() {
    
    // keep track of time
    clock_t t0=clock();
	int t_now;

    float *h_a, *h_b, *h_out;
    float *d_a, *d_b, *d_out;

    // Allocate host memory for a and b
    h_a = (float*)malloc(sizeof(float) * N);
    h_b = (float*)malloc(sizeof(float) * N);
    h_out = (float*)malloc(sizeof(float) * N);

    // Initialize values for a, b, and out
    for(int i = 0; i < N; i++) {
        h_a[i] = 1.0 * i;
        h_b[i] = 2.0 * i;
        h_out[i] = 0.0;
    }

    // Print h_out before the GPU does its stuff
    printf("Before kernel call\n");
    
    /* 
    for(int i = 0; i < N; i++) {
        printf("%f\n", h_out[i]);
    }
    */

    // Allocate device memory for a and b
    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N);
    cudaMalloc((void**)&d_out, sizeof(float) * N);

    // Transfer a and b data from host to device memory
    cudaMemcpy(d_a, h_a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Call the kernel
    vector_add<<<1,1024>>>(d_out, d_a, d_b);

    // Transfer output data from device to host memory
    cudaMemcpy(h_out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Print h_out to see the results
    printf("After kernel call\n");
    for(int i = 0; i < N; i++) {
        printf("%f\n", h_out[i]);
    }

    // Cleanup after kernel execution
    cudaFree(d_a);
    free(h_a);
    cudaFree(d_b);
    free(h_b);
    cudaFree(d_out);
    free(h_out);

    // print time stats
    clock_t t1=clock()-t0;
    printf("This took a total time of %f seconds\n",(double)t1/CLOCKS_PER_SEC);
}
