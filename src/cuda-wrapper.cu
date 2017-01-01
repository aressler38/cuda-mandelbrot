#include "cuda-wrapper.h"

/**
 * Host function that copies the data and launches the work on GPU
 */
unsigned *computeMandelbrot(
	complex_t *points,
	unsigned size,
	unsigned max_iterations=1024)
{
	static const int BLOCK_SIZE = 1024;
	const int blockCount = (size+BLOCK_SIZE-1)/BLOCK_SIZE;
	unsigned *iterations = new unsigned[size];
	unsigned *device_iterations;
	complex_t *device_points;

	// Allocate memory on device
	cudaMalloc((void **)&device_points, sizeof(complex_t)*size);
	cudaMalloc((void **)&device_iterations, sizeof(unsigned)*size);
	// Copy the points to device
	cudaMemcpy(device_points, points, sizeof(complex_t)*size, cudaMemcpyHostToDevice);
	// Run kernel
	mandelbrot<<<blockCount, BLOCK_SIZE>>>(device_points, device_iterations, size, max_iterations);
	// Copy the iterations to host
	cudaMemcpy(iterations, device_iterations, sizeof(unsigned)*size, cudaMemcpyDeviceToHost);
	// Clean up 
	cudaFree(device_iterations);
	cudaFree(device_points);
	return iterations;
}
