#include "mandelbrot.h"
__global__ void mandelbrot (
		complex_t *points,
		unsigned *iterations,
		unsigned size,
		unsigned max_iterations)
{
	unsigned index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size) {
		complex_t &c = points[index];
		complex_t z = c;
		unsigned iteration = 0;

		do {
			z = z*z + c;
		} while (z.area() < 4.0 && ++iteration < max_iterations);

		iterations[index] = iteration;
	}
}
