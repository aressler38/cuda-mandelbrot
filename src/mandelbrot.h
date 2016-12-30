#ifndef __MANDELBROT_CU_H
#define __MANDELBROT_CU_H

#include "complex_t.h"
__global__ void mandelbrot(
		complex_t*,
		unsigned *iterations,
		unsigned size,
		unsigned max_iterations);

#endif
