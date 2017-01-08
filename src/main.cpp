#include <iostream>
#include <memory>
#include "cuda-wrapper.h"
#include "bitmap_t.h"
#include "palette_t.h"
#include <sstream>
#include <numeric>


/**
 * Invoke with cmd line args:
 * x_min x_max y_min y_max
 */

int main (int argc, char *argv[]) {
	const unsigned
		//width = 4096, height = 4096,
		width = 8192, height = 8192,
		size = width*height,
		max_iterations = 2048;
	unsigned i,j, index, iteration;
	double
		x_min = -3.0,
		x_max = 1.5,
		y_min = -2.25,
		y_max = 2.25,
		dx,dy;
	complex_t *points = new complex_t[size];
	bitmap_t img(width, height);
	complex_t *point;

	// Parse args
	if (argc == 5) {
		std::stringstream ss;
		ss << argv[1];
		ss >> x_min;
		ss.clear();
		ss << argv[2];
		ss >> x_max;
		ss.clear();
		ss << argv[3];
		ss >> y_min;
		ss.clear();
		ss << argv[4];
		ss >> y_max;
	}

	dx = (x_max - x_min) / double(width),
	dy = (y_max - y_min) / double(height);

	// Go left to right, top to bottom
	for (i=0; i<height; ++i) {
		for (j=0; j<width; ++j) {
			point = &points[i*width + j];

			point->a = x_min + j*dx;
			point->b = y_max - i*dy;
		}
	}

	std::unique_ptr<unsigned[]> iterations
		(computeMandelbrot(points, size, max_iterations));

	// Make some color pixels
	std::unique_ptr<bitmap_t::pixel_t[]>
		pixels(new bitmap_t::pixel_t[width*height]);

	for (int i=0; i<height; ++i) {
		for (int j=0; j<width; ++j) {
			index = i*width + j;
			iteration = iterations[index];
			pixels.get()[index] = palette_t::grab(float(iteration) / max_iterations);
		}
	}

	img.set_image_data(pixels.get());
	img.save("/tmp/test.bmp");

	long long counter = 0;
	for (i=0;i<size;++i) {
		counter += static_cast<long long>(iterations[i]);
	}

	std::cout << "Total iterations: " << counter << "\n";

	return 0;
}
