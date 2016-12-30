#include <iostream>
#include <memory>
#include "cuda-wrapper.h"
#include "bitmap_t.h"


int main (void) {
	bitmap_t::pixel_t black(0,0,0,255);
	bitmap_t::pixel_t red(255,0,0,255);
	bitmap_t::pixel_t green(0,255,0,255);
	bitmap_t::pixel_t blue(0,0,255,255);
	unsigned 
		width = 1024,
		height = 1024,
		size = width*height,
		max_iterations = 512,
		i,j,
		index, iteration;
	double 
		x_min = -3.0,
		x_max = 1.5,
		y_min = -2.25,
		y_max = 2.25,
		dx = (x_max - x_min) / double(width),
		dy = (y_max - y_min) / double(height);
	complex_t *points = new complex_t[size];
	bitmap_t img(width, height);
	complex_t *point;
	bitmap_t::pixel_t *pixel;

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

	std::unique_ptr<bitmap_t::pixel_t[]>
		pixels(new bitmap_t::pixel_t[width*height]);

	for (int i=0; i<height; ++i) {
		for (int j=0; j<width; ++j) {
			index = i*width + j;
			pixel = &pixels.get()[index];
			iteration = iterations[index];

			//std::cerr << "iteration " << index << "=" << iteration << "\n";
			if (iteration >= max_iterations) {
				pixel->alpha = 255;
				pixel->blue = i > 255 ? 255 : 0;
				pixel->green = j;
				pixel->red = i>255 ? j>255 ? 255 : 0 : 0;
			}
			else {
				*pixel = black;
			}
		}
	}

	img.set_image_data(pixels.get());
	img.save("/tmp/test.bmp");

	return 0;
}
