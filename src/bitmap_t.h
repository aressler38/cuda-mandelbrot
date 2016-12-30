#ifndef __BITMAP_T_HEADER__
#define __BITMAP_T_HEADER__

#include <memory>
#include <cstring>
#include <fstream>
#include <iostream>

class bitmappixel_t;

class bitmap_t {
	public:
	typedef unsigned char byte_t;
	typedef bitmappixel_t pixel_t;
	bitmap_t() = delete;
	bitmap_t(int width, int height);
	void set_image_data(pixel_t*);
	void save(const char* filepath);

	static const int
		channel_count = 4,
		dpi = 2835, //72dpi * 39.3701 in/meter
		header_size = 14,
		info_header_size = 108,
		bits_per_pixel = 32;

	const int image_data_size, width, height;
	std::unique_ptr<byte_t[]> header, info_header, image_data;
};

class bitmappixel_t {
	public:
	bitmappixel_t();	
	bitmappixel_t(unsigned, unsigned, unsigned, unsigned);	
	unsigned red, green, blue, alpha;	
};

#endif
