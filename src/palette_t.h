#ifndef __PALETTE__HEADER__
#define __PALETTE__HEADER__


#include "bitmap_t.h"

class palette_t {
	public:
	static const unsigned size;
	static bitmap_t::pixel_t palette[];
	static bitmap_t::pixel_t grab(float percent);
};

#endif
