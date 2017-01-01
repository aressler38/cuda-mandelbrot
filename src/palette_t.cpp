#include "palette_t.h"


const bitmap_t::pixel_t
	black     (  0,  0,  0,255),
	darkred   (139,  0,  0,255),
	red       (255,  0,  0,255),
	green     (  0,255,  0,255),
	blue      (  0,  0,255,255),
	navy      (  0,  0,128,255),
	yellow    (255,255,  0,255),
	gold      (255,215,  0,255),
	orange    (255,165,  0,255),
	orangered (255, 69,  0,255),
	white     (255,255,255,255);

const unsigned palette_t::size = 11;
bitmap_t::pixel_t palette_t::palette[palette_t::size] = {
	black,
	navy,
	blue,
	gold,
	yellow,
	orange,
	orangered,
	orangered,
	orangered,
	orangered,
	darkred,
};

bitmap_t::pixel_t palette_t::grab (float percent) {
	const unsigned pick = unsigned(percent * palette_t::size);

	if (pick < palette_t::size) {
		return palette_t::palette[pick];
	}
	else {
		return palette_t::palette[palette_t::size - 1];
	}
}
