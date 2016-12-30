#include "bitmap_t.h"


bitmappixel_t::bitmappixel_t () {}
bitmappixel_t::bitmappixel_t (
	unsigned r,
	unsigned g,
	unsigned b,
	unsigned a) : red(r), green(g), blue(b), alpha(a) {}


bitmap_t::bitmap_t (int width, int height)
	: image_data_size(channel_count*width*height), width(width), height(height) 
{
	const int first_byte_offset = header_size + info_header_size;
	image_data = std::unique_ptr<byte_t[]>(new byte_t[image_data_size] {0});
	header = std::unique_ptr<byte_t[]>(
		new byte_t[header_size] {'B','M', 0,0,0,0, 0,0,0,0, first_byte_offset,0,0,0}
	);
	info_header = std::unique_ptr<byte_t[]>(
		new byte_t[info_header_size] {0}
	);
	int filesize = (info_header_size+header_size) + channel_count*width*height;
	auto info_ptr = info_header.get();
	auto header_ptr = header.get();

	header_ptr[2] = static_cast<byte_t>(filesize);
	header_ptr[3] = static_cast<byte_t>(filesize >> 8);
	header_ptr[4] = static_cast<byte_t>(filesize >> 16);
	header_ptr[5] = static_cast<byte_t>(filesize >> 24);
	info_ptr[0] = static_cast<byte_t>(info_header_size);
	info_ptr[1] = static_cast<byte_t>(info_header_size >> 8);
	info_ptr[2] = static_cast<byte_t>(info_header_size >> 16);
	info_ptr[3] = static_cast<byte_t>(info_header_size >> 24);
	info_ptr[4] = static_cast<byte_t>(width);
	info_ptr[5] = static_cast<byte_t>(width >> 8);
	info_ptr[6] = static_cast<byte_t>(width >> 16);
	info_ptr[7] = static_cast<byte_t>(width >> 24);
	info_ptr[8] = static_cast<byte_t>(height);
	info_ptr[9] = static_cast<byte_t>(height >> 8);
	info_ptr[10] = static_cast<byte_t>(height >> 16);
	info_ptr[11] = static_cast<byte_t>(height >> 24);
	info_ptr[12] = 1; // number of planes must be 1
	info_ptr[14] = static_cast<byte_t>(bits_per_pixel);
	info_ptr[16] = 3; // BI_BITFIELDS
	info_ptr[20] = static_cast<byte_t>(image_data_size);
	info_ptr[21] = static_cast<byte_t>(image_data_size >> 8);
	info_ptr[22] = static_cast<byte_t>(image_data_size >> 16);
	info_ptr[23] = static_cast<byte_t>(image_data_size >> 24);
	info_ptr[24] = static_cast<byte_t>(dpi);
	info_ptr[25] = static_cast<byte_t>(dpi >> 8);
	info_ptr[26] = static_cast<byte_t>(dpi >> 16);
	info_ptr[27] = static_cast<byte_t>(dpi >> 24);
	info_ptr[28] = static_cast<byte_t>(dpi);
	info_ptr[29] = static_cast<byte_t>(dpi >> 8);
	info_ptr[30] = static_cast<byte_t>(dpi >> 16);
	info_ptr[31] = static_cast<byte_t>(dpi >> 24);
	info_ptr[32] = 0;
	info_ptr[36] = 0;
	// red channel bitmask
	info_ptr[40] = static_cast<byte_t>(0x0000ff00);
	info_ptr[41] = static_cast<byte_t>(0x0000ff00 >> 8);
	info_ptr[42] = static_cast<byte_t>(0x0000ff00 >> 16);
	info_ptr[43] = static_cast<byte_t>(0x0000ff00 >> 24);
	// green channel bitmask
	info_ptr[44] = static_cast<byte_t>(0x00ff0000);
	info_ptr[45] = static_cast<byte_t>(0x00ff0000 >> 8);
	info_ptr[46] = static_cast<byte_t>(0x00ff0000 >> 16);
	info_ptr[47] = static_cast<byte_t>(0x00ff0000 >> 24);
	// blue channel bitmask 
	info_ptr[48] = static_cast<byte_t>(0xff000000);
	info_ptr[49] = static_cast<byte_t>(0xff000000 >> 8);
	info_ptr[50] = static_cast<byte_t>(0xff000000 >> 16);
	info_ptr[51] = static_cast<byte_t>(0xff000000 >> 24);
	// alpha channel bitmask
	info_ptr[52] = static_cast<byte_t>(0x000000ff);
	info_ptr[53] = static_cast<byte_t>(0x000000ff >> 8);
	info_ptr[54] = static_cast<byte_t>(0x000000ff >> 16);
	info_ptr[55] = static_cast<byte_t>(0x000000ff >> 24);
}


void bitmap_t::set_image_data (pixel_t *pixels) {
	int x,offset;
	pixel_t *p;

	for (int i=0; i<width; ++i) {
		for (int j=0; j<height; ++j) {
			x = (width-1)-i;
			p = &pixels[x*width + j];

			offset = (i*width+j)*channel_count;
			image_data.get()[offset+3] = p->blue % 256;
			image_data.get()[offset+2] = p->green % 256;
			image_data.get()[offset+1] = p->red % 256;
			image_data.get()[offset+0] = p->alpha % 256;
		}
	}
}


void bitmap_t::save (const char *path) {
	std::ofstream file(path, std::ios::trunc | std::ios::binary);

	if (!file.is_open()) {
		std::cerr << "Unable to write to file " << path << std::endl;
		return;
	}

	file.write(reinterpret_cast<const char*>(header.get()), header_size);
	file.write(reinterpret_cast<const char*>(info_header.get()),
			info_header_size);

	file.write(reinterpret_cast<const char*>(image_data.get()), image_data_size);
	file.close();
}

