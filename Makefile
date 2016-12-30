CXX=g++
CXXFLAGS=-std=c++14
SRC_DIR=src
NVCC=nvcc
BUILD_DIR=build
NVCC_FLAGS=-arch=sm_52 -rdc=true
INCLUDES=-I/usr/local/cuda-8.0/include
LINKS=-L/usr/local/cuda-8.0/lib64 -lcuda -lcudart 
CUDA_LINK_OBJECTS=$(BUILD_DIR)/cuda-wrapper.o $(BUILD_DIR)/mandelbrot.o $(BUILD_DIR)/complex_t.o 
FINAL_OBJECTS=$(BUILD_DIR)/cuda-device-code.o $(CUDA_LINK_OBJECTS)
MAIN_CPP_FILES=$(SRC_DIR)/main.cpp $(SRC_DIR)/bitmap_t.cpp


.PHONY: all
all: clean program

.PHONY: program
program: cuda-device-code.o
	@echo Creating program...
	$(CXX) $(CXXFLAGS) -o $(BUILD_DIR)/program $(FINAL_OBJECTS) $(MAIN_CPP_FILES) $(INCLUDES) $(LINKS)

.PHONY: cuda-device-code.o
cuda-device-code.o: cuda-wrapper.o
	@echo making cuda device code
	$(NVCC) $(NVCC_FLAGS) -dlink -o $(BUILD_DIR)/cuda-device-code.o $(CUDA_LINK_OBJECTS) -lcudadevrt -lcudart

complex_t.o:
	@echo making complex_t
	$(NVCC) $(NVCC_FLAGS) -o $(BUILD_DIR)/complex_t.o -c $(SRC_DIR)/complex_t.cu

mandelbrot.o: complex_t.o
	@echo making mandelbrot.o
	$(NVCC) $(NVCC_FLAGS) -o $(BUILD_DIR)/mandelbrot.o -c $(SRC_DIR)/mandelbrot.cu
	
cuda-wrapper.o: mandelbrot.o
	@echo making wrapper
	$(NVCC) $(NVCC_FLAGS) -o $(BUILD_DIR)/cuda-wrapper.o -c $(SRC_DIR)/cuda-wrapper.cu

.PHONY: clean
clean: 
	rm -rf program $(BUILD_DIR)/*.o
