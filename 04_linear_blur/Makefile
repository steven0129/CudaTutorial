# IDIR=./
CXX = nvcc

CXXFLAGS += $(shell pkg-config --cflags --libs opencv4)
LDFLAGS += $(shell pkg-config --libs --static opencv)

all: clean build

build: 
	$(CXX) simpleLinearBlurFilter.cu --std c++17 `pkg-config opencv4 --cflags --libs` -o simpleLinearBlurFilter -Wno-deprecated-gpu-targets $(CXXFLAGS) -I/usr/local/cuda/include -lcuda

run:
	./simpleLinearBlurFilter $(ARGS)

clean:
	rm -f simpleLinearBlurFilter Lena-blur.png
