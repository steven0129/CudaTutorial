# IDIR=./
CXX = nvcc

CXXFLAGS += $(shell pkg-config --cflags --libs opencv4)
LDFLAGS += $(shell pkg-config --libs --static opencv)

all: clean build

build: 
	$(CXX) convertRGB2grey.cu --std c++17 `pkg-config opencv --cflags --libs` -o convertRGB2grey -Wno-deprecated-gpu-targets $(CXXFLAGS) -I/usr/local/cuda/include -lcuda

run:
	./convertRGB2grey $(ARGS)

clean:
	rm -f convertRGB2grey
