IDIR=./
COMPILER=nvcc
COMPILER_FLAGS=-I$(IDIR) -I/usr/local/cuda/include -I/usr/local/cuda/lib64 -lcudart -lcuda --std c++17

.PHONY: clean build run

build: *.cu *.h
	$(COMPILER) $(COMPILER_FLAGS) *.cu -o streams

clean:
	rm -f streams

run:
	./streams

all: clean build run
