IDIR=./
COMPILER=nvcc
COMPILER_FLAGS=-I$(IDIR) -I/usr/local/cuda/include -lcuda --std c++17
EXE_NAME=connect4_multi_gpu

.PHONY: clean build run all

build: $(EXE_NAME).cu $(EXE_NAME).h
	$(COMPILER) $(COMPILER_FLAGS) $(EXE_NAME).cu -o $(EXE_NAME)
clean:
	rm -f $(EXE_NAME)
run:
	./$(EXE_NAME)
all: clean build run
