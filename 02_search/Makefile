IDIR=./
COMPILER=nvcc
COMPILER_FLAGS=-I$(IDIR) -I/usr/local/cuda/include -lcuda --std c++17
EXE_NAME=search

.PHONY: clean build run

build: $(EXE_NAME).cu $(EXE_NAME).h
	$(COMPILER) $(COMPILER_FLAGS) $(EXE_NAME).cu -o $(EXE_NAME)
clean:
	rm -f $(EXE_NAME)
	rm -f *.txt
run:
	./$(EXE_NAME) $(ARGS)
all: clean build run
