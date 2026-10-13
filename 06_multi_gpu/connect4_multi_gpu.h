#ifndef CONNECT4_MULTI_GPU_H
#define CONNECT4_MULTI_GPU_H

#include <cuda_runtime.h>

constexpr int ROWS = 6;
constexpr int COLUMNS = 7;
constexpr int BOARD_CELLS = ROWS * COLUMNS;
constexpr int EMPTY_CELL = 0;
constexpr int PLAYER_ONE = 1;
constexpr int PLAYER_TWO = 2;
constexpr int THREADS_PER_PLAYER = COLUMNS;

__global__ void playerOneKernel(const int *board, int *selectedColumn);
__global__ void playerTwoKernel(const int *board, int *selectedColumn);

void checkCuda(cudaError_t result, const char *message);
void printBoard(const int *board);
bool applyMove(int *board, int column, int player);
bool hasWinner(const int *board, int player);
bool boardIsFull(const int *board);

#endif
