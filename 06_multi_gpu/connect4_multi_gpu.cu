#include "connect4_multi_gpu.h"

#include <cstdio>
#include <cstdlib>
#include <thread>
#include <chrono>

__device__ int findLandingRow(const int *board, int column)
{
    for (int row = ROWS - 1; row >= 0; --row)
    {
        if (board[row * COLUMNS + column] == EMPTY_CELL)
        {
            return row;
        }
    }

    return -1;
}

__device__ int countDirection(const int *board, int row, int column, int rowStep, int columnStep, int player)
{
    int count = 0;
    row += rowStep;
    column += columnStep;

    while (row >= 0 && row < ROWS && column >= 0 && column < COLUMNS && board[row * COLUMNS + column] == player)
    {
        ++count;
        row += rowStep;
        column += columnStep;
    }

    return count;
}

__device__ int maxInt(int left, int right)
{
    return left > right ? left : right;
}

__device__ int absInt(int value)
{
    return value < 0 ? -value : value;
}

__device__ int scoreMove(const int *board, int column, int player, int opponent)
{
    int row = findLandingRow(board, column);
    if (row < 0)
    {
        return -1;
    }

    int bestLine = 1;
    const int rowSteps[4] = {0, 1, 1, 1};
    const int columnSteps[4] = {1, 0, 1, -1};

    for (int direction = 0; direction < 4; ++direction)
    {
        int playerLine = 1;
        playerLine += countDirection(board, row, column, rowSteps[direction], columnSteps[direction], player);
        playerLine += countDirection(board, row, column, -rowSteps[direction], -columnSteps[direction], player);
        bestLine = maxInt(bestLine, playerLine);

        int opponentLine = 1;
        opponentLine += countDirection(board, row, column, rowSteps[direction], columnSteps[direction], opponent);
        opponentLine += countDirection(board, row, column, -rowSteps[direction], -columnSteps[direction], opponent);

        if (playerLine >= 4)
        {
            return 1000 + column;
        }
        if (opponentLine >= 4)
        {
            bestLine = maxInt(bestLine, 100);
        }
    }

    int centerBonus = COLUMNS - absInt(column - (COLUMNS / 2));
    return bestLine * 10 + centerBonus;
}

__device__ void chooseMove(const int *board, int *selectedColumn, int player, int opponent)
{
    int column = threadIdx.x;
    if (column >= COLUMNS)
    {
        return;
    }

    int score = scoreMove(board, column, player, opponent);
    int packedScore = score * COLUMNS + (COLUMNS - 1 - column);
    atomicMax(selectedColumn, packedScore);
}

__global__ void playerOneKernel(const int *board, int *selectedColumn)
{
    chooseMove(board, selectedColumn, PLAYER_ONE, PLAYER_TWO);
}

__global__ void playerTwoKernel(const int *board, int *selectedColumn)
{
    chooseMove(board, selectedColumn, PLAYER_TWO, PLAYER_ONE);
}

void checkCuda(cudaError_t result, const char *message)
{
    if (result != cudaSuccess)
    {
        std::fprintf(stderr, "%s: %s\n", message, cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

void printBoard(const int *board)
{
    std::printf("\n");
    for (int row = 0; row < ROWS; ++row)
    {
        for (int column = 0; column < COLUMNS; ++column)
        {
            int cell = board[row * COLUMNS + column];
            char marker = cell == PLAYER_ONE ? 'X' : (cell == PLAYER_TWO ? 'O' : '.');
            std::printf("%c ", marker);
        }
        std::printf("\n");
    }
    std::printf("0 1 2 3 4 5 6\n");
}

bool applyMove(int *board, int column, int player)
{
    if (column < 0 || column >= COLUMNS)
    {
        return false;
    }

    for (int row = ROWS - 1; row >= 0; --row)
    {
        int index = row * COLUMNS + column;
        if (board[index] == EMPTY_CELL)
        {
            board[index] = player;
            return true;
        }
    }

    return false;
}

bool hasWinner(const int *board, int player)
{
    const int rowSteps[4] = {0, 1, 1, 1};
    const int columnSteps[4] = {1, 0, 1, -1};

    for (int row = 0; row < ROWS; ++row)
    {
        for (int column = 0; column < COLUMNS; ++column)
        {
            if (board[row * COLUMNS + column] != player)
            {
                continue;
            }

            for (int direction = 0; direction < 4; ++direction)
            {
                int count = 1;
                int nextRow = row + rowSteps[direction];
                int nextColumn = column + columnSteps[direction];

                while (nextRow >= 0 && nextRow < ROWS && nextColumn >= 0 && nextColumn < COLUMNS && board[nextRow * COLUMNS + nextColumn] == player)
                {
                    ++count;
                    if (count == 4)
                    {
                        return true;
                    }
                    nextRow += rowSteps[direction];
                    nextColumn += columnSteps[direction];
                }
            }
        }
    }

    return false;
}

bool boardIsFull(const int *board)
{
    for (int column = 0; column < COLUMNS; ++column)
    {
        if (board[column] == EMPTY_CELL)
        {
            return false;
        }
    }

    return true;
}

int runTurn(int *hostBoard, int currentPlayer, int deviceId)
{
    int *deviceBoard = nullptr;
    int *devicePackedMove = nullptr;
    int packedMove = -1;

    checkCuda(cudaSetDevice(deviceId), "Failed to select CUDA device");
    checkCuda(cudaMalloc((void **)&deviceBoard, BOARD_CELLS * sizeof(int)), "Failed to allocate board on the device");
    checkCuda(cudaMalloc((void **)&devicePackedMove, sizeof(int)), "Failed to allocate move on the device");
    checkCuda(cudaMemcpy(deviceBoard, hostBoard, BOARD_CELLS * sizeof(int), cudaMemcpyHostToDevice), "Failed to copy board to the device");
    checkCuda(cudaMemcpy(devicePackedMove, &packedMove, sizeof(int), cudaMemcpyHostToDevice), "Failed to initialize move on the device");

    if (currentPlayer == PLAYER_ONE)
    {
        playerOneKernel<<<1, THREADS_PER_PLAYER>>>(deviceBoard, devicePackedMove);
    }
    else
    {
        playerTwoKernel<<<1, THREADS_PER_PLAYER>>>(deviceBoard, devicePackedMove);
    }

    checkCuda(cudaGetLastError(), "Failed to launch player kernel");
    checkCuda(cudaDeviceSynchronize(), "Failed to synchronize player kernel");
    checkCuda(cudaMemcpy(&packedMove, devicePackedMove, sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy move to the host");
    checkCuda(cudaFree(devicePackedMove), "Failed to free device move");
    checkCuda(cudaFree(deviceBoard), "Failed to free device board");

    return COLUMNS - 1 - (packedMove % COLUMNS);
}

int main()
{
    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "Failed to count CUDA devices");

    if (deviceCount < 2)
    {
        std::fprintf(stderr, "This demo needs at least two CUDA devices, but only %d device(s) were found.\n", deviceCount);
        return EXIT_FAILURE;
    }

    int board[BOARD_CELLS] = {0};
    int playerDevices[3] = {-1, 0, 1};
    int currentPlayer = PLAYER_ONE;

    std::printf("Player 1 uses GPU %d and Player 2 uses GPU %d.\n", playerDevices[PLAYER_ONE], playerDevices[PLAYER_TWO]);

    for (int turn = 1; turn <= BOARD_CELLS; ++turn)
    {
        int deviceId = playerDevices[currentPlayer];
        int column = runTurn(board, currentPlayer, deviceId);

        std::printf("Turn %d: Player %d on GPU %d chooses column %d.\n", turn, currentPlayer, deviceId, column);

        if (!applyMove(board, column, currentPlayer))
        {
            std::fprintf(stderr, "Player %d selected invalid column %d.\n", currentPlayer, column);
            return EXIT_FAILURE;
        }

        printBoard(board);

        if (hasWinner(board, currentPlayer))
        {
            std::printf("Player %d wins.\n", currentPlayer);
            break;
        }

        if (boardIsFull(board))
        {
            std::printf("The game is a draw.\n");
            break;
        }

        currentPlayer = currentPlayer == PLAYER_ONE ? PLAYER_TWO : PLAYER_ONE;
	std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    for (int player = PLAYER_ONE; player <= PLAYER_TWO; ++player)
    {
        checkCuda(cudaSetDevice(playerDevices[player]), "Failed to select CUDA device for reset");
        checkCuda(cudaDeviceReset(), "Failed to reset CUDA device");
    }

    return EXIT_SUCCESS;
}
