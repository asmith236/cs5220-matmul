// w optflags:
//  -O3 -Wall -xCORE-AVX512 -mavx512fp16 -mamx-tile -mamx-int8 -mamx-bf16 -fma -qopt-matmul -funroll-loops -qopenmp -qopt-report -fp-model fast=2 -march=native

#include <immintrin.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <ammintrin.h>

const char* dgemm_desc = "My awesome dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 64)
#endif

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define A_BLOCK_SIZE (BLOCK_SIZE * BLOCK_SIZE)
#define B_BLOCK_SIZE (BLOCK_SIZE * BLOCK_SIZE)

struct tileconfig {
    uint8_t palette_id;
    uint8_t reserved[15];
    uint16_t tile_rows[6];  // Number of rows for each tile (6 tiles)
    uint16_t tile_cols[6];  // Number of columns for each tile (6 tiles)
    uint16_t reserved2[12];
};

// Allocate aligned memory for the tile configuration structure
__attribute__((aligned(64))) struct tileconfig tilecfg;

void configure_tile() {
    tilecfg.palette_id = 0;  // Set palette ID to 0 for AMX TILE
    tilecfg.tile_rows[0] = 8;  // Tile 0 (for A matrix) has 8 rows
    tilecfg.tile_cols[0] = 8 * sizeof(double);  // Tile 0 has 8 columns (each double is 8 bytes)
    
    tilecfg.tile_rows[1] = 8;  // Tile 1 (for B matrix) has 8 rows
    tilecfg.tile_cols[1] = 8 * sizeof(double);  // Tile 1 has 8 columns (double precision)
    
    tilecfg.tile_rows[2] = 8;  // Tile 2 (for storing results) has 8 rows
    tilecfg.tile_cols[2] = 8 * sizeof(double);  // Tile 2 has 8 columns
    
    // Tiles 3 to 5 can be configured similarly or left unused if not needed
}

void load_tile_config() {
    _tile_loadconfig(&tilecfg);  // Load the tile configuration into AMX
}

void reset_tile_config() {
    _tile_zero(0);  // Zero out tile 0
    _tile_zero(1);  // Zero out tile 1
    _tile_zero(2);  // Zero out tile 2
}

bool is_aligned(const void* ptr, size_t alignment) {
    return ((uintptr_t)ptr % alignment) == 0;
}

void do_block(const int lda, const double* A, const double* B, double* C,
              const int i, const int j, const int k)
{
    const int M = MIN(BLOCK_SIZE, lda - i);
    const int N = MIN(BLOCK_SIZE, lda - j);
    const int K = MIN(BLOCK_SIZE, lda - k);

    // Temporary buffers for the blocks
    __attribute__((aligned(64))) double A_block[A_BLOCK_SIZE];
    __attribute__((aligned(64))) double B_block[B_BLOCK_SIZE];

    // Copy A block into A_block
    for (int kk = 0; kk < K; ++kk)
        for (int ii = 0; ii < M; ++ii)
            A_block[kk * BLOCK_SIZE + ii] = A[(k + kk) * lda + (i + ii)];

    // Copy B block into B_block
    printf("N=%d\n", N);
    printf("K=%d\n", K);
    int max_B_index = (j + N - 1) * lda + (k + K - 1);
    printf("Max B_index=%d\n", max_B_index);
    for (int jj = 0; jj < N; ++jj){
        for (int kk = 0; kk < K; ++kk){
            printf("Copying B_block: jj=%d, kk=%d, B_index=%d\n", jj, kk, (j + jj) * lda + (k + kk));
            B_block[jj * BLOCK_SIZE + kk] = B[(j + jj) * lda + (k + kk)];
            // printf("Done Copying\n");
        }
        // printf("Done outer loop\n");
    }
    printf("Done copying block B\n");
    // Initialize and configure AMX tiles
    configure_tile();
    load_tile_config();

    for (int jj = 0; jj < N; jj++) {
        int ii = 0;
        printf("Loading tile A: i=%d, k=%d, lda=%d, A_index=%d\n", i, k, lda, i * lda + k);
        _tile_loadd(0, &A[i * lda + k], lda); // Load tile for A
        printf("Loading tile B: k=%d, j=%d, lda=%d, B_index=%d\n", k, j, lda, k * lda + j);
        _tile_loadd(1, &B[k * lda + j], lda); // Load tile for B

        for (int jj_inner = 0; jj_inner < N; jj_inner += 16) { // Rename inner loop variable
            for (int ii_inner = 0; ii_inner < M; ii_inner += 16) { // Rename inner loop variable
                printf("Loading tile C: i=%d, j=%d, lda=%d, C_index=%d\n", i, j, lda, i * lda + j);
                _tile_loadd(2, &C[i * lda + j], lda); // Load tile for C
                _tile_dpbf16ps(2, 0, 1); // Perform multiply-accumulate
                printf("Storing tile C: i=%d, j=%d, lda=%d, C_index=%d\n", i, j, lda, i * lda + j);
                _tile_stored(2, &C[i * lda + j], lda); // Store back the result in C
            }
        }

        // Handle remaining elements for tiles if M is not divisible by 16
        for (; ii < M; ++ii) {
            printf("Handling remaining elements: jj=%d, ii=%d, C_index=%d\n", jj, ii, (j + jj) * lda + (i + ii));
            double cij = C[(j + jj) * lda + (i + ii)];
            for (int kk = 0; kk < K; ++kk) {
                printf("Updating cij: kk=%d, A_block_index=%d, B_block_index=%d\n", kk, kk * BLOCK_SIZE + ii, jj * BLOCK_SIZE + kk);
                cij += A_block[kk * BLOCK_SIZE + ii] * B_block[jj * BLOCK_SIZE + kk];
            }
            C[(j + jj) * lda + (i + ii)] = cij;
        }
    }
    reset_tile_config();
}

void square_dgemm(const int M, double* A, double* B, double* C)
{
    const int n_blocks = M / BLOCK_SIZE + (M % BLOCK_SIZE ? 1 : 0);

    for (int bj = 0; bj < n_blocks; ++bj) { // Outer loop for j
        for (int bi = 0; bi < n_blocks; ++bi) { // Inner loop for i
            const int j = bj * BLOCK_SIZE;
            const int i = bi * BLOCK_SIZE;
            for (int bk = 0; bk < n_blocks; ++bk) { // Middle loop for k
                const int k = bk * BLOCK_SIZE;
                do_block(M, A, B, C, i, j, k);
            }
        }
    }
}