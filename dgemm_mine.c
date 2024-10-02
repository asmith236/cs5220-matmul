// FINAL
#include <immintrin.h>  // AVX-512
#include <stdbool.h>     
#include <stdint.h>      
# include <string.h>
# include <stdio.h>
# include <math.h>

const char* dgemm_desc = "My awesome dgemm.";

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define BLOCK_SIZE ((int) 248)
#define A_BLOCK_SIZE (BLOCK_SIZE * BLOCK_SIZE)
#define B_BLOCK_SIZE (BLOCK_SIZE * BLOCK_SIZE)

// check if ptr is aligned to given byte boundary
bool is_aligned(const void* ptr, size_t alignment) {
    return ((uintptr_t)ptr % alignment) == 0;
}

void do_block(const int lda, const double* A, const double* B, double* C,
              const int i, const int j, const int k)
{
    const int M = MIN(BLOCK_SIZE, lda - i);
    const int N = MIN(BLOCK_SIZE, lda - j);
    const int K = MIN(BLOCK_SIZE, lda - k);

    __attribute__((aligned(64))) double A_block[A_BLOCK_SIZE];
    __attribute__((aligned(64))) double B_block[B_BLOCK_SIZE];

    // copy current block A into A_block
    for (int kk = 0; kk < K; ++kk)
        for (int ii = 0; ii < M; ++ii) 
            A_block[kk * BLOCK_SIZE + ii] = A[(k + kk) * lda + (i + ii)];

    // copy current block B into B_block
    for (int jj = 0; jj < BLOCK_SIZE; ++jj) {
        for (int kk = 0; kk < BLOCK_SIZE; ++kk) {
            if ((j + jj) < lda && (k + kk) < lda) {
                B_block[jj * BLOCK_SIZE + kk] = B[(j + jj) * lda + (k + kk)];
            } else {
                B_block[jj * BLOCK_SIZE + kk] = 0.0;  // set out-of-bound elements to 0 (instead of memset)
            }
        }
    }

    // register blocking parameters
    const int R_BLOCK = 8; // operate on 8x8 sub-blocks of c
    const int VECTOR_SIZE = 8; // processes 8 values at once

    // loop over C columns (blocked by R_BLOCK)
    for (int jj = 0; jj < N; jj += R_BLOCK) {
        // loop over the rows of C (blocked by VECTOR_SIZE)
        for (int ii = 0; ii < M; ii += VECTOR_SIZE) {

            // initialize C sub-block in registers
            __m512d c_vec[R_BLOCK];
            for (int r = 0; r < R_BLOCK; ++r) {
                if (is_aligned(&C[(j + jj) * lda + (i + ii)], 64)) {
                    c_vec[r] = _mm512_load_pd(&C[(j + jj + r) * lda + (i + ii)]);
                } else {
                    c_vec[r] = _mm512_loadu_pd(&C[(j + jj + r) * lda + (i + ii)]);
                }
            }

            // perform A_block * B_block for current sub-block
            for (int kk = 0; kk < K; ++kk) {
                __m512d a_vec = _mm512_load_pd(&A_block[kk * BLOCK_SIZE + ii]);

                // broadcase and fma for entire R_BLOCK sub-block
                for (int r = 0; r < R_BLOCK; ++r) {
                    __m512d b_val = _mm512_set1_pd(B_block[(jj + r) * BLOCK_SIZE + kk]);
                    c_vec[r] = _mm512_fmadd_pd(a_vec, b_val, c_vec[r]);
                }
            }

            // store updated C sub-block back into memory
            for (int r = 0; r < R_BLOCK; ++r) {
                if (ii + VECTOR_SIZE <= M) {
                    // if there are at least 8 elements left, store using AVX-512
                    if (is_aligned(&C[(j + jj + r) * lda + (i + ii)], 64)) {
                        _mm512_store_pd(&C[(j + jj + r) * lda + (i + ii)], c_vec[r]);
                    } else {
                        _mm512_storeu_pd(&C[(j + jj + r) * lda + (i + ii)], c_vec[r]);
                    }
                } else {
                    // if fewer than 8 elements are left, handle them scalar-wise
                    for (int rem = 0; rem < M - ii; ++rem) {
                        C[(j + jj + r) * lda + (i + ii + rem)] = ((double*)&c_vec[r])[rem];
                    }
                }
            }
        }
    }
}

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    const int n_blocks = M / BLOCK_SIZE + (M % BLOCK_SIZE ? 1 : 0);
    int bi, bj, bk;
    for (bj = 0; bj < n_blocks; ++bj) { 
        const int j = bj * BLOCK_SIZE;
        for (bk = 0; bk < n_blocks; ++bk) {  
            const int k = bk * BLOCK_SIZE;
            for (bi = 0; bi < n_blocks; ++bi) { 
                const int i = bi * BLOCK_SIZE;
                do_block(M, A, B, C, i, j, k);
            }
        }
    }
}