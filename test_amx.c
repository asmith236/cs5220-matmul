#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>
#include <ammintrin.h>  // amx Intrinsics
# include <cpuid.h>

#define BLOCK_SIZE 16

struct tileconfig {
    uint8_t palette_id;
    uint8_t reserved[15];
    uint16_t tile_rows[6];
    uint16_t tile_cols[6];
    uint16_t reserved2[12];
};

int main() {

    unsigned int eax, ebx, ecx, edx;

    // call CPUID with EAX = 0x1D to query tile info
    __cpuid(0x1D, eax, ebx, ecx, edx);

    printf("tile configuration palette ids Supported: %x\n", ebx & 0xFF);

    // init a test matrix (64-byte aligned for proper tile loading)
    double __attribute__((aligned(64))) test_matrix[BLOCK_SIZE][BLOCK_SIZE];

    // fill matrix with some test values
    for (int i = 0; i < BLOCK_SIZE; i++) {
        for (int j = 0; j < BLOCK_SIZE; j++) {
            test_matrix[i][j] = i * BLOCK_SIZE + j + 1;  // Simple test values
        }
    }

    // configure a tile configuration 
    struct tileconfig {
        uint8_t palette_id;
        uint8_t reserved[15];
        uint16_t tile_rows[6];
        uint16_t tile_cols[6];
        uint16_t reserved2[12];
    } cfg;

    // set tile palette ID to 0 
    cfg.palette_id = 0;
    cfg.tile_rows[0] = BLOCK_SIZE;
    cfg.tile_cols[0] = BLOCK_SIZE * sizeof(double);  // num of bytes per row
    _tile_loadconfig(&cfg); 

    printf("loading tile config...\n");

    // perform tile ld from test matrix into tile 0
    printf("attempting to load tile 0 from test matrix...\n");
    _tile_loadd(0, test_matrix, BLOCK_SIZE * sizeof(double));  // ld data into tile 0

    // zero out tile 0 after loading
    _tile_zero(0);

    printf("tile loaded and zeroed out successfully\n");

    // final reset to free tile resources
    _tile_release();

    printf("amx tile load test completed.\n");
    
    return 0;
}