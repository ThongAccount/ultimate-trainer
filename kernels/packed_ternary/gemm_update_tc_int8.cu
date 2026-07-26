/**
 * gemm_update_tc_int8.cu — TC gradient → int8 counter → bit-flip.
 *
 * Uses int8 counters to halve memory traffic vs int16.
 */
#include <cuda_runtime.h>
#include <cstdint>
#include "packed_ternary.cuh"
// Skeleton for int8 implementation.
// Changes: int16_t* counter -> int8_t* counter. Vectorized load as int32 handles 4 counters.
