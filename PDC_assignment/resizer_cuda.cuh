#pragma once
#include <cuda_runtime.h>
#include <cstdint>

void resize_image_cuda(
    uint8_t* cpu_out,
    uint8_t* cpu_in,
    int old_w, int old_h,
    int new_w, int new_h);