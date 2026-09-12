#include "resizer_cuda.cuh"
#include <iostream>
#include <cmath>

// Matching Baseline: Standard Bicubic Weight with a = -0.75f
__device__ inline float cubic_weight(float x) {
    x = fabsf(x);
    float a = -0.75f; // Changed from -0.5f to match baseline
    if (x <= 1.0f) {
        return (a + 2.0f) * (x * x * x) - (a + 3.0f) * (x * x) + 1.0f;
    }
    else if (x < 2.0f) {
        return a * (x * x * x) - 5.0f * a * (x * x) + 8.0f * a * x - 4.0f * a;
    }
    return 0.0f;
}

// Matching Baseline: Clamp and round to [0, 255]
__device__ inline uint8_t clamp_pixel(float val) {
    if (val <= 0.0f) return 0;
    if (val >= 255.0f) return 255;
    return static_cast<uint8_t>(llroundf(val));
}

// BICUBIC CUDA KERNEL
__global__ void bicubic_resize_kernel(
    uint8_t* __restrict__ out_img,
    const uint8_t* __restrict__ in_img,
    int old_w, int old_h,
    int new_w, int new_h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= new_w || y >= new_h) return;

    float x_ratio = (float)old_w / new_w;
    float y_ratio = (float)old_h / new_h;

    // Center-aligned inverse mapping
    float src_y = ((float)y + 0.5f) * y_ratio - 0.5f;
    int iy = (int)floorf(src_y);
    float v = src_y - iy;

    float src_x = ((float)x + 0.5f) * x_ratio - 0.5f;
    int ix = (int)floorf(src_x);
    float u = src_x - ix;

    float b_sum = 0.0f;
    float g_sum = 0.0f;
    float r_sum = 0.0f;
    float total_weight = 0.0f;

    // Precalculate the 4 horizontal weights
    float wx[4];

#pragma unroll
    for (int n = -1; n <= 2; ++n) {
        wx[n + 1] = cubic_weight(u - (float)n);
    }

    // Process the 4 rows
#pragma unroll
    for (int m = -1; m <= 2; ++m) {

        // Calculate vertical weight once for this row
        float wy = cubic_weight(v - (float)m);

        int py = iy + m;

        // Boundary clamping
        if (py < 0) py = 0;

        if (py >= old_h) py = old_h - 1;

        // Accumulate horizontal weighted values first
        float b_row = 0.0f;
        float g_row = 0.0f;
        float r_row = 0.0f;
        float row_weight = 0.0f;

        // Process the 4 pixels in this row
#pragma unroll
        for (int n = -1; n <= 2; ++n) {

            int px = ix + n;

            // Boundary clamping
            if (px < 0) px = 0;

            if (px >= old_w) px = old_w - 1;

            int old_offset = (py * old_w + px) * 3;

            float horizontal_weight = wx[n + 1];

            // Apply horizontal weight first
            b_row += in_img[old_offset + 0] * horizontal_weight;
            g_row += in_img[old_offset + 1] * horizontal_weight;
            r_row += in_img[old_offset + 2] * horizontal_weight;

            row_weight += horizontal_weight;
        }

        // Apply vertical weight once to the accumulated row
        b_sum += b_row * wy;
        g_sum += g_row * wy;
        r_sum += r_row * wy;

        total_weight += row_weight * wy;
    }

    // Normalize
    if (total_weight > 0.0f) {
        b_sum /= total_weight;
        g_sum /= total_weight;
        r_sum /= total_weight;
    }

    // Store output pixel
    int new_offset = (y * new_w + x) * 3;

    out_img[new_offset + 0] = clamp_pixel(b_sum);
    out_img[new_offset + 1] = clamp_pixel(g_sum);
    out_img[new_offset + 2] = clamp_pixel(r_sum);
}

const int THREADS_PER_BLOCK = 16;

void resize_image_cuda(
    uint8_t* cpu_out,
    uint8_t* cpu_in,
    int old_w, int old_h,
    int new_w, int new_h)
{
    uint8_t* d_in = nullptr, * d_out = nullptr;
    size_t in_bytes = old_w * old_h * 3 * sizeof(uint8_t);
    size_t out_bytes = new_w * new_h * 3 * sizeof(uint8_t);

    cudaMalloc(&d_in, in_bytes);
    cudaMalloc(&d_out, out_bytes);

    cudaMemcpy(d_in, cpu_in, in_bytes, cudaMemcpyHostToDevice);

    dim3 blockSize(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 gridSize((new_w + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
        (new_h + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    bicubic_resize_kernel << <gridSize, blockSize >> > (d_out, d_in, old_w, old_h, new_w, new_h);

    cudaDeviceSynchronize();

    cudaMemcpy(cpu_out, d_out, out_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}