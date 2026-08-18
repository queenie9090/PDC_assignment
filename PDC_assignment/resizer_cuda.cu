#include "resizer_cuda.cuh"
#include <iostream>
#include <cmath>

// OpenCV's cubic spline kernel uses a = -0.75f
__device__ inline float cubic_weight(float x) {
    x = fabsf(x);
    const float a = -0.75f;
    if (x <= 1.0f) {
        return (a + 2.0f) * x * x * x - (a + 3.0f) * x * x + 1.0f;
    }
    else if (x < 2.0f) {
        return a * x * x * x - 5.0f * a * x * x + 8.0f * a * x - 4.0f * a;
    }
    return 0.0f;
}

// OpenCV border reflection mode: BORDER_REFLECT_101
__device__ inline int reflect_101(int p, int len) {
    if (len <= 1) return 0;
    while (p < 0 || p >= len) {
        if (p < 0) {
            p = -p;
        }
        else if (p >= len) {
            p = 2 * len - 2 - p;
        }
    }
    return p;
}

// Fast CUDA Clamp & Rounding to uint8_t
__device__ inline uint8_t clamp_pixel(float val) {
    // Round to nearest integer and clamp directly to [0.0f, 255.0f]
    float rounded = rintf(val);
    float clamped = fminf(fmaxf(rounded, 0.0f), 255.0f);
    return static_cast<uint8_t>(clamped);
}

// BICUBIC CUDA KERNEL (3-Channel BGR format)
__global__ void bicubic_resize_kernel(
    uint8_t* out_img,
    const uint8_t* __restrict__ in_img,
    int old_w, int old_h,
    int new_w, int new_h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= new_w || y >= new_h) return;

    float x_ratio = (float)old_w / new_w;
    float y_ratio = (float)old_h / new_h;

    float src_y = ((float)y + 0.5f) * y_ratio - 0.5f;
    int iy = (int)floorf(src_y);
    float v = src_y - iy;

    // Precompute row weights (4 weights)
    float wy[4];
    wy[0] = cubic_weight(v + 1.0f);
    wy[1] = cubic_weight(v);
    wy[2] = cubic_weight(1.0f - v);
    wy[3] = cubic_weight(2.0f - v);

    float src_x = ((float)x + 0.5f) * x_ratio - 0.5f;
    int ix = (int)floorf(src_x);
    float u = src_x - ix;

    // Precompute col weights (4 weights)
    float wx[4];
    wx[0] = cubic_weight(u + 1.0f);
    wx[1] = cubic_weight(u);
    wx[2] = cubic_weight(1.0f - u);
    wx[3] = cubic_weight(2.0f - u);

    float b_sum = 0.0f, g_sum = 0.0f, r_sum = 0.0f;

    // 4x4 Grid sampling (16 pixels per thread)
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        int py = reflect_101(iy + m - 1, old_h);
        float weight_y = wy[m];

#pragma unroll
        for (int n = 0; n < 4; ++n) {
            int px = reflect_101(ix + n - 1, old_w);
            float weight = weight_y * wx[n];

            int old_offset = (py * old_w + px) * 3;

            b_sum += in_img[old_offset + 0] * weight;
            g_sum += in_img[old_offset + 1] * weight;
            r_sum += in_img[old_offset + 2] * weight;
        }
    }

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