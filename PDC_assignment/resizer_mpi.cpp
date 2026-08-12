#include "resizer_mpi.h"
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <mpi.h>

// MSVC / GCC / Clang cross-platform pointer restrict definition
#if defined(_MSC_VER)
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

// Standard integer clamping
static inline int clamp_int(int val, int low, int high)
{
    return (val < low) ? low : ((val > high) ? high : val);
}

// Fast pixel intensity clamping [0, 255]
static inline uint8_t clamp_pixel_fast(float value)
{
    if (value <= 0.0f)   return 0;
    if (value >= 255.0f) return 255;
    return static_cast<uint8_t>(value);
}

// Bicubic Kernel Weight Function (Catmull-Rom)
static inline float cubic_weight_mpi(float x)
{
    x = std::fabs(x);
    const float a = -0.5f;

    if (x <= 1.0f)
    {
        return (a + 2.0f) * (x * x * x) - (a + 3.0f) * (x * x) + 1.0f;
    }

    if (x < 2.0f)
    {
        return a * (x * x * x) - 5.0f * a * (x * x) + 8.0f * a * x - 4.0f * a;
    }

    return 0.0f;
}

// PRECOMPUTED HORIZONTAL LOOKUP STRUCT
struct XWeights {
    int px_offset[4]; // Precalculated byte offsets for RGB channels
    float wx[4];      // Precalculated horizontal weights
};

// MAIN PURE MPI BICUBIC RESIZER
void resize_image_mpi(
    uint8_t* RESTRICT cpu_out,
    uint8_t* RESTRICT cpu_in,
    int old_w,
    int old_h,
    int new_w,
    int new_h
)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t input_bytes = static_cast<size_t>(old_w) * old_h * 3;
    size_t output_bytes = static_cast<size_t>(new_w) * new_h * 3;

    // 1. Broadcast input image from Rank 0 to all processes
    std::vector<uint8_t> input_image(input_bytes);

    if (rank == 0 && cpu_in != nullptr)
    {
        std::copy(cpu_in, cpu_in + input_bytes, input_image.begin());
    }

    MPI_Bcast(
        input_image.data(),
        static_cast<int>(input_bytes),
        MPI_UNSIGNED_CHAR,
        0,
        MPI_COMM_WORLD
    );

    // Ensure worker ranks have a valid output memory target
    std::vector<uint8_t> fallback_out;
    uint8_t* RESTRICT effective_out = cpu_out;

    if (effective_out == nullptr)
    {
        fallback_out.resize(output_bytes, 0);
        effective_out = fallback_out.data();
    }

    // 2. Divide output rows evenly across MPI ranks
    int base_rows = new_h / size;
    int extra_rows = new_h % size;
    int output_rows = base_rows + (rank < extra_rows ? 1 : 0);
    int output_start = rank * base_rows + std::min(rank, extra_rows);

    size_t local_bytes = static_cast<size_t>(output_rows) * new_w * 3;
    std::vector<uint8_t> local_output(local_bytes);

    // 3. Precompute Horizontal (X) Lookup Table (LUT)
    const float x_ratio = static_cast<float>(old_w) / static_cast<float>(new_w);
    const float y_ratio = static_cast<float>(old_h) / static_cast<float>(new_h);

    std::vector<XWeights> x_lut(new_w);

    for (int x = 0; x < new_w; ++x)
    {
        float src_x = (static_cast<float>(x) + 0.5f) * x_ratio - 0.5f;
        int ix = static_cast<int>(std::floor(src_x));
        float u = src_x - static_cast<float>(ix);

        for (int n = -1; n <= 2; ++n)
        {
            int px = clamp_int(ix + n, 0, old_w - 1);
            int idx = n + 1;

            x_lut[x].px_offset[idx] = px * 3;
            x_lut[x].wx[idx] = cubic_weight_mpi(u - static_cast<float>(n));
        }
    }

    // 4. Bicubic Interpolation Computation (MPI Row Domain)
    const uint8_t* RESTRICT src_base = input_image.data();
    uint8_t* RESTRICT dst_base = local_output.data();

    for (int local_y = 0; local_y < output_rows; ++local_y)
    {
        int y = output_start + local_y;

        float src_y = (static_cast<float>(y) + 0.5f) * y_ratio - 0.5f;
        int iy = static_cast<int>(std::floor(src_y));
        float v = src_y - static_cast<float>(iy);

        // Precompute row pointers and vertical weights for this row
        float wy[4];
        const uint8_t* RESTRICT row_ptrs[4];

        for (int m = -1; m <= 2; ++m)
        {
            int py = clamp_int(iy + m, 0, old_h - 1);
            int idx = m + 1;

            wy[idx] = cubic_weight_mpi(v - static_cast<float>(m));
            row_ptrs[idx] = src_base + static_cast<size_t>(py) * old_w * 3;
        }

        uint8_t* RESTRICT out_ptr = dst_base + static_cast<size_t>(local_y) * new_w * 3;

        for (int x = 0; x < new_w; ++x)
        {
            const XWeights& x_info = x_lut[x];

            float b_sum = 0.0f;
            float g_sum = 0.0f;
            float r_sum = 0.0f;
            float total_weight = 0.0f;

            // 4x4 matrix accumulation
            for (int m = 0; m < 4; ++m)
            {
                float wy_m = wy[m];
                const uint8_t* RESTRICT src_row = row_ptrs[m];

                for (int n = 0; n < 4; ++n)
                {
                    float weight = x_info.wx[n] * wy_m;
                    int offset = x_info.px_offset[n];

                    b_sum += src_row[offset + 0] * weight;
                    g_sum += src_row[offset + 1] * weight;
                    r_sum += src_row[offset + 2] * weight;

                    total_weight += weight;
                }
            }

            if (total_weight > 0.0f)
            {
                float inv_w = 1.0f / total_weight;
                b_sum *= inv_w;
                g_sum *= inv_w;
                r_sum *= inv_w;
            }

            int out_idx = x * 3;
            out_ptr[out_idx + 0] = clamp_pixel_fast(b_sum);
            out_ptr[out_idx + 1] = clamp_pixel_fast(g_sum);
            out_ptr[out_idx + 2] = clamp_pixel_fast(r_sum);
        }
    }

    // 5. Gather all local outputs into the global output buffer
    std::vector<int> recv_counts(size);
    std::vector<int> displacements(size);

    for (int r = 0; r < size; ++r)
    {
        int r_rows = base_rows + (r < extra_rows ? 1 : 0);
        int r_start = r * base_rows + std::min(r, extra_rows);

        recv_counts[r] = r_rows * new_w * 3;
        displacements[r] = r_start * new_w * 3;
    }

    MPI_Allgatherv(
        local_output.data(),
        static_cast<int>(local_bytes),
        MPI_UNSIGNED_CHAR,

        effective_out,
        recv_counts.data(),
        displacements.data(),
        MPI_UNSIGNED_CHAR,

        MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);
}