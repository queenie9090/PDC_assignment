#include "resizer_mpi.h"

#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <mpi.h>

#if defined(_MSC_VER)
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

static inline int clamp_int(int v, int lo, int hi)
{
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

static inline uint8_t clamp_pixel(float val)
{
    if (val <= 0.0f) return 0;
    if (val >= 255.0f) return 255;
    return static_cast<uint8_t>(val + 0.5f);
}

// Catmull-Rom style bicubic spline kernel
static inline float cubic_weight(float x)
{
    x = std::fabs(x);
    constexpr float a = -0.75f;

    if (x <= 1.0f)
        return (a + 2.0f) * x * x * x - (a + 3.0f) * x * x + 1.0f;

    if (x < 2.0f)
        return a * x * x * x - 5.0f * a * x * x + 8.0f * a * x - 4.0f * a;

    return 0.0f;
}

struct XWeights
{
    int offset[4];
    float weight[4];
};

struct YWeights
{
    int row[4];
    float weight[4];
};

void resize_image_mpi(
    uint8_t* RESTRICT cpu_out,
    uint8_t* RESTRICT cpu_in,
    int old_w,
    int old_h,
    int new_w,
    int new_h)
{
    int rank = 0;
    int size = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const float x_ratio = static_cast<float>(old_w) / static_cast<float>(new_w);
    const float y_ratio = static_cast<float>(old_h) / static_cast<float>(new_h);
    const int input_row_bytes = old_w * 3;
    const int output_row_bytes = new_w * 3;

    // Divide output rows among MPI processes
    const int base_rows = new_h / size;
    const int extra_rows = new_h % size;
    const int output_rows = base_rows + (rank < extra_rows ? 1 : 0);
    const int output_start = rank * base_rows + std::min(rank, extra_rows);

    // Calculate source rows required by each process
    std::vector<int> src_min(size);
    std::vector<int> src_max(size);
    std::vector<int> src_rows(size);
    std::vector<int> scatter_counts(size);
    std::vector<int> scatter_displacements(size);

    for (int r = 0; r < size; ++r)
    {
        const int r_output_rows = base_rows + (r < extra_rows ? 1 : 0);
        const int r_output_start = r * base_rows + std::min(r, extra_rows);

        if (r_output_rows == 0)
        {
            src_min[r] = 0;
            src_max[r] = -1;
            src_rows[r] = 0;
            scatter_counts[r] = 0;
            scatter_displacements[r] = 0;
            continue;
        }

        const float y_first = (static_cast<float>(r_output_start) + 0.5f) * y_ratio - 0.5f;
        const int iy_first = static_cast<int>(std::floor(y_first));

        const int r_output_end = r_output_start + r_output_rows - 1;
        const float y_last = (static_cast<float>(r_output_end) + 0.5f) * y_ratio - 0.5f;
        const int iy_last = static_cast<int>(std::floor(y_last));

        src_min[r] = clamp_int(iy_first - 1, 0, old_h - 1);
        src_max[r] = clamp_int(iy_last + 2, 0, old_h - 1);
        src_rows[r] = src_max[r] - src_min[r] + 1;
        scatter_counts[r] = src_rows[r] * input_row_bytes;
        scatter_displacements[r] = src_min[r] * input_row_bytes;
    }

    const int my_src_start = src_min[rank];
    const int my_src_rows = src_rows[rank];

    const size_t local_input_bytes = static_cast<size_t>(my_src_rows) * input_row_bytes;

    std::vector<uint8_t> local_input(local_input_bytes);

    // Distribute only the input rows needed by each process
    MPI_Scatterv(
        cpu_in,
        scatter_counts.data(),
        scatter_displacements.data(),
        MPI_UNSIGNED_CHAR,
        local_input.empty() ? nullptr : local_input.data(),
        static_cast<int>(local_input_bytes),
        MPI_UNSIGNED_CHAR,
        0,
        MPI_COMM_WORLD);

    // Precompute horizontal bicubic LUT only once on rank 0
    std::vector<XWeights> x_lut(new_w);

    if (rank == 0)
    {
        for (int x = 0; x < new_w; ++x)
        {
            const float src_x = (static_cast<float>(x) + 0.5f) * x_ratio - 0.5f;
            const int ix = static_cast<int>(std::floor(src_x));
            const float u = src_x - static_cast<float>(ix);

            float sum = 0.0f;

            for (int n = -1; n <= 2; ++n)
            {
                const int index = n + 1;
                const int px = clamp_int(ix + n, 0, old_w - 1);
                const float w = cubic_weight(u - static_cast<float>(n));

                x_lut[x].offset[index] = px * 3;
                x_lut[x].weight[index] = w;

                sum += w;
            }

            if (sum != 0.0f)
            {
                const float inv = 1.0f / sum;
                x_lut[x].weight[0] *= inv;
                x_lut[x].weight[1] *= inv;
                x_lut[x].weight[2] *= inv;
                x_lut[x].weight[3] *= inv;
            }
        }
    }

    // Broadcast horizontal LUT to all processes
    MPI_Bcast(
        x_lut.data(),
        static_cast<int>(new_w * sizeof(XWeights)),
        MPI_BYTE,
        0,
        MPI_COMM_WORLD);

    // Precompute vertical bicubic LUT
    std::vector<YWeights> y_lut(output_rows);

    for (int local_y = 0; local_y < output_rows; ++local_y)
    {
        const int y = output_start + local_y;
        const float src_y = (static_cast<float>(y) + 0.5f) * y_ratio - 0.5f;
        const int iy = static_cast<int>(std::floor(src_y));
        const float v = src_y - static_cast<float>(iy);

        float sum = 0.0f;

        for (int m = -1; m <= 2; ++m)
        {
            const int index = m + 1;
            const int py = clamp_int(iy + m, 0, old_h - 1);
            const float w = cubic_weight(v - static_cast<float>(m));

            y_lut[local_y].row[index] = py - my_src_start;
            y_lut[local_y].weight[index] = w;

            sum += w;
        }

        if (sum != 0.0f)
        {
            const float inv = 1.0f / sum;
            y_lut[local_y].weight[0] *= inv;
            y_lut[local_y].weight[1] *= inv;
            y_lut[local_y].weight[2] *= inv;
            y_lut[local_y].weight[3] *= inv;
        }
    }

    const size_t local_output_bytes = static_cast<size_t>(output_rows) * output_row_bytes;

    std::vector<uint8_t> local_output(local_output_bytes);

    const uint8_t* RESTRICT src = local_input.empty() ? nullptr : local_input.data();
    uint8_t* RESTRICT dst = local_output.empty() ? nullptr : local_output.data();

    // Bicubic interpolation
    for (int local_y = 0; local_y < output_rows; ++local_y)
    {
        const YWeights& yl = y_lut[local_y];

        const uint8_t* RESTRICT r0 = src + static_cast<size_t>(yl.row[0]) * input_row_bytes;
        const uint8_t* RESTRICT r1 = src + static_cast<size_t>(yl.row[1]) * input_row_bytes;
        const uint8_t* RESTRICT r2 = src + static_cast<size_t>(yl.row[2]) * input_row_bytes;
        const uint8_t* RESTRICT r3 = src + static_cast<size_t>(yl.row[3]) * input_row_bytes;

        const float wy0 = yl.weight[0];
        const float wy1 = yl.weight[1];
        const float wy2 = yl.weight[2];
        const float wy3 = yl.weight[3];

        uint8_t* RESTRICT out = dst + static_cast<size_t>(local_y) * output_row_bytes;

#if defined(_MSC_VER)
#pragma loop(ivdep)
#endif
        for (int x = 0; x < new_w; ++x)
        {
            const XWeights& xl = x_lut[x];

            const int o0 = xl.offset[0];
            const int o1 = xl.offset[1];
            const int o2 = xl.offset[2];
            const int o3 = xl.offset[3];

            const float wx0 = xl.weight[0];
            const float wx1 = xl.weight[1];
            const float wx2 = xl.weight[2];
            const float wx3 = xl.weight[3];

            const float h0_b = r0[o0] * wx0 + r0[o1] * wx1 + r0[o2] * wx2 + r0[o3] * wx3;
            const float h0_g = r0[o0 + 1] * wx0 + r0[o1 + 1] * wx1 + r0[o2 + 1] * wx2 + r0[o3 + 1] * wx3;
            const float h0_r = r0[o0 + 2] * wx0 + r0[o1 + 2] * wx1 + r0[o2 + 2] * wx2 + r0[o3 + 2] * wx3;

            const float h1_b = r1[o0] * wx0 + r1[o1] * wx1 + r1[o2] * wx2 + r1[o3] * wx3;
            const float h1_g = r1[o0 + 1] * wx0 + r1[o1 + 1] * wx1 + r1[o2 + 1] * wx2 + r1[o3 + 1] * wx3;
            const float h1_r = r1[o0 + 2] * wx0 + r1[o1 + 2] * wx1 + r1[o2 + 2] * wx2 + r1[o3 + 2] * wx3;

            const float h2_b = r2[o0] * wx0 + r2[o1] * wx1 + r2[o2] * wx2 + r2[o3] * wx3;
            const float h2_g = r2[o0 + 1] * wx0 + r2[o1 + 1] * wx1 + r2[o2 + 1] * wx2 + r2[o3 + 1] * wx3;
            const float h2_r = r2[o0 + 2] * wx0 + r2[o1 + 2] * wx1 + r2[o2 + 2] * wx2 + r2[o3 + 2] * wx3;

            const float h3_b = r3[o0] * wx0 + r3[o1] * wx1 + r3[o2] * wx2 + r3[o3] * wx3;
            const float h3_g = r3[o0 + 1] * wx0 + r3[o1 + 1] * wx1 + r3[o2 + 1] * wx2 + r3[o3 + 1] * wx3;
            const float h3_r = r3[o0 + 2] * wx0 + r3[o1 + 2] * wx1 + r3[o2 + 2] * wx2 + r3[o3 + 2] * wx3;

            const float b = h0_b * wy0 + h1_b * wy1 + h2_b * wy2 + h3_b * wy3;
            const float g = h0_g * wy0 + h1_g * wy1 + h2_g * wy2 + h3_g * wy3;
            const float r = h0_r * wy0 + h1_r * wy1 + h2_r * wy2 + h3_r * wy3;

            const int out_idx = x * 3;

            out[out_idx] = clamp_pixel(b);
            out[out_idx + 1] = clamp_pixel(g);
            out[out_idx + 2] = clamp_pixel(r);
        }
    }

    // Gather output from all processes
    std::vector<int> recv_counts(size);
    std::vector<int> recv_displacements(size);

    for (int r = 0; r < size; ++r)
    {
        const int r_rows = base_rows + (r < extra_rows ? 1 : 0);
        const int r_start = r * base_rows + std::min(r, extra_rows);

        recv_counts[r] = r_rows * output_row_bytes;
        recv_displacements[r] = r_start * output_row_bytes;
    }

    MPI_Gatherv(
        local_output.empty() ? nullptr : local_output.data(),
        static_cast<int>(local_output_bytes),
        MPI_UNSIGNED_CHAR,
        cpu_out,
        recv_counts.data(),
        recv_displacements.data(),
        MPI_UNSIGNED_CHAR,
        0,
        MPI_COMM_WORLD);
}