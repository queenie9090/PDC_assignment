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

static inline uint8_t clamp_pixel_fast(float v)
{
    if (v <= 0.0f) return 0;
    if (v >= 255.0f) return 255;
    return static_cast<uint8_t>(v + 0.5f);
}

// Catmull-Rom bicubic kernel
static inline float cubic_weight_mpi(float x)
{
    x = std::fabs(x);
    constexpr float a = -0.75f;

    if (x <= 1.0f)
    {
        return (a + 2.0f) * x * x * x - (a + 3.0f) * x * x + 1.0f;
    }

    if (x < 2.0f)
    {
        return a * x * x * x - 5.0f * a * x * x + 8.0f * a * x - 4.0f * a;
    }

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

void resize_image_mpi_io(
    const char* input_filename,
    const char* output_filename,
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

    // Divide output rows among MPI processes
    const int base_rows = new_h / size;
    const int extra_rows = new_h % size;
    const int output_rows = base_rows + (rank < extra_rows ? 1 : 0);
    const int output_start = rank * base_rows + std::min(rank, extra_rows);

    // Calculate source rows required by THIS process
    int my_src_min = 0;
    int my_src_max = -1;

    if (output_rows > 0)
    {
        const float y_first = (static_cast<float>(output_start) + 0.5f) * y_ratio - 0.5f;
        const int iy_first = static_cast<int>(std::floor(y_first));

        const int output_end = output_start + output_rows - 1;
        const float y_last = (static_cast<float>(output_end) + 0.5f) * y_ratio - 0.5f;
        const int iy_last = static_cast<int>(std::floor(y_last));

        my_src_min = clamp_int(iy_first - 1, 0, old_h - 1);
        my_src_max = clamp_int(iy_last + 2, 0, old_h - 1);
    }

    const int my_src_rows = my_src_max - my_src_min + 1;
    const int input_row_bytes = old_w * 3;
    const size_t local_input_bytes = static_cast<size_t>(my_src_rows) * static_cast<size_t>(input_row_bytes);

    std::vector<uint8_t> local_input(local_input_bytes);

    // --- 1. MPI-IO READ ---
    MPI_File fh_in;
    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_in);

    if (output_rows > 0)
    {
        MPI_Offset read_offset = static_cast<MPI_Offset>(my_src_min) * input_row_bytes;
        MPI_File_read_at_all(fh_in, read_offset, local_input.data(),
            static_cast<int>(local_input_bytes), MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&fh_in);

    // Precompute horizontal bicubic weights
    std::vector<XWeights> x_lut(new_w);
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
            const float w = cubic_weight_mpi(u - static_cast<float>(n));

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

    // Precompute vertical bicubic weights
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
            const float w = cubic_weight_mpi(v - static_cast<float>(m));

            y_lut[local_y].row[index] = py - my_src_min; // Adjusted to my_src_min
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

    const size_t local_output_bytes = static_cast<size_t>(output_rows) * static_cast<size_t>(new_w) * 3;
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

        uint8_t* RESTRICT out = dst + static_cast<size_t>(local_y) * new_w * 3;

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

            out[out_idx] = clamp_pixel_fast(b);
            out[out_idx + 1] = clamp_pixel_fast(g);
            out[out_idx + 2] = clamp_pixel_fast(r);
        }
    }

    // --- 2. MPI-IO WRITE ---
    MPI_File fh_out;
    MPI_File_open(MPI_COMM_WORLD, output_filename,
        MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh_out);

    if (output_rows > 0)
    {
        MPI_Offset write_offset = static_cast<MPI_Offset>(output_start) * new_w * 3;
        MPI_File_write_at_all(fh_out, write_offset, local_output.data(),
            static_cast<int>(local_output_bytes), MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&fh_out);
}