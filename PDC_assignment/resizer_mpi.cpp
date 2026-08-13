#include "resizer_mpi.h"

#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <mpi.h>

// Cross-platform pointer restrict definition
#if defined(_MSC_VER)
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

// ============================================================
// HELPER FUNCTIONS (C++11 Compatible)
// ============================================================

static inline int clamp_int(int val, int low, int high)
{
    return (val < low) ? low : ((val > high) ? high : val);
}

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
        return (a + 2.0f) * (x * x * x)
            - (a + 3.0f) * (x * x)
            + 1.0f;
    }

    if (x < 2.0f)
    {
        return a * (x * x * x)
            - 5.0f * a * (x * x)
            + 8.0f * a * x
            - 4.0f * a;
    }

    return 0.0f;
}

// ============================================================
// PRECOMPUTED HORIZONTAL LOOKUP STRUCT
// ============================================================
struct XWeights {
    int px_offset[4]; // Byte offsets for RGB channels (px * 3)
    float wx[4];      // Pre-normalized horizontal weights
};

// ============================================================
// MAIN ULTRA-FAST PURE MPI RESIZER
// ============================================================
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

    const float x_ratio = static_cast<float>(old_w) / static_cast<float>(new_w);
    const float y_ratio = static_cast<float>(old_h) / static_cast<float>(new_h);

    // --------------------------------------------------------
    // 1. Calculate Output Row Division & Source Row Ranges
    // --------------------------------------------------------
    int base_rows = new_h / size;
    int extra_rows = new_h % size;

    int output_rows = base_rows + (rank < extra_rows ? 1 : 0);
    int output_start = rank * base_rows + std::min(rank, extra_rows);

    std::vector<int> src_min_py(size, 0);
    std::vector<int> src_max_py(size, 0);
    std::vector<int> src_row_count(size, 0);

    for (int r = 0; r < size; ++r)
    {
        int r_rows = base_rows + (r < extra_rows ? 1 : 0);
        int r_start = r * base_rows + std::min(r, extra_rows);

        if (r_rows == 0)
        {
            src_min_py[r] = 0;
            src_max_py[r] = 0;
            src_row_count[r] = 0;
            continue;
        }

        float y_min_src = (static_cast<float>(r_start) + 0.5f) * y_ratio - 0.5f;
        int iy_min = static_cast<int>(std::floor(y_min_src));
        src_min_py[r] = clamp_int(iy_min - 1, 0, old_h - 1);

        int r_end = r_start + r_rows - 1;
        float y_max_src = (static_cast<float>(r_end) + 0.5f) * y_ratio - 0.5f;
        int iy_max = static_cast<int>(std::floor(y_max_src));
        src_max_py[r] = clamp_int(iy_max + 2, 0, old_h - 1);

        src_row_count[r] = src_max_py[r] - src_min_py[r] + 1;
    }

    int my_min_py = src_min_py[rank];
    int my_src_rows = src_row_count[rank];
    size_t my_input_bytes = static_cast<size_t>(my_src_rows) * old_w * 3;

    std::vector<uint8_t> local_input(my_input_bytes);

    // --------------------------------------------------------
    // 2. High-Speed Domain Slicing (Transfer exact rows needed)
    // --------------------------------------------------------
    if (rank == 0)
    {
        // Copy Rank 0's local slice directly
        size_t rank0_offset = static_cast<size_t>(my_min_py) * old_w * 3;
        std::copy(cpu_in + rank0_offset, cpu_in + rank0_offset + my_input_bytes, local_input.begin());

        // Send minimal required slices to workers
        for (int r = 1; r < size; ++r)
        {
            size_t r_offset = static_cast<size_t>(src_min_py[r]) * old_w * 3;
            int r_bytes = src_row_count[r] * old_w * 3;

            MPI_Send(cpu_in + r_offset, r_bytes, MPI_UNSIGNED_CHAR, r, 0, MPI_COMM_WORLD);
        }
    }
    else
    {
        if (my_input_bytes > 0)
        {
            MPI_Recv(local_input.data(), static_cast<int>(my_input_bytes), MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // Allocate local output buffer
    size_t local_bytes = static_cast<size_t>(output_rows) * new_w * 3;
    std::vector<uint8_t> local_output(local_bytes);

    // --------------------------------------------------------
    // 3. Precompute Pre-Normalized Horizontal Lookup Table (LUT)
    // --------------------------------------------------------
    std::vector<XWeights> x_lut(new_w);

    for (int x = 0; x < new_w; ++x)
    {
        float src_x = (static_cast<float>(x) + 0.5f) * x_ratio - 0.5f;
        int ix = static_cast<int>(std::floor(src_x));
        float u = src_x - static_cast<float>(ix);

        float sum_wx = 0.0f;
        for (int n = -1; n <= 2; ++n)
        {
            int px = clamp_int(ix + n, 0, old_w - 1);
            int idx = n + 1;

            x_lut[x].px_offset[idx] = px * 3;
            float w = cubic_weight_mpi(u - static_cast<float>(n));
            x_lut[x].wx[idx] = w;
            sum_wx += w;
        }

        // Normalize weights so sum == 1.0 (eliminates inner-loop division)
        if (sum_wx > 0.0f)
        {
            float inv_wx = 1.0f / sum_wx;
            x_lut[x].wx[0] *= inv_wx;
            x_lut[x].wx[1] *= inv_wx;
            x_lut[x].wx[2] *= inv_wx;
            x_lut[x].wx[3] *= inv_wx;
        }
    }

    // --------------------------------------------------------
    // 4. Ultra-Fast Core Interpolation Loop
    // --------------------------------------------------------
    const uint8_t* RESTRICT local_src_base = local_input.data();
    uint8_t* RESTRICT dst_base = local_output.data();

    for (int local_y = 0; local_y < output_rows; ++local_y)
    {
        int y = output_start + local_y;

        float src_y = (static_cast<float>(y) + 0.5f) * y_ratio - 0.5f;
        int iy = static_cast<int>(std::floor(src_y));
        float v = src_y - static_cast<float>(iy);

        float wy[4];
        const uint8_t* RESTRICT row_ptrs[4];
        float sum_wy = 0.0f;

        for (int m = -1; m <= 2; ++m)
        {
            int py = clamp_int(iy + m, 0, old_h - 1);
            int idx = m + 1;

            float w = cubic_weight_mpi(v - static_cast<float>(m));
            wy[idx] = w;
            sum_wy += w;

            int local_py = py - my_min_py;
            row_ptrs[idx] = local_src_base + static_cast<size_t>(local_py) * old_w * 3;
        }

        // Pre-normalize vertical weights
        if (sum_wy > 0.0f)
        {
            float inv_wy = 1.0f / sum_wy;
            wy[0] *= inv_wy;
            wy[1] *= inv_wy;
            wy[2] *= inv_wy;
            wy[3] *= inv_wy;
        }

        uint8_t* RESTRICT out_ptr = dst_base + static_cast<size_t>(local_y) * new_w * 3;

        const uint8_t* RESTRICT r0 = row_ptrs[0];
        const uint8_t* RESTRICT r1 = row_ptrs[1];
        const uint8_t* RESTRICT r2 = row_ptrs[2];
        const uint8_t* RESTRICT r3 = row_ptrs[3];

        float wy0 = wy[0], wy1 = wy[1], wy2 = wy[2], wy3 = wy[3];

        for (int x = 0; x < new_w; ++x)
        {
            const XWeights& x_info = x_lut[x];

            int o0 = x_info.px_offset[0];
            int o1 = x_info.px_offset[1];
            int o2 = x_info.px_offset[2];
            int o3 = x_info.px_offset[3];

            float wx0 = x_info.wx[0];
            float wx1 = x_info.wx[1];
            float wx2 = x_info.wx[2];
            float wx3 = x_info.wx[3];

            // Horizontal row dot-products
            float h0_b = r0[o0] * wx0 + r0[o1] * wx1 + r0[o2] * wx2 + r0[o3] * wx3;
            float h0_g = r0[o0 + 1] * wx0 + r0[o1 + 1] * wx1 + r0[o2 + 1] * wx2 + r0[o3 + 1] * wx3;
            float h0_r = r0[o0 + 2] * wx0 + r0[o1 + 2] * wx1 + r0[o2 + 2] * wx2 + r0[o3 + 2] * wx3;

            float h1_b = r1[o0] * wx0 + r1[o1] * wx1 + r1[o2] * wx2 + r1[o3] * wx3;
            float h1_g = r1[o0 + 1] * wx0 + r1[o1 + 1] * wx1 + r1[o2 + 1] * wx2 + r1[o3 + 1] * wx3;
            float h1_r = r1[o0 + 2] * wx0 + r1[o1 + 2] * wx1 + r1[o2 + 2] * wx2 + r1[o3 + 2] * wx3;

            float h2_b = r2[o0] * wx0 + r2[o1] * wx1 + r2[o2] * wx2 + r2[o3] * wx3;
            float h2_g = r2[o0 + 1] * wx0 + r2[o1 + 1] * wx1 + r2[o2 + 1] * wx2 + r2[o3 + 1] * wx3;
            float h2_r = r2[o0 + 2] * wx0 + r2[o1 + 2] * wx1 + r2[o2 + 2] * wx2 + r2[o3 + 2] * wx3;

            float h3_b = r3[o0] * wx0 + r3[o1] * wx1 + r3[o2] * wx2 + r3[o3] * wx3;
            float h3_g = r3[o0 + 1] * wx0 + r3[o1 + 1] * wx1 + r3[o2 + 1] * wx2 + r3[o3 + 1] * wx3;
            float h3_r = r3[o0 + 2] * wx0 + r3[o1 + 2] * wx1 + r3[o2 + 2] * wx2 + r3[o3 + 2] * wx3;

            // Vertical combination
            float b_sum = h0_b * wy0 + h1_b * wy1 + h2_b * wy2 + h3_b * wy3;
            float g_sum = h0_g * wy0 + h1_g * wy1 + h2_g * wy2 + h3_g * wy3;
            float r_sum = h0_r * wy0 + h1_r * wy1 + h2_r * wy2 + h3_r * wy3;

            int out_idx = x * 3;
            out_ptr[out_idx + 0] = clamp_pixel_fast(b_sum);
            out_ptr[out_idx + 1] = clamp_pixel_fast(g_sum);
            out_ptr[out_idx + 2] = clamp_pixel_fast(r_sum);
        }
    }

    // --------------------------------------------------------
    // 5. Gather Output Buffer directly to Rank 0
    // --------------------------------------------------------
    std::vector<int> recv_counts(size);
    std::vector<int> displacements(size);

    if (rank == 0)
    {
        for (int r = 0; r < size; ++r)
        {
            int r_rows = base_rows + (r < extra_rows ? 1 : 0);
            int r_start = r * base_rows + std::min(r, extra_rows);

            recv_counts[r] = r_rows * new_w * 3;
            displacements[r] = r_start * new_w * 3;
        }
    }

    MPI_Gatherv(
        local_output.data(),
        static_cast<int>(local_bytes),
        MPI_UNSIGNED_CHAR,

        cpu_out,
        recv_counts.data(),
        displacements.data(),
        MPI_UNSIGNED_CHAR,
        0,
        MPI_COMM_WORLD
    );
}