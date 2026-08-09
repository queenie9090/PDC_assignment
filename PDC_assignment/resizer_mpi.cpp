#include "resizer_mpi.h"

#include <mpi.h>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

// ============================================================
// Standard Bicubic Spline Weight Function
// ============================================================

static float cubic_weight(float x)
{
    x = std::fabs(x);

    float a = -0.5f;

    if (x <= 1.0f)
    {
        return (a + 2.0f) * (x * x * x)
            - (a + 3.0f) * (x * x)
            + 1.0f;
    }
    else if (x < 2.0f)
    {
        return a * (x * x * x)
            - 5.0f * a * (x * x)
            + 8.0f * a * x
            - 4.0f * a;
    }

    return 0.0f;
}

// ============================================================
// Clamp pixel value to [0, 255]
// ============================================================

static uint8_t clamp_pixel(float val)
{
    if (val < 0.0f)
        return 0;

    if (val > 255.0f)
        return 255;

    return static_cast<uint8_t>(val);
}

// ============================================================
// MPI Bicubic Image Resizing
// ============================================================

void resize_image_mpi(
    uint8_t* cpu_out,
    const uint8_t* cpu_in,
    int old_w,
    int old_h,
    int new_w,
    int new_h)
{
    int rank;
    int size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // --------------------------------------------------------
    // 1. Calculate input/output image sizes
    // --------------------------------------------------------

    int input_bytes =
        old_w * old_h * 3;

    int output_bytes =
        new_w * new_h * 3;

    // --------------------------------------------------------
    // 2. Broadcast input image to all MPI processes
    // --------------------------------------------------------

    std::vector<uint8_t> input_image(input_bytes);

    if (rank == 0)
    {
        std::copy(
            cpu_in,
            cpu_in + input_bytes,
            input_image.begin()
        );
    }

    MPI_Bcast(
        input_image.data(),
        input_bytes,
        MPI_UINT8_T,
        0,
        MPI_COMM_WORLD
    );

    // --------------------------------------------------------
    // 3. Divide output rows among MPI processes
    // --------------------------------------------------------

    int base_rows = new_h / size;
    int extra_rows = new_h % size;

    int start_row;

    int local_rows;

    if (rank < extra_rows)
    {
        local_rows = base_rows + 1;
        start_row = rank * local_rows;
    }
    else
    {
        local_rows = base_rows + 1;
        start_row =
            extra_rows * (base_rows + 1)
            + (rank - extra_rows) * base_rows;

        local_rows = base_rows;
    }

    // --------------------------------------------------------
    // 4. Allocate local output buffer
    // --------------------------------------------------------

    int local_bytes =
        local_rows * new_w * 3;

    std::vector<uint8_t> local_output(local_bytes);

    // --------------------------------------------------------
    // 5. Bicubic interpolation for assigned rows
    // --------------------------------------------------------

    float x_ratio =
        static_cast<float>(old_w) / new_w;

    float y_ratio =
        static_cast<float>(old_h) / new_h;

    for (int local_y = 0;
        local_y < local_rows;
        ++local_y)
    {
        // Convert local row to global output row
        int y = start_row + local_y;

        float src_y =
            (static_cast<float>(y) + 0.5f)
            * y_ratio
            - 0.5f;

        int iy =
            static_cast<int>(std::floor(src_y));

        float v = src_y - iy;

        for (int x = 0;
            x < new_w;
            ++x)
        {
            float src_x =
                (static_cast<float>(x) + 0.5f)
                * x_ratio
                - 0.5f;

            int ix =
                static_cast<int>(std::floor(src_x));

            float u = src_x - ix;

            float b_sum = 0.0f;
            float g_sum = 0.0f;
            float r_sum = 0.0f;

            float total_weight = 0.0f;

            // ------------------------------------------------
            // 4 × 4 neighbourhood
            // ------------------------------------------------

            for (int m = -1; m <= 2; ++m)
            {
                float wy =
                    cubic_weight(
                        v - static_cast<float>(m)
                    );

                int py = iy + m;

                // Clamp Y coordinate
                if (py < 0)
                    py = 0;

                if (py >= old_h)
                    py = old_h - 1;

                for (int n = -1; n <= 2; ++n)
                {
                    float wx =
                        cubic_weight(
                            u - static_cast<float>(n)
                        );

                    int px = ix + n;

                    // Clamp X coordinate
                    if (px < 0)
                        px = 0;

                    if (px >= old_w)
                        px = old_w - 1;

                    float weight = wx * wy;

                    int old_offset =
                        (py * old_w + px) * 3;

                    b_sum +=
                        input_image[old_offset + 0]
                        * weight;

                    g_sum +=
                        input_image[old_offset + 1]
                        * weight;

                    r_sum +=
                        input_image[old_offset + 2]
                        * weight;

                    total_weight += weight;
                }
            }

            // ------------------------------------------------
            // Normalize result
            // ------------------------------------------------

            if (total_weight > 0.0f)
            {
                b_sum /= total_weight;
                g_sum /= total_weight;
                r_sum /= total_weight;
            }

            // ------------------------------------------------
            // Store local output pixel
            // ------------------------------------------------

            int local_offset =
                (local_y * new_w + x) * 3;

            local_output[local_offset + 0] =
                clamp_pixel(b_sum);

            local_output[local_offset + 1] =
                clamp_pixel(g_sum);

            local_output[local_offset + 2] =
                clamp_pixel(r_sum);
        }
    }

    // --------------------------------------------------------
    // 6. Prepare counts/displacements for MPI_Gatherv
    // --------------------------------------------------------

    std::vector<int> recv_counts(size);
    std::vector<int> displacements(size);

    for (int process = 0;
        process < size;
        ++process)
    {
        int process_rows;

        if (process < extra_rows)
            process_rows = base_rows + 1;
        else
            process_rows = base_rows;

        recv_counts[process] =
            process_rows * new_w * 3;

        if (process == 0)
        {
            displacements[process] = 0;
        }
        else
        {
            displacements[process] =
                displacements[process - 1]
                + recv_counts[process - 1];
        }
    }

    // --------------------------------------------------------
    // 7. Gather all local results at process 0
    // --------------------------------------------------------

    MPI_Gatherv(
        local_output.data(),
        local_bytes,
        MPI_UINT8_T,
        cpu_out,
        recv_counts.data(),
        displacements.data(),
        MPI_UINT8_T,
        0,
        MPI_COMM_WORLD
    );
}