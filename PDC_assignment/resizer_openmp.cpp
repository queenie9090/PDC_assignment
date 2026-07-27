#include "resizer_openmp.h"

#include <cmath>
#include <omp.h>

// Same bicubic weight function as the baseline version
static float cubic_weight_openmp(float x)
{
    x = std::fabs(x);
    const float a = -0.5f;

    if (x <= 1.0f) {
        return (a + 2.0f) * (x * x * x)
            - (a + 3.0f) * (x * x)
            + 1.0f;
    }
    else if (x < 2.0f) {
        return a * (x * x * x)
            - 5.0f * a * (x * x)
            + 8.0f * a * x
            - 4.0f * a;
    }

    return 0.0f;
}

// Keep pixel values between 0 and 255
static uint8_t clamp_pixel_openmp(float value)
{
    if (value < 0.0f) {
        return 0;
    }

    if (value > 255.0f) {
        return 255;
    }

    return static_cast<uint8_t>(value);
}

void resize_image_openmp(
    uint8_t* cpu_out,
    uint8_t* cpu_in,
    int old_w, int old_h,
    int new_w, int new_h)
{
    const float x_ratio =
        static_cast<float>(old_w) / static_cast<float>(new_w);

    const float y_ratio =
        static_cast<float>(old_h) / static_cast<float>(new_h);

    /*
        Each thread receives different output rows.

        Shared:
        - cpu_out
        - cpu_in
        - image dimensions
        - x_ratio and y_ratio

        Private:
        - y and x
        - src_x and src_y
        - ix and iy
        - u and v
        - colour sums
        - loop variables m and n
        - offsets and weights

        Variables declared inside the loop are automatically private.
    */
#pragma omp parallel for schedule(static)
    for (int y = 0; y < new_h; ++y) {

        float src_y =
            (static_cast<float>(y) + 0.5f) * y_ratio - 0.5f;

        int iy = static_cast<int>(std::floor(src_y));
        float v = src_y - static_cast<float>(iy);

        for (int x = 0; x < new_w; ++x) {

            float src_x =
                (static_cast<float>(x) + 0.5f) * x_ratio - 0.5f;

            int ix = static_cast<int>(std::floor(src_x));
            float u = src_x - static_cast<float>(ix);

            float b_sum = 0.0f;
            float g_sum = 0.0f;
            float r_sum = 0.0f;
            float total_weight = 0.0f;

            // Sample the surrounding 4 × 4 pixel area
            for (int m = -1; m <= 2; ++m) {

                float wy =
                    cubic_weight_openmp(v - static_cast<float>(m));

                int py = iy + m;

                if (py < 0) {
                    py = 0;
                }

                if (py >= old_h) {
                    py = old_h - 1;
                }

                for (int n = -1; n <= 2; ++n) {

                    float wx =
                        cubic_weight_openmp(u - static_cast<float>(n));

                    int px = ix + n;

                    if (px < 0) {
                        px = 0;
                    }

                    if (px >= old_w) {
                        px = old_w - 1;
                    }

                    float weight = wx * wy;

                    int old_offset =
                        (py * old_w + px) * 3;

                    b_sum +=
                        static_cast<float>(cpu_in[old_offset]) * weight;

                    g_sum +=
                        static_cast<float>(cpu_in[old_offset + 1]) * weight;

                    r_sum +=
                        static_cast<float>(cpu_in[old_offset + 2]) * weight;

                    total_weight += weight;
                }
            }

            if (total_weight > 0.0f) {
                b_sum /= total_weight;
                g_sum /= total_weight;
                r_sum /= total_weight;
            }

            int new_offset =
                (y * new_w + x) * 3;

            cpu_out[new_offset] =
                clamp_pixel_openmp(b_sum);

            cpu_out[new_offset + 1] =
                clamp_pixel_openmp(g_sum);

            cpu_out[new_offset + 2] =
                clamp_pixel_openmp(r_sum);
        }
    }
}