#include "resizer_baseline.h"
#include <cmath>
#include <algorithm>

// Standard Bicubic Spline Weight Function
static float cubic_weight(float x) {
    x = std::fabs(x);
    float a = -0.75f;
    if (x <= 1.0f) {
        return (a + 2.0f) * (x * x * x) - (a + 3.0f) * (x * x) + 1.0f;
    }
    else if (x < 2.0f) {
        return a * (x * x * x) - 5.0f * a * (x * x) + 8.0f * a * x - 4.0f * a;
    }
    return 0.0f;
}

static uint8_t clamp_pixel(float val)
{
    if (val <= 0.0f)
        return 0;

    if (val >= 255.0f)
        return 255;

    return static_cast<uint8_t>(std::lround(val));
}

void resize_image_sequential(
    uint8_t* cpu_out,
    uint8_t* cpu_in,
    int old_w, int old_h,
    int new_w, int new_h)
{
    float x_ratio = (float)old_w / new_w;
    float y_ratio = (float)old_h / new_h;

    for (int y = 0; y < new_h; ++y) {
        float src_y = ((float)y + 0.5f) * y_ratio - 0.5f;
        int iy = (int)std::floor(src_y);
        float v = src_y - iy;

        for (int x = 0; x < new_w; ++x) {
            float src_x = ((float)x + 0.5f) * x_ratio - 0.5f;
            int ix = (int)std::floor(src_x);
            float u = src_x - ix;

            float b_sum = 0.0f;
            float g_sum = 0.0f;
            float r_sum = 0.0f;
            float total_weight = 0.0f;

            // Precalculate the 4 horizontal weights
            float wx[4];

            for (int n = -1; n <= 2; ++n) {
                wx[n + 1] = cubic_weight(u - (float)n);
            }

            // 4x4 Grid sampling (16 pixels)
            for (int m = -1; m <= 2; ++m) {

                // Calculate vertical weight once for this row
                float wy = cubic_weight(v - (float)m);

                int py = iy + m;

                if (py < 0) py = 0;
                if (py >= old_h) py = old_h - 1;

                // Accumulate horizontal weighted values first
                float b_row = 0.0f;
                float g_row = 0.0f;
                float r_row = 0.0f;

                float row_weight = 0.0f;

                for (int n = -1; n <= 2; ++n) {

                    int px = ix + n;

                    if (px < 0) px = 0;
                    if (px >= old_w) px = old_w - 1;

                    int old_offset = (py * old_w + px) * 3;

                    float horizontal_weight = wx[n + 1];

                    // Apply horizontal weight first
                    b_row += cpu_in[old_offset + 0] * horizontal_weight;
                    g_row += cpu_in[old_offset + 1] * horizontal_weight;
                    r_row += cpu_in[old_offset + 2] * horizontal_weight;

                    row_weight += horizontal_weight;
                }

                // Apply vertical weight once to the accumulated row
                b_sum += b_row * wy;
                g_sum += g_row * wy;
                r_sum += r_row * wy;

                total_weight += row_weight * wy;
            }

            if (total_weight > 0.0f) {
                b_sum /= total_weight;
                g_sum /= total_weight;
                r_sum /= total_weight;
            }

            int new_offset = (y * new_w + x) * 3;

            cpu_out[new_offset + 0] = clamp_pixel(b_sum);
            cpu_out[new_offset + 1] = clamp_pixel(g_sum);
            cpu_out[new_offset + 2] = clamp_pixel(r_sum);
        }
    }
}