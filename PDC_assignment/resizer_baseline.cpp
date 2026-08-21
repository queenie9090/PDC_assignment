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

/*
static uint8_t clamp_pixel(float val) {
    if (val < 0.0f) return 0;
    if (val > 255.0f) return 255;
    return static_cast<uint8_t>(val);
}
*/

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

            float b_sum = 0.0f, g_sum = 0.0f, r_sum = 0.0f;
            float total_weight = 0.0f;

            // 4x4 Grid sampling (16 pixels)
            for (int m = -1; m <= 2; ++m) {
                float wy = cubic_weight(v - (float)m);
                int py = iy + m;
                if (py < 0) py = 0;
                if (py >= old_h) py = old_h - 1;

                for (int n = -1; n <= 2; ++n) {
                    float wx = cubic_weight(u - (float)n);
                    int px = ix + n;
                    if (px < 0) px = 0;
                    if (px >= old_w) px = old_w - 1;

                    float weight = wx * wy;
                    int old_offset = (py * old_w + px) * 3;

                    b_sum += cpu_in[old_offset + 0] * weight;
                    g_sum += cpu_in[old_offset + 1] * weight;
                    r_sum += cpu_in[old_offset + 2] * weight;

                    total_weight += weight;
                }
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