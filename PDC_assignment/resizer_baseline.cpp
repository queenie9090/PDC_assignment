#include "resizer_baseline.h"
#include <cmath>
#include <algorithm>
#include <cstdint>

// OpenCV's cubic spline kernel uses a = -0.75f
static inline float cubic_weight(float x) {
    x = std::fabs(x);
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
static inline int reflect_101(int p, int len) {
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

// C++11 compatible clamp implementation
static inline uint8_t clamp_pixel(float val) {
    long rounded = std::lround(val);
    if (rounded < 0) return 0;
    if (rounded > 255) return 255;
    return static_cast<uint8_t>(rounded);
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

        // Precompute vertical weights for this row
        float wy[4];
        wy[0] = cubic_weight(v + 1.0f);
        wy[1] = cubic_weight(v);
        wy[2] = cubic_weight(1.0f - v);
        wy[3] = cubic_weight(2.0f - v);

        for (int x = 0; x < new_w; ++x) {
            float src_x = ((float)x + 0.5f) * x_ratio - 0.5f;
            int ix = (int)std::floor(src_x);
            float u = src_x - ix;

            // Precompute horizontal weights for this pixel
            float wx[4];
            wx[0] = cubic_weight(u + 1.0f);
            wx[1] = cubic_weight(u);
            wx[2] = cubic_weight(1.0f - u);
            wx[3] = cubic_weight(2.0f - u);

            float b_sum = 0.0f, g_sum = 0.0f, r_sum = 0.0f;

            // 4x4 Grid sampling (16 pixels)
            for (int m = 0; m < 4; ++m) {
                int py = reflect_101(iy + m - 1, old_h);
                float weight_y = wy[m];

                for (int n = 0; n < 4; ++n) {
                    int px = reflect_101(ix + n - 1, old_w);
                    float weight = weight_y * wx[n];

                    int old_offset = (py * old_w + px) * 3;

                    b_sum += cpu_in[old_offset + 0] * weight;
                    g_sum += cpu_in[old_offset + 1] * weight;
                    r_sum += cpu_in[old_offset + 2] * weight;
                }
            }

            int new_offset = (y * new_w + x) * 3;
            cpu_out[new_offset + 0] = clamp_pixel(b_sum);
            cpu_out[new_offset + 1] = clamp_pixel(g_sum);
            cpu_out[new_offset + 2] = clamp_pixel(r_sum);
        }
    }
}