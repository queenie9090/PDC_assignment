#include "resizer_openmp.h"

#include <cmath>
#include <omp.h>
#include <vector>
#include <cstdint>

static inline float cubic_weight_openmp(float x)
{
    x = std::fabs(x);
    const float a = -0.75f;

    if (x <= 1.0f) {
        return (a + 2.0f) * x * x * x
            - (a + 3.0f) * x * x
            + 1.0f;
    }
    else if (x < 2.0f) {
        return a * x * x * x
            - 5.0f * a * x * x
            + 8.0f * a * x
            - 4.0f * a;
    }

    return 0.0f;
}

static inline int reflect_101_openmp(int p, int len)
{
    if (len <= 1)
        return 0;

    while (p < 0 || p >= len)
    {
        if (p < 0) {
            p = -p;
        }
        else if (p >= len) {
            p = 2 * len - 2 - p;
        }
    }

    return p;
}

static inline uint8_t clamp_pixel_openmp(float value)
{
    long rounded = std::lround(value);

    if (rounded < 0)
        return 0;

    if (rounded > 255)
        return 255;

    return static_cast<uint8_t>(rounded);
}


// Precomputed horizontal information
struct XInfo
{
    int px[4];
    float wx[4];
};


// Precomputed vertical information
struct YInfo
{
    int py[4];
    float wy[4];
};


void resize_image_openmp(
    uint8_t* cpu_out,
    uint8_t* cpu_in,
    int old_w, int old_h,
    int new_w, int new_h)
{
    const float x_ratio =
        static_cast<float>(old_w) /
        static_cast<float>(new_w);

    const float y_ratio =
        static_cast<float>(old_h) /
        static_cast<float>(new_h);

    std::vector<XInfo> x_info(new_w);

    for (int x = 0; x < new_w; ++x)
    {
        float src_x =
            (static_cast<float>(x) + 0.5f)
            * x_ratio - 0.5f;

        int ix =
            static_cast<int>(std::floor(src_x));

        float u =
            src_x - static_cast<float>(ix);


        x_info[x].wx[0] =
            cubic_weight_openmp(u + 1.0f);

        x_info[x].wx[1] =
            cubic_weight_openmp(u);

        x_info[x].wx[2] =
            cubic_weight_openmp(1.0f - u);

        x_info[x].wx[3] =
            cubic_weight_openmp(2.0f - u);


        for (int n = 0; n < 4; ++n)
        {
            x_info[x].px[n] =
                reflect_101_openmp(
                    ix + n - 1,
                    old_w
                );
        }
    }

    std::vector<YInfo> y_info(new_h);

    for (int y = 0; y < new_h; ++y)
    {
        float src_y =
            (static_cast<float>(y) + 0.5f)
            * y_ratio - 0.5f;

        int iy =
            static_cast<int>(std::floor(src_y));

        float v =
            src_y - static_cast<float>(iy);


        y_info[y].wy[0] =
            cubic_weight_openmp(v + 1.0f);

        y_info[y].wy[1] =
            cubic_weight_openmp(v);

        y_info[y].wy[2] =
            cubic_weight_openmp(1.0f - v);

        y_info[y].wy[3] =
            cubic_weight_openmp(2.0f - v);


        for (int m = 0; m < 4; ++m)
        {
            y_info[y].py[m] =
                reflect_101_openmp(
                    iy + m - 1,
                    old_h
                );
        }
    }

#pragma omp parallel for schedule(static)
    for (int y = 0; y < new_h; ++y)
    {
        const YInfo& yi = y_info[y];

        for (int x = 0; x < new_w; ++x)
        {
            const XInfo& xi = x_info[x];

            float b_sum = 0.0f;
            float g_sum = 0.0f;
            float r_sum = 0.0f;

            for (int m = 0; m < 4; ++m)
            {
                int py = yi.py[m];
                float weight_y = yi.wy[m];

                const int row_offset =
                    py * old_w * 3;


                for (int n = 0; n < 4; ++n)
                {
                    int px = xi.px[n];

                    float weight =
                        weight_y * xi.wx[n];

                    int old_offset =
                        row_offset + px * 3;


                    b_sum +=
                        static_cast<float>(
                            cpu_in[old_offset]
                            ) * weight;

                    g_sum +=
                        static_cast<float>(
                            cpu_in[old_offset + 1]
                            ) * weight;

                    r_sum +=
                        static_cast<float>(
                            cpu_in[old_offset + 2]
                            ) * weight;
                }
            }


            const int new_offset =
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