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
        return (a + 2.0f) * (x * x * x) - (a + 3.0f) * (x * x) + 1.0f;
    }
    else if (x < 2.0f) {
        return a * (x * x * x) - 5.0f * a * (x * x) + 8.0f * a * x - 4.0f * a;
    }

    return 0.0f;
}


static inline uint8_t clamp_pixel_openmp(float value)
{
    if (value <= 0.0f) {
        return 0;
    }

    if (value >= 255.0f) {
        return 255;
    }

    return static_cast<uint8_t>(std::lround(value));
}


// use struct to store precomputed infos
struct XInfo
{
    int px[4];
    float wx[4];
};


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
    const float x_ratio = static_cast<float>(old_w) / static_cast<float>(new_w);

    const float y_ratio = static_cast<float>(old_h) / static_cast<float>(new_h);


    std::vector<XInfo> x_info(new_w);

    for (int x = 0; x < new_w; ++x)
    {
        float src_x =
            (static_cast<float>(x) + 0.5f)
            * x_ratio
            - 0.5f;

        int ix =
            static_cast<int>(std::floor(src_x));

        float u =
            src_x - static_cast<float>(ix);


        for (int n = -1; n <= 2; ++n)
        {
            int index = n + 1;

            int px = ix + n;

            if (px < 0) {
                px = 0;
            }

            if (px >= old_w) {
                px = old_w - 1;
            }

            x_info[x].px[index] = px;

            x_info[x].wx[index] =
                cubic_weight_openmp(
                    u - static_cast<float>(n)
                );
        }
    }


    std::vector<YInfo> y_info(new_h);

    for (int y = 0; y < new_h; ++y)
    {
        float src_y =
            (static_cast<float>(y) + 0.5f)
            * y_ratio
            - 0.5f;

        int iy =
            static_cast<int>(std::floor(src_y));

        float v =
            src_y - static_cast<float>(iy);


        for (int m = -1; m <= 2; ++m)
        {
            int index = m + 1;

            int py = iy + m;

            if (py < 0) {
                py = 0;
            }

            if (py >= old_h) {
                py = old_h - 1;
            }

            y_info[y].py[index] = py;

            y_info[y].wy[index] =
                cubic_weight_openmp(
                    v - static_cast<float>(m)
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

            float total_weight = 0.0f;


            for (int m = 0; m < 4; ++m)
            {
                const float wy = yi.wy[m];

                const int row_offset = yi.py[m] * old_w * 3;

                for (int n = 0; n < 4; ++n)
                {
                    const float weight =
                        xi.wx[n] * wy;

                    const int old_offset =
                        row_offset
                        + xi.px[n] * 3;


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


                    total_weight += weight;
                }
            }


            if (total_weight > 0.0f)
            {
                const float inverse_weight =
                    1.0f / total_weight;

                b_sum *= inverse_weight;
                g_sum *= inverse_weight;
                r_sum *= inverse_weight;
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