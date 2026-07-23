#ifndef RESIZER_BASELINE_H
#define RESIZER_BASELINE_H

#include <cstdint>

void resize_image_sequential(
    uint8_t* cpu_out,
    uint8_t* cpu_in,
    int old_w, int old_h,
    int new_w, int new_h
);

#endif