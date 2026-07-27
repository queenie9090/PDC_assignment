#ifndef RESIZER_OPENMP_H
#define RESIZER_OPENMP_H

#include <cstdint>

void resize_image_openmp(
    uint8_t* cpu_out,
    uint8_t* cpu_in,
    int old_w, int old_h,
    int new_w, int new_h
);

#endif#pragma once
