#ifndef RESIZER_MPI_H
#define RESIZER_MPI_H

#include <cstdint>

void resize_image_mpi(
    uint8_t* cpu_out,
    const uint8_t* cpu_in,
    int old_w,
    int old_h,
    int new_w,
    int new_h
);

#endif