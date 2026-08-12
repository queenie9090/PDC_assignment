#include <iostream>
#include <chrono>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <mpi.h>
#include "resizer_baseline.h"
#include "resizer_cuda.cuh"
#include "resizer_openmp.h"
#include "resizer_mpi.h"


int main(int argc, char* argv[])
{
    // MPI INITIALIZATION
    MPI_Init(&argc, &argv);

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // CHECK ARGUMENTS
    if (argc < 5)
    {
        if (rank == 0)
        {
            std::cerr << "Error: Missing arguments.\n";
            std::cerr << "Usage: " << argv[0] << " <input> <output> <scale> <mode>\n";
        }

        MPI_Finalize();
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];
    float scale = 0.0f;

    // CHECK PATH
    if (input_path.empty() ||
        output_path.empty())
    {
        if (rank == 0)
        {
            std::cerr  << "Error: Input or Output " << "file path is empty!\n";
        }

        MPI_Finalize();
        return 1;
    }

    // READ SCALE
    try
    {
        scale = std::stof(argv[3]);
    }
    catch (const std::exception& e)
    {
        if (rank == 0)
        {
            std::cerr << "Error: Invalid scale factor '" << argv[3] << "': "  << e.what() << "\n";
        }

        MPI_Finalize();
        return 1;
    }

    std::string mode = argv[4];

    // LOAD IMAGE
    // Every MPI process loads the complete input image.
    // This means every process has access to every source pixel.
    // Therefore, halo exchange is NOT required in this design.

    cv::Mat img = cv::imread(input_path, cv::IMREAD_COLOR);

    if (img.empty())
    {
        if (rank == 0)
        {
            std::cerr << "Error: Could not load input image: "  << input_path << "\n";
        }

        MPI_Finalize();
        return 1;
    }

    // IMAGE DIMENSIONS
    int old_w = img.cols;
    int old_h = img.rows;
    int new_w = std::max( 1, static_cast<int>(old_w * scale));
    int new_h = std::max(1, static_cast<int>( old_h * scale));

    // CHECK OUTPUT DIMENSIONS
    if (new_w <= 0 ||
        new_h <= 0)
    {
        if (rank == 0)
        {
            std::cerr  << "Error: Invalid output dimensions: "  << new_w << "x" << new_h << "\n";
        }

        MPI_Finalize();
        return 1;
    }

    // ENSURE INPUT IS CONTINUOUS
    if (!img.isContinuous())
    {
        img = img.clone();
    }

    // CREATE OUTPUT IMAGE
    cv::Mat out_img =
        cv::Mat::zeros(new_h, new_w, CV_8UC3 );


    if (!out_img.isContinuous())
    {
        out_img = out_img.clone();
    }

    // CUDA WARMUP
    if (mode == "cuda")
    {
        cudaFree(0);
    }

    // MPI SYNCHRONIZATION
    if (mode == "mpi")
    {
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // START TIMER

    auto start = std::chrono::high_resolution_clock::now();

    // SELECT ALGORITHM

    if (mode == "baseline")
    {
        resize_image_sequential(out_img.data, img.data, old_w, old_h, new_w, new_h);
    }


    else if (mode == "cuda")
    {
        resize_image_cuda(out_img.data, img.data, old_w, old_h, new_w, new_h);
        cudaDeviceSynchronize();
    }


    else if (mode == "openmp")
    {
        resize_image_openmp(out_img.data, img.data, old_w, old_h, new_w, new_h);
    }


    else if (mode == "mpi")
    {
        resize_image_mpi(out_img.data, img.data, old_w, old_h, new_w, new_h);

        // Wait for all processes after gathering
        MPI_Barrier(MPI_COMM_WORLD);
    }


    else
    {
        if (rank == 0)
        {
            std::cerr << "Error: Unknown mode: " << mode << "\n";
        }

        MPI_Finalize();
        return 1;
    }

    // STOP TIMER

    auto end = std::chrono::high_resolution_clock::now();

    double duration_ms =
        std::chrono::duration<double, std::milli>(
            end - start
        ).count();

    // ONLY RANK 0 PRINTS TIME

    if (rank == 0)
    {
        std::cout
            << "TIME_MS:"
            << duration_ms
            << std::endl;
    }

    // ONLY RANK 0 SAVES OUTPUT
    // THIS IS CRITICAL.
    // Rank 1/2/3 must NOT write the image.

    if (rank == 0)
    {
        bool saved = cv::imwrite(output_path, out_img);

        if (!saved)
        {
            std::cerr << "Error: Failed to write output image: " << output_path << "\n";
            MPI_Finalize();
            return 1;
        }

        std::cout << "Output saved: " << output_path << std::endl;
    }

    // FINAL MPI SYNCHRONIZATION
    if (mode == "mpi")
    {
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // MPI FINALIZE
    MPI_Finalize();
    return 0;
}