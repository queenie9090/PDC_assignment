#include <iostream>
#include <chrono>
#include <string>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h> // Included for cudaDeviceSynchronize safety

// Include all resizers
#include "resizer_baseline.h"  // Baseline Sequential
#include "resizer_cuda.cuh"    // CUDA
//#include "resizer_openmp.h"  // OpenMP
//#include "resizer_mpi.h"     // MPI

int main(int argc, char* argv[]) {
    // Usage: PDC_assignment.exe <input_path> <output_path> <scale_factor> <mode>
    if (argc < 5) {
        std::cerr << "Error: Missing arguments.\n";
        std::cerr << "Usage: " << argv[0] << " <input> <output> <scale> <mode>\n";
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];
    float scale = 0.0f;

    // SAFEGUARD 1: Check for empty string arguments
    if (input_path.empty() || output_path.empty()) {
        std::cerr << "Error: Input or Output file path argument is empty!\n";
        return 1;
    }

    // SAFEGUARD 2: Catch std::stof conversion exceptions
    try {
        scale = std::stof(argv[3]);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: Invalid scale factor value '" << argv[3] << "': " << e.what() << "\n";
        return 1;
    }

    std::string mode = argv[4]; // "baseline", "cuda", "openmp", "mpi"

    // 1. Load image using OpenCV
    cv::Mat img = cv::imread(input_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error: Could not load input image: " << input_path << "\n";
        return 1;
    }

    int old_w = img.cols;
    int old_h = img.rows;
    int new_w = static_cast<int>(old_w * scale);
    int new_h = static_cast<int>(old_h * scale);

    // Ensure output dimensions are valid
    if (new_w <= 0 || new_h <= 0) {
        std::cerr << "Error: Calculated output dimensions are invalid (" << new_w << "x" << new_h << ")\n";
        return 1;
    }

    // Prepare allocated continuous output buffer
    cv::Mat out_img = cv::Mat::zeros(new_h, new_w, CV_8UC3);

    // Ensure OpenCV memory layout is continuous
    if (!img.isContinuous() || !out_img.isContinuous()) {
        img = img.clone();
        out_img = out_img.clone();
    }

    // GPU Warmup (Initializes CUDA context before clock starts)
    if (mode == "cuda") {
        cudaFree(0);
    }

    // 2. Start High-Resolution Timer
    auto start = std::chrono::high_resolution_clock::now();

    // 3. Execute selected algorithm
    if (mode == "baseline") {
        resize_image_sequential(out_img.data, img.data, old_w, old_h, new_w, new_h);
    }
    else if (mode == "cuda") {
        resize_image_cuda(out_img.data, img.data, old_w, old_h, new_w, new_h);
        cudaDeviceSynchronize(); // Ensure kernel completes inside function
    }
    else if (mode == "openmp") {
        // resize_image_openmp(out_img.data, img.data, old_w, old_h, new_w, new_h);
    }
    else if (mode == "mpi") {
        // resize_image_mpi(out_img.data, img.data, old_w, old_h, new_w, new_h);
    }
    else {
        std::cerr << "Error: Unknown mode specified: " << mode << "\n";
        return 1;
    }

    // 4. Stop Timer
    auto end = std::chrono::high_resolution_clock::now();
    double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // 5. Output Execution Time for Python regex parsing
    std::cout << "TIME_MS:" << duration_ms << std::endl;

    // 6. Save output image
    bool saved = cv::imwrite(output_path, out_img);
    if (!saved) {
        std::cerr << "Error: Failed to write output image to: " << output_path << "\n";
        return 1;
    }

    return 0;
}