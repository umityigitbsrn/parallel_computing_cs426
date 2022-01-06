cmake_minimum_required(VERSION 3.21)
project(project_04 CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(project_04 demo.cu util.cu util.h kmer_serial.cu)

set_target_properties(project_04 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
