cmake_minimum_required(VERSION 3.20)
project(codes C)

set(CMAKE_C_STANDARD 99)

add_executable(codes src/util.c src/util.h src/kdtree_serial.c)
