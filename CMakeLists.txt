cmake_minimum_required(VERSION 3.10)
project(DynamicFusion)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_PREFIX_PATH "C:/Users/Natalia/Downloads/libtorch/share/cmake/Torch")
set(Torch_LIBRARY "C:/Users/Natalia/Downloads/libtorch")

find_package(Torch REQUIRED)

add_executable(DynamicFusion main.cpp def.h def.cpp smpl/smpl.h smpl/smpl.cpp
        smpl/exception.cpp smpl/exception.h smpl/smpl_maths.cpp smpl/TorchEx.h tests/Singleton.h tests/Tester.cpp)