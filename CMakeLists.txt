cmake_minimum_required(VERSION 3.10)
project(DynamicFusion)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(Torch_DIR /usr/local/Cellar/libtorch/share/cmake/Torch)
set(xtensor_DIR /usr/local/Cellar/xtensor/lib/cmake/xtensor)
set(xtl_DIR //anaconda3/pkgs/xtl-0.6.5-h770b8ee_0/lib/cmake/xtl)
set(nlohmann_json_DIR /Users/Natalia/Documents/nlohmann_json/build)

include_directories(/usr/local/include)

# Torch
find_package(Torch REQUIRED)
# xtensor
find_package(xtensor REQUIRED)
# json
find_package(nlohmann_json REQUIRED)
#GTest required
find_package(GTest REQUIRED)


file(GLOB HEADER "include/smpl/*.h" "include/tests/*.h" "include/*.h")
file(GLOB SOURCE "src/smpl/*.cpp" "src/tests/*.cpp" "src/*.cpp")

##
## INCLUDE DIRECTORIES
##
set(SMPLXX_INCLUDE_DIRS "include")

set(PROJECT_INCLUDE_DIRS
        ${SMPLXX_INCLUDE_DIRS}
        ${GLEW_INCLUDE_DIRS}
        ${xtensor_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
        )

##
## LIBRARIES
##
set(PROJECT_LIBRARIES
        xtensor
        ${xtensor_blas_LIBRARIES}
        nlohmann_json::nlohmann_json
        ${TORCH_LIBRARIES}
        # stdc++fs                        # std::experimental::filesystem
        /usr/local/lib/libgtest.a
        )

set(CMAKE_EXE_LINKER_FLAGS "")

add_executable(DynamicFusion ${HEADER} ${SOURCE})
target_include_directories(DynamicFusion PRIVATE ${PROJECT_INCLUDE_DIRS})
target_link_libraries(DynamicFusion ${PROJECT_LIBRARIES})