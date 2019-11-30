cmake_minimum_required(VERSION 3.10)
project(SMPL)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_language(CUDA)

set(xtensor_DIR  /home/nponomareva/usr/local/lib/cmake/xtensor)
set(xtl_DIR  /home/nponomareva/usr/local/lib/cmake/xtl)
set(nlohmann_json_DIR  /home/nponomareva/usr/local/lib/cmake/nlohmann_json)
set(GTest_DIR /home/nponomareva/usr/local/lib/cmake/GTest)
set(GTEST_LIBRARY /home/nponomareva/usr/local/lib/libgtest.a)
set(GTEST_MAIN_LIBRARY /home/nponomareva/usr/local/lib/libgtest_main.a)
set(GTEST_INCLUDE_DIR /home/nponomareva/usr/local/include/gtest)


include_directories(/usr/local/include)

# xtensor
find_package(xtensor REQUIRED)
# json
find_package(nlohmann_json REQUIRED)
#GTest required
find_package(GTest REQUIRED)


file(GLOB HEADER "smpl/*.h")
file(GLOB SOURCE "*.cpp" "smpl/*.cpp" "tests/*.cpp" "smpl/cuda/*.cu" "tests/*.cu")

set(PROJECT_INCLUDE_DIRS
        ${SMPL_INCLUDE_DIRS}
        ${GLEW_INCLUDE_DIRS}
        ${xtensor_INCLUDE_DIRS}
        )
##
## LIBRARIES
##
set(PROJECT_LIBRARIES
        xtensor
        ${xtensor_blas_LIBRARIES}
        nlohmann_json::nlohmann_json
        ${GTEST_LIBRARY}
        )

set(CMAKE_EXE_LINKER_FLAGS "")

add_executable(SMPL ${HEADER} ${SOURCE})
target_include_directories(SMPL PRIVATE ${PROJECT_INCLUDE_DIRS})
target_link_libraries(SMPL ${PROJECT_LIBRARIES})
