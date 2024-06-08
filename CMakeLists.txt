cmake_minimum_required(VERSION 3.19.1)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

project(Exact CUDA CXX)

include_directories(${PROJECT_SOURCE_DIR}/headers)

file(GLOB sources ${PROJECT_SOURCE_DIR}/sources/*.cu
                  ${PROJECT_SOURCE_DIR}/sources/*.cpp)
file(GLOB headers ${PROJECT_SOURCE_DIR}/headers/*.h)

add_executable(Exact main.cu ${headers} ${sources})

target_link_libraries(Exact gmp gmpxx)

set_target_properties(Exact PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
