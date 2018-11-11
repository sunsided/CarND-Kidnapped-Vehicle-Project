project(PARTICLE_FILTER)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS "${CXX_FLAGS}")

set(NUM_PARTICLES 1000 CACHE STRING "The number of particles to use.")
option(USE_ASSOCIATION_NAIVE "Use brute-force nearest neighbors for association." OFF)

set(sources src/particle_filter.cpp src/main.cpp src/data_association_libssrckdtree.cpp)

include_directories(vendor/libssrckdtree)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

include_directories(/usr/local/include)
include_directories(/usr/local/opt/openssl/include)
link_directories(/usr/local/lib)
link_directories(/usr/local/opt/openssl/lib)
link_directories(/usr/local/Cellar/libuv/1*/lib)

endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin") 


add_executable(particle_filter ${sources})

target_link_libraries(particle_filter z ssl uv uWS)

target_compile_definitions(particle_filter PRIVATE NUM_PARTICLES=${NUM_PARTICLES})
if(USE_ASSOCIATION_NAIVE)
    target_compile_definitions(particle_filter PRIVATE ASSOCIATION_NAIVE=1)
endif(USE_ASSOCIATION_NAIVE)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()