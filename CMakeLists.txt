cmake_minimum_required(VERSION 3.16)
project(opencv_cuda_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use same compiler as for OpenCV
set(CMAKE_C_COMPILER /usr/bin/gcc-12)
set(CMAKE_CXX_COMPILER /usr/bin/g++-12)

#Export compile_commands.json for tools/IDE
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED COMPONENTS
  core
  imgproc
  highgui
  cudaarithm
  cudaimgproc
  cudafilters

)
find_package(OpenMP REQUIRED)


add_executable(perforamnceTest src/run_track.cpp)
target_link_libraries(perforamnceTest PRIVATE ${OpenCV_LIBS})
target_link_libraries(perforamnceTest PRIVATE OpenMP::OpenMP_CXX)
