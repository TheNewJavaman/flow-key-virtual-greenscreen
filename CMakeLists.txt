cmake_minimum_required(VERSION 3.19)
project(Flow_Key_Virtual_Greenscreen)

set(CMAKE_CXX_STANDARD 14)

add_executable(Flow_Key_Virtual_Greenscreen
    app/build/resources/main/net/javaman/flowkey/hardwareapis/cuda/CudaKernels.cu
    app/src/main/resources/net/javaman/flowkey/hardwareapis/cuda/CudaKernels.cu)

set_target_properties(Flow_Key_Virtual_Greenscreen PROPERTIES LINKER_LANGUAGE CXX)