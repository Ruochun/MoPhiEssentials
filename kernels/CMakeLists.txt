# Computational kernels

add_library(MoPhiKernels INTERFACE)

target_include_directories(MoPhiKernels INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/MoPhiEssentials/kernels>
)

target_link_libraries(MoPhiKernels INTERFACE MoPhiCommon MoPhiCore)

if(MOPHI_ENABLE_CUDA)
    target_link_libraries(MoPhiKernels INTERFACE CUDA::cudart)
endif()
