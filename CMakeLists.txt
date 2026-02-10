cmake_minimum_required(VERSION 3.15)
project(MoPhiEssentials VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(MOPHI_BUILD_TESTS "Build tests" ON)
option(MOPHI_BUILD_EXAMPLES "Build examples" ON)
option(MOPHI_ENABLE_CUDA "Enable CUDA support" OFF)

# Find CUDA if enabled
if(MOPHI_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
)

# Add subdirectories
add_subdirectory(common)
add_subdirectory(core)
add_subdirectory(algorithms)
add_subdirectory(kernels)
add_subdirectory(utils)

# Create main library
add_library(MoPhiEssentials INTERFACE)
target_include_directories(MoPhiEssentials INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Link component libraries
target_link_libraries(MoPhiEssentials INTERFACE
    MoPhiCommon
    MoPhiCore
    MoPhiAlgorithms
    MoPhiKernels
    MoPhiUtils
)

# Tests
if(MOPHI_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Examples
if(MOPHI_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Installation
install(TARGETS MoPhiEssentials MoPhiCommon MoPhiCore MoPhiAlgorithms MoPhiKernels MoPhiUtils
    EXPORT MoPhiEssentialsTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY common core algorithms kernels utils
    DESTINATION include/MoPhiEssentials
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# Export targets
install(EXPORT MoPhiEssentialsTargets
    FILE MoPhiEssentialsTargets.cmake
    NAMESPACE MoPhiEssentials::
    DESTINATION lib/cmake/MoPhiEssentials
)

# Create config file
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/MoPhiEssentialsConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/MoPhiEssentialsConfig.cmake
    INSTALL_DESTINATION lib/cmake/MoPhiEssentials
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/MoPhiEssentialsConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/MoPhiEssentialsConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/MoPhiEssentialsConfigVersion.cmake
    DESTINATION lib/cmake/MoPhiEssentials
)
