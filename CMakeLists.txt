# Copyright (c) 2025, Ruochun Zhang
# SPDX-License-Identifier: BSD-3-Clause

# ---------------------------------------------------------------------------- #
# CMake Project Settings
# ---------------------------------------------------------------------------- #

cmake_minimum_required(VERSION 3.18)

# Version information
set(MOPHI_ESSENTIALS_VERSION_MAJOR 1)
set(MOPHI_ESSENTIALS_VERSION_MINOR 0)

# ---------------------------------------------------------------------------- #
# Declare the project with CXX first, then probe for CUDA
# ---------------------------------------------------------------------------- #

project(
    MoPhiEssentials
    VERSION ${MOPHI_ESSENTIALS_VERSION_MAJOR}.${MOPHI_ESSENTIALS_VERSION_MINOR}
    LANGUAGES CXX
)

# check_language(CUDA) requires an active project (CXX) to work correctly.
include(CheckLanguage)
check_language(CUDA)

# Allow the user to explicitly disable CUDA even when a CUDA compiler is found.
# Defaults to ON when a CUDA compiler is detected, OFF otherwise.
if(CMAKE_CUDA_COMPILER)
    set(MOPHI_ENABLE_CUDA_DEFAULT ON)
else()
    set(MOPHI_ENABLE_CUDA_DEFAULT OFF)
endif()
option(MOPHI_ENABLE_CUDA "Enable CUDA/GPU support" ${MOPHI_ENABLE_CUDA_DEFAULT})

if(CMAKE_CUDA_COMPILER AND MOPHI_ENABLE_CUDA)
    enable_language(CUDA)
    set(MOPHI_CUDA_AVAILABLE TRUE)
    message(STATUS "CUDA found and enabled: GPU support enabled")
else()
    set(MOPHI_CUDA_AVAILABLE FALSE)
    if(CMAKE_CUDA_COMPILER AND NOT MOPHI_ENABLE_CUDA)
        message(STATUS "CUDA disabled via MOPHI_ENABLE_CUDA=OFF: building in CPU-only mode")
    else()
        message(STATUS "CUDA not found: building in CPU-only mode (GPU functionality will be unavailable)")
    endif()
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting CMAKE_BUILD_TYPE to 'Release' (default)")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

add_compile_definitions(MOPHI_ESSENTIALS_BUILD)

if(MOPHI_CUDA_AVAILABLE)
    # Define MOPHI_USE_CUDA globally so all sub-targets (moCore, moAlgorithms,
    # demos) see it during compilation, not just the final mophi_essentials target.
    add_compile_definitions(MOPHI_USE_CUDA)
endif()

# ---------------------------------------------------------------------------- #
# Additional Packages
# ---------------------------------------------------------------------------- #

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

include(cmake/FixNinjaColors.cmake)
fix_ninja_colors()

if(MOPHI_CUDA_AVAILABLE)
    find_package(CUDAToolkit REQUIRED)

    # Find CUB library
    find_package(
        CUB REQUIRED
        HINTS ${CUDAToolkit_ROOT}/lib64/cmake/cub
    )
endif()

# ---------------------------------------------------------------------------- #
# Global Configuration
# ---------------------------------------------------------------------------- #

include(cmake/CxxStdAutodetect.cmake)

# The compiler will build against this C++ standard, if available.
set(TargetCXXStandard "STD_AUTODETECT" CACHE STRING "The C++ standard used by the compiler")
set_property(
    CACHE TargetCXXStandard
    PROPERTY
    STRINGS STD_AUTODETECT STD_CXX11 STD_CXX14 STD_CXX17 STD_CXX20
)

# Convert the standard into something CMake will understand
if(TargetCXXStandard STREQUAL STD_CXX11)
    set(CXXSTD_MAX 11)
elseif(TargetCXXStandard STREQUAL STD_CXX14)
    set(CXXSTD_MAX 14)
elseif(TargetCXXStandard STREQUAL STD_CXX17)
    set(CXXSTD_MAX 17)
elseif(TargetCXXStandard STREQUAL STD_CXX20)
    set(CXXSTD_MAX 20)
else()
    set(CXXSTD_MAX 17)
endif()
cxx_std_autodetect()

# Allow the use of #include <> for project headers
set(ProjectIncludeSource "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(ProjectIncludeGenerated "${CMAKE_BINARY_DIR}/src")

if(MOPHI_CUDA_AVAILABLE)
    # Global fix for CUDA language bug
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

#------------------------------------------------------------
# Install destinations for data and demo programs
#------------------------------------------------------------

set(MOPHI_ESSENTIALS_INSTALL_DEMO "bin")

# ---------------------------------------------------------------------------- #
# Source-level configuration
# ---------------------------------------------------------------------------- #

add_subdirectory(src/core)
if(MOPHI_CUDA_AVAILABLE)
    add_subdirectory(src/algorithms)
endif()

# ---------------------------------------------------------------------------- #
# Final Library Generation
# ---------------------------------------------------------------------------- #

if(MOPHI_CUDA_AVAILABLE)
    add_library(
        mophi_essentials 
        STATIC
        $<TARGET_OBJECTS:moCore>
        $<TARGET_OBJECTS:moAlgorithms>
    )

    # Extract include directories from the object bundles
    get_target_property(MOCORE_INTERFACE moCore INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(MOALGORITHMS_INTERFACE moAlgorithms INTERFACE_INCLUDE_DIRECTORIES) 

    # Set include directories
    target_include_directories(mophi_essentials 
        PUBLIC ${MOCORE_INTERFACE}
    )

    target_link_libraries(mophi_essentials
        PUBLIC CUDA::cudart
        PUBLIC CUDA::nvrtc
        PUBLIC CUDA::cuda_driver
    )

    # Specific to Windows...
    if(WIN32)
        target_link_libraries(mophi_essentials 
            PUBLIC dbghelp)
    endif()

    # Attach include directories to the top-level library target
    set_target_properties(mophi_essentials
        PROPERTIES
        LINKER_LANGUAGE CUDA
    )

    # Propagate MOPHI_USE_CUDA to consumers
    target_compile_definitions(mophi_essentials PUBLIC MOPHI_USE_CUDA)
else()
    # CPU-only: header-only interface library
    add_library(mophi_essentials INTERFACE)

    target_include_directories(mophi_essentials
        INTERFACE
        $<BUILD_INTERFACE:${ProjectIncludeSource}>
        $<BUILD_INTERFACE:${ProjectIncludeGenerated}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
endif()

# ---------------------------------------------------------------------------- #
# Install unified include headers
# ---------------------------------------------------------------------------- #
install(
    FILES 
        "${CMAKE_CURRENT_SOURCE_DIR}/src/MoPhiEssentials.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/MoPhiEssentialsGPU.cuh"
    DESTINATION
        ${CMAKE_INSTALL_INCLUDEDIR}
)

# ---------------------------------------------------------------------------- #
# Build embedded demos
# ---------------------------------------------------------------------------- #
# Only build demos when MoPhiEssentials is the main project, not as a submodule
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    add_subdirectory(demo)
endif()