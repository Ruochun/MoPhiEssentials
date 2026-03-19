# Copyright (c) 2025, SBEL GPU Development Team
# Copyright (c) 2025, University of Wisconsin - Madison
# 
# SPDX-License-Identifier: BSD-3-Clause

message(STATUS "==== Configuring Core Components ====")

add_library(moCore OBJECT)

target_include_directories(
    moCore
    PUBLIC $<BUILD_INTERFACE:${ProjectIncludeSource}> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PUBLIC $<BUILD_INTERFACE:${ProjectIncludeGenerated}> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(MOPHI_CUDA_AVAILABLE)
    target_link_libraries(
        moCore
        PUBLIC CUB::CUB
    )
endif()

set(core_headers
    ${CMAKE_CURRENT_SOURCE_DIR}/ApiVersion.h
    ${CMAKE_CURRENT_SOURCE_DIR}/BaseClasses.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CudaAllocator.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ManagedMemory.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ThreadManager.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Logger.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/GpuManager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/WavefrontMeshLoader.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DataMigrationHelper.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DataClasses.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Real3.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Quaternion.hpp
)

if(MOPHI_CUDA_AVAILABLE)
    set(core_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/GpuManager.cpp
    )
else()
    # In CPU-only mode, use a stub to satisfy the OBJECT library source requirement
    set(core_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/CpuOnlyStub.cpp
    )
endif()

target_sources(
    moCore
    PUBLIC ${core_headers}
    PRIVATE ${core_sources}
)

set_target_properties(
    moCore PROPERTIES 
    POSITION_INDEPENDENT_CODE True
    CXX_STANDARD ${CXXSTD_SUPPORTED}
    PUBLIC_HEADER "${core_headers}"
)

# Install Core Headers
if(PROJECT_IS_TOP_LEVEL)
    install(
        DIRECTORY ${ProjectIncludeSource}/core
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
            FILES_MATCHING 
                PATTERN "*.h"
                PATTERN "*.hpp"
    )
endif()
