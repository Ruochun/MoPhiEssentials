#  Copyright (c) 2025, Ruochun Zhang
#  SPDX-License-Identifier: BSD-3-Clause

message(STATUS "==== Configuring GPU Static Algorithm Components ====")

set (moAlgorithms_message "[moAlgorithms]")



add_library(moAlgorithms OBJECT)

target_include_directories(
	moAlgorithms	
	PUBLIC ${ProjectIncludeSource}
	PUBLIC ${ProjectIncludeGenerated}
)

target_link_libraries(
	moAlgorithms
	PUBLIC CUB::CUB
)

### HOST HEADERS ONLY (.h, .hpp) ###
set(moAlgorithms_interface
	${CMAKE_CURRENT_SOURCE_DIR}/StaticDeviceSubroutines.h
)

### INTERNAL HEADERS ONLY (.h, .hpp, or .cuh) ###
set(moAlgorithms_headers
	${CMAKE_CURRENT_SOURCE_DIR}/Utilities.cuh
	${CMAKE_CURRENT_SOURCE_DIR}/CubWrappers.cuh
)

set(moAlgorithms_sources
	${CMAKE_CURRENT_SOURCE_DIR}/CubInstantiations.cu
	${CMAKE_CURRENT_SOURCE_DIR}/DynamicsMisc.cu
)

target_sources(
	moAlgorithms
	PUBLIC ${moAlgorithms_interface}
	INTERFACE ${moAlgorithms_headers}
	PRIVATE ${moAlgorithms_sources}
)

# Get a list of supported architectures
include(../../cmake/CudaSupportedArchitectures.cmake)
cuda_supported_architectures()

# Convert the cuda supported architectures into their respective CMake properties
set(virtual_architectures "")
foreach(arch ${CUDASUP_ARCHITECTURES})
	list(APPEND virtual_architectures "${arch}-virtual")
endforeach()

set_target_properties(
	moAlgorithms PROPERTIES 
	POSITION_INDEPENDENT_CODE True
	CXX_STANDARD ${CXXSTD_SUPPORTED}
	PUBLIC_HEADER "${moAlgorithms_interface}"
	CUDA_ARCHITECTURES "${virtual_architectures}"
)

install(
	TARGETS moAlgorithms
	PUBLIC_HEADER
		DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/algorithms
)
