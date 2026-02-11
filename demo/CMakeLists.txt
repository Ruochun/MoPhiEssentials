#  Copyright (c) 2025, Ruochun Zhang
#  SPDX-License-Identifier: BSD-3-Clause

# ------------------------------------------------------------------------------
# Additional include paths and libraries
# ------------------------------------------------------------------------------

SET(LIBRARIES
    mophi_essentials
)

# ------------------------------------------------------------------------------
# List of all executables
# ------------------------------------------------------------------------------

SET(DEMOS
    HelloWorld.cpp
    TestMsg.cpp
    TestDualArray.cpp
    TestContainers.cpp
    TestReal3.cu
    TestUnifiedHeader.cpp
    TestUnifiedGPUHeader.cu
)

# ------------------------------------------------------------------------------
# Add all executables
# ------------------------------------------------------------------------------

message(STATUS "Demo programs for MoPhiEssentials...")

FOREACH(PROGRAM_FILE ${DEMOS})
    
    get_filename_component(PROGRAM ${PROGRAM_FILE} NAME_WE)
    message(STATUS "...add ${PROGRAM}")

    add_executable(${PROGRAM} "${PROGRAM_FILE}")

    set_target_properties(
        ${PROGRAM} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${MOPHI_ESSENTIALS_INSTALL_DEMO}"
    )

    source_group("" FILES "${PROGRAM_FILE}")
    
    target_link_libraries(${PROGRAM} 
        PUBLIC ${LIBRARIES}
    )
    
    add_dependencies(${PROGRAM} ${LIBRARIES})

    set_target_properties(
        ${PROGRAM} PROPERTIES
        CXX_STANDARD ${CXXSTD_SUPPORTED}
    )

ENDFOREACH(PROGRAM_FILE)