# Utility functions

add_library(MoPhiUtils INTERFACE)

target_include_directories(MoPhiUtils INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/MoPhiEssentials/utils>
)

target_link_libraries(MoPhiUtils INTERFACE MoPhiCommon)
