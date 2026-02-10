# Core functionality

add_library(MoPhiCore INTERFACE)

target_include_directories(MoPhiCore INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/MoPhiEssentials/core>
)

target_link_libraries(MoPhiCore INTERFACE MoPhiCommon)
