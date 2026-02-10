# Common utilities and definitions

add_library(MoPhiCommon INTERFACE)

target_include_directories(MoPhiCommon INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/MoPhiEssentials/common>
)
