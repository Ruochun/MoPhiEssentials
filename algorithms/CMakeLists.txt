# Algorithms

add_library(MoPhiAlgorithms INTERFACE)

target_include_directories(MoPhiAlgorithms INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/MoPhiEssentials/algorithms>
)

target_link_libraries(MoPhiAlgorithms INTERFACE MoPhiCommon MoPhiCore)
