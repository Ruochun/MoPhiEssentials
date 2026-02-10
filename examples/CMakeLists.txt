# Examples and demos

# Simple vector example
add_executable(example_vector example_vector.cpp)
target_link_libraries(example_vector PRIVATE MoPhiEssentials)

# Simple mesh example
add_executable(example_mesh example_mesh.cpp)
target_link_libraries(example_mesh PRIVATE MoPhiEssentials)

# Algorithm examples
add_executable(example_algorithms example_algorithms.cpp)
target_link_libraries(example_algorithms PRIVATE MoPhiEssentials)
