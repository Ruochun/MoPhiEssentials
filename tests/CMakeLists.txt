# Tests for MoPhiEssentials

# Find or fetch a testing framework (e.g., Google Test)
# For now, we'll create a simple test structure

enable_testing()

# Test utilities
add_executable(test_vector test_vector.cpp)
target_link_libraries(test_vector PRIVATE MoPhiEssentials)
add_test(NAME test_vector COMMAND test_vector)

add_executable(test_mesh test_mesh.cpp)
target_link_libraries(test_mesh PRIVATE MoPhiEssentials)
add_test(NAME test_mesh COMMAND test_mesh)

# Add more tests as needed
