add_executable(test_index_flat
  test_index_flat.cpp
)
target_link_libraries(test_index_flat PRIVATE lattice_index_flat GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(test_index_flat)
