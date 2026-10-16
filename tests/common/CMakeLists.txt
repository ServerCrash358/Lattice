add_executable(test_common
  test_aligned_buffer.cpp
  test_row_filter.cpp
  test_top_k_heap.cpp
  test_grouped_top_k.cpp
)
target_link_libraries(test_common PRIVATE lattice_common GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(test_common)
