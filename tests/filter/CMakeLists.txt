add_executable(test_filter
  test_filter.cpp
)
target_link_libraries(test_filter PRIVATE lattice_filter GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(test_filter)
