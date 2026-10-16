add_executable(test_kernels
  test_kernel_equivalence.cpp
)
target_link_libraries(test_kernels PRIVATE lattice_kernels GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(test_kernels)
