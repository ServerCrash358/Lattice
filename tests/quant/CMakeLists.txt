add_executable(test_quant
  test_sq8.cpp
  test_rabitq.cpp
)
target_link_libraries(test_quant PRIVATE lattice_quant GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(test_quant)
