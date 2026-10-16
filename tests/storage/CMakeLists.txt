add_executable(test_storage
  test_storage.cpp
)
target_link_libraries(test_storage PRIVATE lattice_storage GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(test_storage)
