find_package(Threads REQUIRED)

add_executable(test_index_graph
  test_vamana.cpp
  test_vamana_concurrency.cpp
)
target_link_libraries(test_index_graph PRIVATE lattice_index_graph lattice_index_flat
                                                GTest::gtest_main Threads::Threads)
include(GoogleTest)
gtest_discover_tests(test_index_graph)
