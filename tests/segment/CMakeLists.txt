add_executable(test_segment
  test_segment.cpp
  test_disk_vamana.cpp
  test_merge.cpp
  test_ram_budget.cpp
)
target_link_libraries(test_segment PRIVATE lattice_segment GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(test_segment)

# Crash-injection child process (D-STO-06): a standalone binary, not part of
# the GTest executable, so it can be fork+exec'd and deliberately _exit()
# mid-operation without taking the test runner down with it.
add_executable(crash_harness_main crash_harness_main.cpp)
target_link_libraries(crash_harness_main PRIVATE lattice_segment)

add_executable(test_crash_injection test_crash_injection.cpp)
target_link_libraries(test_crash_injection PRIVATE lattice_segment lattice_storage GTest::gtest_main)
target_compile_definitions(test_crash_injection PRIVATE
  CRASH_HARNESS_PATH="$<TARGET_FILE:crash_harness_main>")
add_dependencies(test_crash_injection crash_harness_main)
gtest_discover_tests(test_crash_injection)

# Needs RocksDB headers directly (constructs rocksdb::Options/DB::Open
# itself, to inject the custom FileSystem at open time) -- its own target
# so lattice_segment's own RocksDB visibility stays PRIVATE (D-CPP-06:
# RocksDB linked only by segment); this test binary is explicitly allowed
# to see RocksDB since it's testing segment's RocksDB adapter directly.
find_library(ROCKSDB_LIBRARY_TEST NAMES rocksdb)
find_path(ROCKSDB_INCLUDE_DIR_TEST NAMES rocksdb/db.h)
add_executable(test_rocksdb_fault_injection test_rocksdb_fault_injection.cpp)
target_include_directories(test_rocksdb_fault_injection PRIVATE ${ROCKSDB_INCLUDE_DIR_TEST})
target_link_libraries(test_rocksdb_fault_injection PRIVATE lattice_segment ${ROCKSDB_LIBRARY_TEST} GTest::gtest_main)
gtest_discover_tests(test_rocksdb_fault_injection)
