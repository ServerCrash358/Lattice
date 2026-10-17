cmake_minimum_required(VERSION 3.25)
project(lattice CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LATTICE_BUILD_TESTS "Build unit/property tests" ON)
option(LATTICE_BUILD_PYTHON "Build the Python extension module (needs pybind11)" OFF)
# Only the pieces the Python package needs (no RocksDB/CRoaring, tests, benches or examples).
option(LATTICE_PYTHON_ONLY "Build just what the Python package links" OFF)
if(LATTICE_BUILD_PYTHON)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # static libs are linked into a shared module
endif()
if(LATTICE_PYTHON_ONLY)
  set(LATTICE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

add_compile_options(-Wall -Wextra)

# --- Test frameworks (D-CPP-06) ---
# Phase 1 pragmatic choice: FetchContent instead of vcpkg for GoogleTest.
# D-CPP-05's Rec is vcpkg manifest mode; vcpkg's from-source bootstrap is a
# poor fit for a 4-core/16GB dev VM on a slow first checkout. The actual
# dependency (GoogleTest, per D-CPP-06) is unchanged -- only the fetch
# mechanism differs. Revisit when Phase 3 needs RocksDB/CRoaring, where
# vcpkg's binary caching starts to pay for itself.
if(LATTICE_BUILD_TESTS)
  include(FetchContent)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
  )
  FetchContent_MakeAvailable(googletest)
  enable_testing()
endif()

add_subdirectory(src/common)
add_subdirectory(src/kernels)
add_subdirectory(src/quant)
if(NOT LATTICE_PYTHON_ONLY)
  add_subdirectory(src/index_flat)
endif()
add_subdirectory(src/index_graph)
if(NOT LATTICE_PYTHON_ONLY)
  add_subdirectory(src/filter)
endif()
add_subdirectory(src/storage)
add_subdirectory(src/segment)
if(NOT LATTICE_PYTHON_ONLY AND EXISTS "${CMAKE_SOURCE_DIR}/bench/cpp/CMakeLists.txt")
  add_subdirectory(bench/cpp)
endif()

if(LATTICE_BUILD_PYTHON)
  add_subdirectory(python)
endif()

if(NOT LATTICE_PYTHON_ONLY AND EXISTS "${CMAKE_SOURCE_DIR}/examples/CMakeLists.txt")
  add_subdirectory(examples)
endif()

if(LATTICE_BUILD_TESTS)
  add_subdirectory(tests/common)
  add_subdirectory(tests/kernels)
  add_subdirectory(tests/quant)
  add_subdirectory(tests/index_flat)
  add_subdirectory(tests/index_graph)
  add_subdirectory(tests/filter)
  add_subdirectory(tests/storage)
  add_subdirectory(tests/segment)
endif()
