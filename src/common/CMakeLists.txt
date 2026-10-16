add_library(lattice_common INTERFACE)
target_include_directories(lattice_common INTERFACE
  ${CMAKE_SOURCE_DIR}/include
)
target_compile_features(lattice_common INTERFACE cxx_std_23)
