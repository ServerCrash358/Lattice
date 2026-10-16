add_library(lattice_index_flat STATIC
  index_flat.cpp
)
target_include_directories(lattice_index_flat PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(lattice_index_flat PUBLIC lattice_common lattice_kernels)
