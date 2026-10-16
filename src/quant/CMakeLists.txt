add_library(lattice_quant STATIC
  sq8.cpp
  rabitq.cpp
)
target_include_directories(lattice_quant PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(lattice_quant PUBLIC lattice_common lattice_kernels)
