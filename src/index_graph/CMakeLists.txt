add_library(lattice_index_graph STATIC
  index_graph.cpp
)
target_include_directories(lattice_index_graph PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(lattice_index_graph PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(lattice_index_graph PUBLIC lattice_common lattice_kernels lattice_quant)
