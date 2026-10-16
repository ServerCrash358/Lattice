find_library(URING_LIBRARY NAMES uring)

add_library(lattice_storage STATIC
  storage.cpp
)
target_include_directories(lattice_storage PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(lattice_storage PUBLIC lattice_common)
target_link_libraries(lattice_storage PRIVATE ${URING_LIBRARY})
