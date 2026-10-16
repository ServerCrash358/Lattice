find_package(PkgConfig QUIET)
find_library(ROARING_LIBRARY NAMES roaring)
find_path(ROARING_INCLUDE_DIR NAMES roaring/roaring.h)

add_library(lattice_filter STATIC
  filter.cpp
)
target_include_directories(lattice_filter PUBLIC ${CMAKE_SOURCE_DIR}/include ${ROARING_INCLUDE_DIR})
target_link_libraries(lattice_filter PUBLIC lattice_common ${ROARING_LIBRARY})
