# Disk-resident index access: depends only on index_graph, quant and storage (no RocksDB), so the
# Python package can link it on its own.
add_library(lattice_disk_index STATIC
  disk_index.cpp
)
target_include_directories(lattice_disk_index PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(lattice_disk_index PUBLIC
  lattice_common lattice_storage lattice_quant lattice_index_graph
)

if(NOT LATTICE_PYTHON_ONLY)
  find_library(ROCKSDB_LIBRARY NAMES rocksdb)
  find_path(ROCKSDB_INCLUDE_DIR NAMES rocksdb/db.h)

  add_library(lattice_segment STATIC
    segment.cpp
    rocksdb_fault_injection.cpp
  )
  target_include_directories(lattice_segment PUBLIC ${CMAKE_SOURCE_DIR}/include)
  target_include_directories(lattice_segment PRIVATE ${ROCKSDB_INCLUDE_DIR})
  target_link_libraries(lattice_segment PUBLIC
    lattice_common lattice_storage lattice_filter lattice_quant
    lattice_index_graph lattice_index_flat lattice_disk_index
  )
  # RocksDB is a dependency of `segment` ONLY (D-CPP-06) -- no other module links it.
  target_link_libraries(lattice_segment PRIVATE ${ROCKSDB_LIBRARY})
endif()
