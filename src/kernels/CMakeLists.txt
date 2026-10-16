add_library(lattice_kernels STATIC
  scalar.cpp
  avx2.cpp
  avxvnni.cpp
  avx512.cpp
  kernel_table.cpp
)
target_include_directories(lattice_kernels PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(lattice_kernels PUBLIC lattice_common)

# Each ISA-specific translation unit gets exactly the flags it needs so the
# rest of the binary stays portable; KernelTable::resolve() (D-VEC-04) never
# calls into a TU whose ISA the CPU didn't report at runtime.
set_source_files_properties(avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
set_source_files_properties(avxvnni.cpp PROPERTIES COMPILE_OPTIONS "-mavxvnni;-mavx2")
set_source_files_properties(avx512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f")
