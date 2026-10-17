find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(_core MODULE lattice_py.cpp)
target_link_libraries(_core PRIVATE lattice_disk_index lattice_quant)
install(TARGETS _core LIBRARY DESTINATION lattice_vdb)
