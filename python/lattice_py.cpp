// Python bindings: build an index from a float32 numpy array, open it disk-resident (zero-copy
// mmap), and search. The GIL is released during build and search, so Python threads issuing
// concurrent searches run in parallel.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "lattice/quant/sq8.h"
#include "lattice/segment/disk_index.h"

namespace py = pybind11;
using namespace lattice;
using namespace lattice::index_graph;

namespace {
using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

Metric parse_metric(const std::string& m) {
    if (m == "l2") return Metric::kL2;
    if (m == "ip" || m == "cosine") return Metric::kInnerProduct;  // cosine: caller normalizes
    throw std::invalid_argument("metric must be 'l2', 'ip' or 'cosine'");
}

void build_index(FloatArray data, const std::string& path, const std::string& metric, std::uint32_t r,
                 std::uint32_t build_list, std::uint32_t build_threads) {
    if (data.ndim() != 2) throw std::invalid_argument("data must be a 2-D float32 array (n, dim)");
    const auto n = static_cast<std::size_t>(data.shape(0));
    const auto dim = static_cast<std::uint32_t>(data.shape(1));
    if (n == 0 || dim == 0) throw std::invalid_argument("data must be non-empty");
    const float* base = data.data();
    VamanaBuildParams params;
    params.r = r;
    params.build_list_size = build_list;
    params.metric = parse_metric(metric);
    params.build_threads = build_threads;

    std::string error;
    {
        py::gil_scoped_release release;
        std::vector<const float*> ptrs(n);
        for (std::size_t i = 0; i < n; ++i) ptrs[i] = base + i * dim;
        auto book = quant::Sq8CodeBook::train(ptrs, dim);
        if (!book) {
            error = book.error().message;
        } else {
            quant::Sq8Codes codes(*book);
            auto layout = NodeRecordLayout::make_compact(dim, ElemType::kFloat32, r);
            if (!layout) {
                error = layout.error().message;
            } else {
                auto index = VamanaIndex::build(ptrs, dim, codes, params, *layout);
                if (!index) {
                    error = index.error().message;
                } else if (auto s = index->serialize_to_disk(path); !s) {
                    error = s.error().message;
                }
            }
        }
    }
    if (!error.empty()) throw std::runtime_error(error);
}

class PyIndex {
public:
    PyIndex(const std::string& path, bool preload) {
        auto opened = segment::open_disk_resident_vamana(path, segment::NodeAccess::kMmap, preload);
        if (!opened) throw std::runtime_error(opened.error().message);
        index_ = std::make_unique<VamanaIndex>(std::move(*opened));
        dim_ = index_->layout().dim;
    }

    py::array_t<std::int64_t> search(FloatArray query, std::uint32_t k, std::uint32_t ef_search) const {
        if (query.ndim() != 1 || static_cast<std::uint32_t>(query.shape(0)) != dim_) {
            throw std::invalid_argument("query must be a 1-D float32 array of length dim");
        }
        SearchParams sp;
        sp.ef_search = ef_search;
        Result<std::vector<VectorId>> r;
        {
            py::gil_scoped_release release;
            r = index_->search(query.data(), k, sp);
        }
        if (!r) throw std::runtime_error(r.error().message);
        py::array_t<std::int64_t> out(static_cast<py::ssize_t>(r->size()));
        auto* o = out.mutable_data();
        for (std::size_t i = 0; i < r->size(); ++i) o[i] = static_cast<std::int64_t>((*r)[i]);
        return out;
    }

    std::size_t size() const { return index_->size(); }
    std::uint32_t dim() const { return dim_; }

private:
    std::unique_ptr<VamanaIndex> index_;
    std::uint32_t dim_ = 0;
};
}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "Lattice: disk-resident vector index (Vamana graph, SQ8 codes, mmap node store)";
    m.def("build_index", &build_index, py::arg("data"), py::arg("path"), py::arg("metric") = "l2",
          py::arg("r") = 64, py::arg("build_list") = 100, py::arg("build_threads") = 0,
          "Build an index from an (n, dim) float32 array and save it to `path`. Row i gets id i.");
    py::class_<PyIndex>(m, "Index")
        .def(py::init<const std::string&, bool>(), py::arg("path"), py::arg("preload") = false)
        .def("search", &PyIndex::search, py::arg("query"), py::arg("k") = 10, py::arg("ef_search") = 64,
             "Return the ids of the k nearest neighbours (best first).")
        .def("__len__", &PyIndex::size)
        .def_property_readonly("dim", &PyIndex::dim);
}
