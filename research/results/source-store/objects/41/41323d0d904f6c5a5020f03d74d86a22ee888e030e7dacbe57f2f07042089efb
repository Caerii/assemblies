#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "brain.h"

namespace py = pybind11;

class PyBrain {
public:
    PyBrain(float p, float beta, float max_weight, uint32_t seed) 
        : brain_(p, beta, max_weight, seed) {}
    
    void AddArea(const std::string& name, uint32_t n, uint32_t k, 
                 bool recurrent = true, bool is_explicit = false) {
        brain_.AddArea(name, n, k, recurrent, is_explicit);
    }
    
    void AddStimulus(const std::string& name, uint32_t k) {
        brain_.AddStimulus(name, k);
    }
    
    void AddFiber(const std::string& from, const std::string& to, 
                  bool bidirectional = false) {
        brain_.AddFiber(from, to, bidirectional);
    }
    
    void Project(const std::map<std::string, std::vector<std::string>>& graph, 
                 uint32_t num_steps, bool update_plasticity = true) {
        brain_.Project(graph, num_steps, update_plasticity);
    }
    
    void ActivateArea(const std::string& name, uint32_t assembly_index) {
        brain_.ActivateArea(name, assembly_index);
    }
    
    void InhibitAll() {
        brain_.InhibitAll();
    }
    
    void InhibitFiber(const std::string& from, const std::string& to) {
        brain_.InhibitFiber(from, to);
    }
    
    void ActivateFiber(const std::string& from, const std::string& to) {
        brain_.ActivateFiber(from, to);
    }
    
    void InitProjection(const std::map<std::string, std::vector<std::string>>& graph) {
        brain_.InitProjection(graph);
    }
    
    // Get activated neurons for an area
    std::vector<uint32_t> GetActivated(const std::string& name) {
        const auto& area = brain_.GetArea(name);
        return area.activated;
    }
    
    // Get area properties
    uint32_t GetAreaN(const std::string& name) {
        return brain_.GetArea(name).n;
    }
    
    uint32_t GetAreaK(const std::string& name) {
        return brain_.GetArea(name).k;
    }
    
    uint32_t GetAreaSupport(const std::string& name) {
        return brain_.GetArea(name).support;
    }
    
    void SetLogLevel(int log_level) {
        brain_.SetLogLevel(log_level);
    }
    
    void LogGraphStats() {
        brain_.LogGraphStats();
    }
    
    void LogActivated(const std::string& area_name) {
        brain_.LogActivated(area_name);
    }
    
    // Read assembly with overlap calculation
    std::pair<size_t, size_t> ReadAssembly(const std::string& name) {
        size_t index, overlap;
        brain_.ReadAssembly(name, index, overlap);
        return std::make_pair(index, overlap);
    }

private:
    nemo::Brain brain_;
};

PYBIND11_MODULE(brain_cpp, m) {
    m.doc() = "High-performance C++ Brain simulation module";
    
    py::class_<PyBrain>(m, "Brain")
        .def(py::init<float, float, float, uint32_t>(),
             py::arg("p"), py::arg("beta"), py::arg("max_weight"), py::arg("seed"))
        .def("add_area", &PyBrain::AddArea,
             py::arg("name"), py::arg("n"), py::arg("k"), 
             py::arg("recurrent") = true, py::arg("is_explicit") = false)
        .def("add_stimulus", &PyBrain::AddStimulus,
             py::arg("name"), py::arg("k"))
        .def("add_fiber", &PyBrain::AddFiber,
             py::arg("from"), py::arg("to"), py::arg("bidirectional") = false)
        .def("project", &PyBrain::Project,
             py::arg("graph"), py::arg("num_steps"), py::arg("update_plasticity") = true)
        .def("activate_area", &PyBrain::ActivateArea,
             py::arg("name"), py::arg("assembly_index"))
        .def("inhibit_all", &PyBrain::InhibitAll)
        .def("inhibit_fiber", &PyBrain::InhibitFiber,
             py::arg("from"), py::arg("to"))
        .def("activate_fiber", &PyBrain::ActivateFiber,
             py::arg("from"), py::arg("to"))
        .def("init_projection", &PyBrain::InitProjection,
             py::arg("graph"))
        .def("get_activated", &PyBrain::GetActivated,
             py::arg("name"))
        .def("get_area_n", &PyBrain::GetAreaN,
             py::arg("name"))
        .def("get_area_k", &PyBrain::GetAreaK,
             py::arg("name"))
        .def("get_area_support", &PyBrain::GetAreaSupport,
             py::arg("name"))
        .def("set_log_level", &PyBrain::SetLogLevel,
             py::arg("log_level"))
        .def("log_graph_stats", &PyBrain::LogGraphStats)
        .def("log_activated", &PyBrain::LogActivated,
             py::arg("area_name"))
        .def("read_assembly", &PyBrain::ReadAssembly,
             py::arg("name"));
}
