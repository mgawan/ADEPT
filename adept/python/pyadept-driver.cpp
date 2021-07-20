// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "pyadept-driver.hpp"

namespace py = pybind11;
using namespace py::literals;

namespace pydriver
{

//
// bind options
//
void options(py::module &adp)
{
    // define a submodule for Adept
    py::module_ opts = adp.def_submodule("options", "ADEPT Configuration Options");

    py::enum_<ADEPT::options::ALG_TYPE>(opts, "ALG_TYPE", "Alignment Algorithm: Smith-Waterman (SW) or Needleman-Wunsch (NW)")
    .value("SW", ADEPT::options::ALG_TYPE::SW)
    .value("NW", ADEPT::options::ALG_TYPE::NW)
    .export_values();

    py::enum_<ADEPT::options::CIGAR>(opts, "CIGAR", "CIGAR Availability: NO or YES")
    .value("NO", ADEPT::options::CIGAR::NO)
    .value("YES", ADEPT::options::CIGAR::YES)
    .export_values();

    py::enum_<ADEPT::options::SEQ_TYPE>(opts, "SEQ_TYPE", "Sequence Type: DNA or Amino Acid (AA)")
    .value("DNA", ADEPT::options::SEQ_TYPE::DNA)
    .value("AA", ADEPT::options::SEQ_TYPE::AA)
    .export_values();
}

//
// bind structs
//
void structs(py::module &adp)
{
    //
    // struct gap_scores
    //
    py::class_<ADEPT::gap_scores>(adp, "gap_scores", "Gap scores for ADEPT alignment")
    .def(py::init<>())
    .def(py::init<short, short>())
    .def_readwrite("open", &ADEPT::gap_scores::open)
    .def_readwrite("extend", &ADEPT::gap_scores::extend)
    .def("set", &ADEPT::gap_scores::set_scores)
    .def("get", &ADEPT::gap_scores::get_scores)
    .def("__repr__",
        [](const ADEPT::gap_scores &a) 
        { 
            return "<pyadept.gap_scores: open: " + std::to_string(a.open) + " extend: " + std::to_string(a.extend) + ">";
        });

    //
    // struct 
    //

}

//
// bind the driver class
//
void driver(py::module &adp)
{

}

//
// bind functions
//
void functions(py::module &adp)
{
    //
    // function: get_batch_size
    //
    adp.def("get_batch_size", &ADEPT::get_batch_size, "Get maximum alignment batch size for sycl::device@gpu_id", py::arg("gpu_id") = 0, py::arg("max_q_size"), py::arg("max_r_size"), py::arg("per_gpu_mem") = 100);
}

} // namespace pydriver