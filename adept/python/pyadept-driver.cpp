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


#define ALNPTR2ARRAY(array)                                                              \
    def(#array, [](ADEPT::aln_results &aln){                                             \
        py::capsule free_array(aln.array, [](void *f)                                    \
        {                                                                                \
            short *foo = reinterpret_cast<short *>(f);                                   \
            ;                                                                            \
        });                                                                              \
                                                                                         \
        return py::array_t<short>({aln.size}, {sizeof(short)}, aln.array);               \
    })

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
    .def("set", &ADEPT::gap_scores::set_scores, "Set gap scores")
    .def("get", &ADEPT::gap_scores::get_scores, "Get current gap scores")
    .def("__repr__",
        [](const ADEPT::gap_scores &a) 
        { 
            return "<pyadept.gap_scores: open: " + std::to_string(a.open) + " extend: " + std::to_string(a.extend) + ">";
        });

    //
    // struct aln_results
    //
    py::class_<ADEPT::aln_results>(adp, "alignments", "Alignment results from ADEPT")
    .ALNPTR2ARRAY(top_scores)
    .ALNPTR2ARRAY(ref_begin)
    .ALNPTR2ARRAY(ref_end)
    .ALNPTR2ARRAY(query_begin)
    .ALNPTR2ARRAY(query_end)
    .def_readonly("size", &ADEPT::aln_results::size);

    //
    // struct all_aln
    //
    py::class_<ADEPT::all_alns>(adp, "multiAlign", "MultiGPU alignments")
    .def(py::init<int>())
    .def_readonly("results", &ADEPT::all_alns::results)
    .def_readonly("per_gpu", &ADEPT::all_alns::per_gpu)
    .def_readonly("left_over", &ADEPT::all_alns::left_over)
    .def_readonly("gpus", &ADEPT::all_alns::gpus);
}

//
// bind the driver class
//
void driver(py::module &adp)
{
    py::class_<ADEPT::driver>(adp, "driver", "The ADEPT alignment driver")
    .def(py::init<>())
    .def("initialize", static_cast<void (ADEPT::driver::*)(ShortList &, ADEPT::gap_scores, ADEPT::options::ALG_TYPE, ADEPT::options::SEQ_TYPE, ADEPT::options::CIGAR, int, int, int, int, int)>(&ADEPT::driver::initialize), "Initialize the ADEPT driver")
    .def("kernel_launch", &ADEPT::driver::kernel_launch, "Launch the asynchronous alignment kernel", py::arg("ref_seqs"), py::arg("query_seqs"), py::arg("res_offset") = 0)
    .def("get_alignments", &ADEPT::driver::get_alignments, "Get ADEPT alignment results")
    .def("mem_cpy_dth", &ADEPT::driver::mem_cpy_dth, "Asynchronously copy the alignment results from device to host", py::arg("offset") = 0)
    .def("kernel_done", &ADEPT::driver::kernel_done, "Query if the asynchronous kernel is done")
    .def("dth_done", &ADEPT::driver::kernel_done, "Query if the alignment results have been copied from device to host")
    .def("dth_synch", &ADEPT::driver::dth_synch, "Synchronize the D2H transfer")
    .def("kernel_synch", &ADEPT::driver::kernel_synch, "Synchronize the SYCL kernel")
    .def("cleanup", &ADEPT::driver::cleanup, "Clean up the ADEPT driver")
    .def("set_gap_scores", &ADEPT::driver::set_gap_scores, "Set driver's gap scores");
}

//
// bind functions
//
void functions(py::module &adp)
{
    //
    // function: get_batch_size
    //
    adp.def("get_batch_size", &ADEPT::get_batch_size, "Get maximum alignment batch size for SYCL device@gpu_id", py::arg("gpu_id") = 0, py::arg("max_q_size"), py::arg("max_r_size"), py::arg("per_gpu_mem") = 100);

    //
    // function: multi_gpu
    //
    adp.def("multiGPU", &ADEPT::multi_gpu, "Automatic Multiple GPU support for ADEPT alignments", py::arg("ref_sequences"), py::arg("que_sequences"), py::arg("algorithm"), py::arg("sequences"), py::arg("cigar_avail"), py::arg("max_ref_size"), py::arg("max_que_size"), py::arg("scores"), py::arg("gaps"), py::arg("batch_size_") = -1);
}

} // namespace pydriver