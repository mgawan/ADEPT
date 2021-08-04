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

//
// module: pyadept
//

#include "pyadept.hpp"
#include "pyadept-driver.hpp"
#include "pybind11/stl_bind.h"

namespace py = pybind11;
using namespace py::literals;

//
// adept module
//
PYBIND11_MODULE(adept, adp) 
{
    adp.doc() = "Python interface for GPU-Inpendent ADEPT SYCL";

    // bind opaque types before anything else
    opaques(adp);

    // bind enums
    pydriver::options(adp);

    // bind structs
    pydriver::structs(adp);

    // bind driver
    pydriver::driver(adp);

    // bind functions
    pydriver::functions(adp);

}

//
// bind STL opaques before anything else
//
void opaques(py::module &adp)
{
    // bind vector<int>
    py::bind_vector<IntList>(adp, "IntList", py::module_local(false));

    // bind vector<short>
    py::bind_vector<ShortList>(adp, "ShortList", py::module_local(false));

    // bind vector<string>
    py::bind_vector<StringList>(adp, "StringList", py::module_local(false));
}