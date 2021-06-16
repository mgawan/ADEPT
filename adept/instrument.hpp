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

#pragma once

#include <chrono>

#if defined (ADEPT_INSTR)

using time_point_t = std::chrono::system_clock::time_point;

#define MARK_START(mark)                  static thread_local time_point_t mark = std::chrono::system_clock::now()
#define MARK_END(mark)                    static thread_local time_point_t mark = std::chrono::system_clock::now()

#define ELAPSED_SECONDS(mark1, mark2)     std::chrono::duration<double>(mark2 - mark1).count()

#define ELAPSED_SECONDS_FROM(mark)        std::chrono::duration<double>(std::chrono::system_clock::now() - mark).count()

//
// MACRO for printing elapsed time
//
#define PRINT_ELAPSED(es)                 std::cout << "Elapsed Time: " << es << "s" << std::endl

#else

#define MARK_START(mark)
#define MARK_END(mark)

#define ELAPSED_SECONDS(mark1, mark2)     (double) 0
#define ELAPSED_SECONDS_FROM(mark)        (double) 0
#define PRINT_ELAPSED(es)

#endif // ADEPT_INSTR