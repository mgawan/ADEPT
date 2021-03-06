#!@PYTHON_EXECUTABLE@
# MIT License
#
# Copyright (c) 2021, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of any
# required approvals from the U.S. Dept. of Energy).  All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

''' @file test/__main__.py
Run all PyADEPT unittests
'''

from __future__ import absolute_import

__author__ = "Muhammad Haseeb"
__copyright__ = "Copyright 2021, The Regents of the University of California"
__credits__ = ["Muhammad Haseeb"]
__license__ = "MIT"
__maintainer__ = "Muaaz Awan"
__email__ = "mgawan@lbl.gov"
__status__ = "Development"

import os
import unittest
import pyadept as adept
from pyadept import options as opt

# discover and run all adept unittests in the current directory
def run_all_tests():
    # auto discover unittests from test_*.py files into the adept test suite
    adeptTestSuite = unittest.defaultTestLoader.discover(start_dir=os.path.dirname(os.path.abspath(__file__)), 
                                                       pattern='test*.py')

    # print the loaded tests
    print('============= Loaded Tests =============\n\n {}\n'.format(adeptTestSuite))

    # create a results object to store test results
    result = unittest.TestResult()

    # enable stdout buffer
    result.buffer = True

    # run all tests in adeptTestSuite, use result object to store results
    print ('\n============= Tests Stdout =============\n')
    # run the tests
    adeptTestSuite.run(result)

    # print the results
    print ('\n============= Results =============\n')
    print("{}\n".format(result))


# run all tests
if __name__ == "__main__":
    run_all_tests()