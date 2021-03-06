#!/usr/common/software/python/3.8-anaconda-2020.11/bin/python3.8
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
#

from __future__ import absolute_import

import sys

__author__ = "Muhammad Haseeb"
__copyright__ = "Copyright 2021, The Regents of the University of California"
__credits__ = "Muhammad Haseeb"
__license__ = "MIT"
__maintainer__ = "Muaaz Awan"
__email__ = "mgawan@lbl.gov"
__status__ = "Development"

"""
This submodule imports the PyADEPT driver options.

    from pyadept import options
    driver = pyadept.driver()
    algorithm = options.ALG_TYPE.<NW|SW>
    sequence = options.SEQ_TYEP.<DNA|AA>
    cigar = options.CIGAR.<YES|NO>
    driver.initialize(.., algorithm, sequence, cigar, ...)
"""

try:
    from ..adept.options import *
except Exception as e:
    print("{}".format(e))
