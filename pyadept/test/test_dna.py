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
#

from __future__ import absolute_import

__author__ = "Muhammad Haseeb"
__copyright__ = "Copyright 2021, The Regents of the University of California"
__credits__ = ["Muhammad Haseeb"]
__license__ = "MIT"
__maintainer__ = "Muaaz Awan"
__email__ = "mgawan@lbl.gov"
__status__ = "Development"

import os
import sys
import time
import math
import unittest
import datetime
import numpy as np
import pandas as pd
import pyadept as adept
from pyadept import options as opt

# --------------------------- test setup variables ----------------------------------- #
MAX_REF_LEN    =      1200
MAX_QUERY_LEN  =       300
GPU_ID         =         0
DATA_SIZE      =  math.inf

MATCH          =  3
MISMATCH       = -3
GAP_OPEN       = -6
GAP_EXTEND     = -1

# FIXME: should these go to class config?

# --------------------------- helper functions ----------------------------------------- #

# parse FASTAs
def parseFASTAs(rfile, qfile):
    # empty lists for reference and query sequences
    rseqs = []
    qseqs = []

    # parse FASTA files together
    rfile = open(rfile)
    r = rfile.readlines()
    rfile.close()

    qfile = open(qfile)
    q = qfile.readlines()
    qfile.close()

    print('STATUS: Reading ref and query files', flush=True)

    for rline, qline in zip(r,q):
        if(rline[0] == '>'):
            if (qline[0] == '>'):
                continue
            else:
                    print("FATAL: Mismatch in lines")
                    exit(-2)
        else:
            if (len(rline) <= MAX_REF_LEN and len(qline) <= MAX_QUERY_LEN):
                # IMPORTANT: remove all whitespaces if present
                rseqs.append(rline.lstrip().rstrip())
                qseqs.append(qline.lstrip().rstrip())

        if (len(rseqs) == DATA_SIZE):
            break

    return rseqs, qseqs

# -------------------------- DNA Tests set ---------------------------------------- #
# DNA tests class
class PyAdeptDNATests(unittest.TestCase):
    # setup class: testing settings
    @classmethod
    def setUpClass(self):
        pass
        # set up DNA scoring 

    # runs at the start of each test
    def setUp(self):
        pass

    # runs at the end of each test
    def tearDown(self):
        pass
        # config.include_internal = False

    # Tear down class: finalize
    @classmethod
    def tearDownClass(self):
        pass

    # ---------------------------------------------------------------------------------- #
    # test simple DNA run 
    def test_simple(self):
        """simple"""
        self.assertTrue(0, 0)

    # ---------------------------------------------------------------------------------- #
    # test asynchronous DNA run
    def test_async(self):
        """async"""
        self.assertTrue(0, 0)

# ----------------------------- main test runner -------------------------------------- #
# main runner
def run():
    # run all tests
    unittest.main()

if __name__ == "__main__":
    run()
