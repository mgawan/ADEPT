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

''' @file test/test_dna.py
Test PyADEPT on DNA sequences
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
import sys
import time
import math
import unittest
import datetime
import numpy as np
import pandas as pd
import pyadept as adept
from pyadept import options as opts


# --------------------------- helper functions ----------------------------------------- #

# parse FASTAs
def parseFASTAs(rfile, qfile, maxrlen, maxqlen):
    # empty lists for reference and query sequences
    rseqs = adept.StringList()
    qseqs = adept.StringList()

    # parse FASTA files together
    rfile = open(rfile)
    r = rfile.readlines()
    rfile.close()

    qfile = open(qfile)
    q = qfile.readlines()
    qfile.close()

    for rline, qline in zip(r,q):
        if(rline[0] == '>'):
            if (qline[0] == '>'):
                continue
            else:
                    print("FATAL: Mismatch in lines")
                    exit(-2)
        else:
            if (len(rline) <= maxrlen and len(qline) <= maxqlen):
                # IMPORTANT: remove all whitespaces if present
                rseqs.append(rline.lstrip().rstrip())
                qseqs.append(qline.lstrip().rstrip())

    return rseqs, qseqs

# -------------------------- DNA Tests set ---------------------------------------- #
# DNA tests class
class PyAdeptDNATests(unittest.TestCase):
    # setup class: testing settings
    @classmethod
    def setUpClass(self):
        self.MAX_REF_LEN    = 1200
        self.MAX_QUERY_LEN  = 300
        self.GPU_ID         = 0

        # set up DNA scoring 
        self.MATCH          =  3
        self.MISMATCH       = -3
        self.GAP_OPEN       = -6
        self.GAP_EXTEND     = -1

        self.rseqs, self.qseqs = parseFASTAs(os.path.dirname(os.path.realpath(__file__)) + '/../../test-data/dna-reference.fasta', 
                                           os.path.dirname(os.path.realpath(__file__)) + '/../../test-data/dna-query.fasta',
                                           self.MAX_REF_LEN, self.MAX_QUERY_LEN)

        self.dfref = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + '/../../test-data/expected256.algn', sep='\t')

    # runs at the start of each test
    def setUp(self):
        self.algorithm = opts.ALG_TYPE.SW
        self.sequence  = opts.SEQ_TYPE.DNA
        self.cigar     = opts.CIGAR.YES
        self.total_alignments = len(self.rseqs)

        self.batch_size = adept.get_batch_size(self.GPU_ID, self.MAX_QUERY_LEN, self.MAX_REF_LEN, 100)

        # instantiate a driver object
        self.drv = adept.driver()

    # runs at the end of each test
    def tearDown(self):
        self.drv.cleanup()

    # Tear down class: finalize
    @classmethod
    def tearDownClass(self):
        pass

    # ---------------------------------------------------------------------------------- #
    # test simple DNA run 
    def test_simple_dna(self):
        """DNA simple"""

        gaps = adept.gap_scores(self.GAP_OPEN, self.GAP_EXTEND)
        score_matrix = adept.ShortList([self.MATCH, self.MISMATCH])
        
        self.drv.initialize(score_matrix, gaps, self.algorithm, self.sequence, self.cigar, self.MAX_REF_LEN, self.MAX_QUERY_LEN, self.total_alignments, int(self.batch_size), int(self.GPU_ID))

        # add instrumentation
        stime = time.time()

        # launch the kernel
        self.drv.kernel_launch(self.rseqs, self.qseqs)

        # synchronize kernel
        self.drv.kernel_synch()

        # copy data from device
        self.drv.mem_cpy_dth()

        # sync d2h transfers
        self.drv.dth_synch()

        # get results
        results = self.drv.get_alignments()

        print('\nDNA simple completed')
        print("--- Elapsed: %s seconds ---" % round((time.time() - stime), 4), flush=True)

        # separate out arrays
        ts = results.top_scores()
        rb = results.ref_begin()
        re = results.ref_end()
        re -= 1
        qb = results.query_begin()
        qe = results.query_end()
        qe -= 1

        # create a dataframe from the output
        df = pd.DataFrame(zip(ts, rb, re, qb, qe), columns=['alignment_scores', 'reference_begin_location', 'reference_end_location', 'query_begin_location','query_end_location'], dtype=np.int16)

        # compare the dfs
        diff = pd.concat([df,self.dfref]).drop_duplicates(keep=False)

        # same results expected
        self.assertTrue(diff.empty)

    # ---------------------------------------------------------------------------------- #
    # test asynchronous DNA run
    def test_async_dna(self):
        """DNA async"""

        gaps = adept.gap_scores(self.GAP_OPEN, self.GAP_EXTEND)
        score_matrix = adept.ShortList([self.MATCH, self.MISMATCH])
        
        self.drv.initialize(score_matrix, gaps, self.algorithm, self.sequence, self.cigar, self.MAX_REF_LEN, self.MAX_QUERY_LEN, self.total_alignments, int(self.batch_size), int(self.GPU_ID))
        # add instrumentation
        stime = time.time()

        # launch the kernel
        self.drv.kernel_launch(self.rseqs, self.qseqs)

        work_cpu = 0

        while not self.drv.kernel_done():
            work_cpu += 1

        self.drv.mem_cpy_dth()

        while not self.drv.dth_done():
            work_cpu += 1

        # sync d2h transfers
        self.drv.dth_synch()

        # get results
        results = self.drv.get_alignments()

        print('\nDNA async completed')
        print("--- Elapsed: %s seconds ---" % round((time.time() - stime), 4), flush=True)

        # separate out arrays
        ts = results.top_scores()
        rb = results.ref_begin()
        re = results.ref_end()
        re -= 1
        qb = results.query_begin()
        qe = results.query_end()
        qe -= 1

        # create a dataframe from the output
        df = pd.DataFrame(zip(ts, rb, re, qb, qe), columns=['alignment_scores', 'reference_begin_location', 'reference_end_location', 'query_begin_location','query_end_location'], dtype=np.int16)

        # compare the dfs
        diff = pd.concat([df,self.dfref]).drop_duplicates(keep=False)

        # same results expected
        self.assertTrue(diff.empty)

        # check if any work done in async
        self.assertTrue(work_cpu > 0)

# ----------------------------- main test runner -------------------------------------- #
# main runner
def run():
    # run all tests
    unittest.main()

if __name__ == "__main__":
    run()
