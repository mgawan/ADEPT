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

''' @file test/test_multigpu.py
Test PyADEPT on multiple GPUs (if available) in batch mode
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
    rseqs = []
    qseqs = []

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

# -------------------------- multiGPU Tests set ---------------------------------------- #
# multiGPU tests class
class PyAdeptMultiGPUTests(unittest.TestCase):
    # setup class: testing settings
    @classmethod
    def setUpClass(self):
        # set up AA scoring 
        self.MAX_REF_LEN    = 1200
        self.GPU_ID         = 0

        # set up DNA scoring
        self.MATCH          =  3
        self.MISMATCH       = -3
        self.GAP_OPEN       = -6
        self.GAP_EXTEND     = -1

        self.gaps = adept.gap_scores(self.GAP_OPEN, self.GAP_EXTEND)

    # runs at the start of each test
    def setUp(self):
        self.algorithm = opts.ALG_TYPE.SW
        self.cigar     = opts.CIGAR.YES

    # runs at the end of each test
    def tearDown(self):
        pass
        # config.include_internal = False

    # Tear down class: finalize
    @classmethod
    def tearDownClass(self):
        pass

    # ---------------------------------------------------------------------------------- #
    # test multiGPU protein 
    def test_multigpu_aa(self):
        """Protein MultiGPU"""

        MAX_QUERY_LEN  = 600
        sequence  = opts.SEQ_TYPE.AA
        
        rseqs, qseqs = parseFASTAs(os.path.dirname(os.path.realpath(__file__)) + '/../../test-data/protein-reference.fasta', 
                                           os.path.dirname(os.path.realpath(__file__)) + '/../../test-data/protein-query.fasta',
                                           self.MAX_REF_LEN, MAX_QUERY_LEN)

        self.total_alignments = len(rseqs)

        dfref = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + '/../../test-data/protein_expected256.algn', sep='\t')

        # at least 2 batches to test
        batch_size = int(len(rseqs) / 2 + 1)


        # blosum 62 scoring matrix for AA kernels
        score_matrix = [4 ,-1 ,-2 ,-2 ,0 ,-1 ,-1 ,0 ,-2 ,-1 ,-1 ,-1 ,-1 ,-2 ,-1 ,1 ,0 ,-3 ,-2 ,0 ,-2 ,-1 ,0 ,-4 , -1 ,5 ,0 ,-2 ,
                    -3 ,1 ,0 ,-2 ,0 ,-3 ,-2 ,2 ,-1 ,-3 ,-2 ,-1 ,-1 ,-3 ,-2 ,-3 ,-1 ,0 ,-1 ,-4 ,
                    -2 ,0 ,6 ,1 ,-3 ,0 ,0 ,0 ,1 ,-3 ,-3 ,0 ,-2 ,-3 ,-2 ,1 ,0 ,-4 ,-2 ,-3 ,3 ,0 ,-1 ,-4 ,
                    -2 ,-2 ,1 ,6 ,-3 ,0 ,2 ,-1 ,-1 ,-3 ,-4 ,-1 ,-3 ,-3 ,-1 ,0 ,-1 ,-4 ,-3 ,-3 ,4 ,1 ,-1 ,-4 ,
                    0 ,-3 ,-3 ,-3 ,9 ,-3 ,-4 ,-3 ,-3 ,-1 ,-1 ,-3 ,-1 ,-2 ,-3 ,-1 ,-1 ,-2 ,-2 ,-1 ,-3 ,-3 ,-2 ,-4 ,
                    -1 ,1 ,0 ,0 ,-3 ,5 ,2 ,-2 ,0 ,-3 ,-2 ,1 ,0 ,-3 ,-1 ,0 ,-1 ,-2 ,-1 ,-2 ,0 ,3 ,-1 ,-4 ,
                    -1 ,0 ,0 ,2 ,-4 ,2 ,5 ,-2 ,0 ,-3 ,-3 ,1 ,-2 ,-3 ,-1 ,0 ,-1 ,-3 ,-2 ,-2 ,1 ,4 ,-1 ,-4 ,
                    0 ,-2 ,0 ,-1 ,-3 ,-2 ,-2 ,6 ,-2 ,-4 ,-4 ,-2 ,-3 ,-3 ,-2 ,0 ,-2 ,-2 ,-3 ,-3 ,-1 ,-2 ,-1 ,-4 ,
                    -2 ,0 ,1 ,-1 ,-3 ,0 ,0 ,-2 ,8 ,-3 ,-3 ,-1 ,-2 ,-1 ,-2 ,-1 ,-2 ,-2 ,2 ,-3 ,0 ,0 ,-1 ,-4 ,
                    -1 ,-3 ,-3 ,-3 ,-1 ,-3 ,-3 ,-4 ,-3 ,4 ,2 ,-3 ,1 ,0 ,-3 ,-2 ,-1 ,-3 ,-1 ,3 ,-3 ,-3 ,-1 ,-4 ,
                    -1 ,-2 ,-3 ,-4 ,-1 ,-2 ,-3 ,-4 ,-3 ,2 ,4 ,-2 ,2 ,0 ,-3 ,-2 ,-1 ,-2 ,-1 ,1 ,-4 ,-3 ,-1 ,-4 ,
                    -1 ,2 ,0 ,-1 ,-3 ,1 ,1 ,-2 ,-1 ,-3 ,-2 ,5 ,-1 ,-3 ,-1 ,0 ,-1 ,-3 ,-2 ,-2 ,0 ,1 ,-1 ,-4 ,
                    -1 ,-1 ,-2 ,-3 ,-1 ,0 ,-2 ,-3 ,-2 ,1 ,2 ,-1 ,5 ,0 ,-2 ,-1 ,-1 ,-1 ,-1 ,1 ,-3 ,-1 ,-1 ,-4 ,
                    -2 ,-3 ,-3 ,-3 ,-2 ,-3 ,-3 ,-3 ,-1 ,0 ,0 ,-3 ,0 ,6 ,-4 ,-2 ,-2 ,1 ,3 ,-1 ,-3 ,-3 ,-1 ,-4 ,
                    -1 ,-2 ,-2 ,-1 ,-3 ,-1 ,-1 ,-2 ,-2 ,-3 ,-3 ,-1 ,-2 ,-4 ,7 ,-1 ,-1 ,-4 ,-3 ,-2 ,-2 ,-1 ,-2 ,-4 ,
                    1 ,-1 ,1 ,0 ,-1 ,0 ,0 ,0 ,-1 ,-2 ,-2 ,0 ,-1 ,-2 ,-1 ,4 ,1 ,-3 ,-2 ,-2 ,0 ,0 ,0 ,-4 ,
                    0 ,-1 ,0 ,-1 ,-1 ,-1 ,-1 ,-2 ,-2 ,-1 ,-1 ,-1 ,-1 ,-2 ,-1 ,1 ,5 ,-2 ,-2 ,0 ,-1 ,-1 ,0 ,-4 ,
                    -3 ,-3 ,-4 ,-4 ,-2 ,-2 ,-3 ,-2 ,-2 ,-3 ,-2 ,-3 ,-1 ,1 ,-4 ,-3 ,-2 ,11 ,2 ,-3 ,-4 ,-3 ,-2 ,-4 ,
                    -2 ,-2 ,-2 ,-3 ,-2 ,-1 ,-2 ,-3 ,2 ,-1 ,-1 ,-2 ,-1 ,3 ,-3 ,-2 ,-2 ,2 ,7 ,-1 ,-3 ,-2 ,-1 ,-4 ,
                    0 ,-3 ,-3 ,-3 ,-1 ,-2 ,-2 ,-3 ,-3 ,3 ,1 ,-2 ,1 ,-1 ,-2 ,-2 ,0 ,-3 ,-1 ,4 ,-3 ,-2 ,-1 ,-4 ,
                    -2 ,-1 ,3 ,4 ,-3 ,0 ,1 ,-1 ,0 ,-3 ,-4 ,0 ,-3 ,-3 ,-2 ,0 ,-1 ,-4 ,-3 ,-3 ,4 ,1 ,-1 ,-4 ,
                    -1 ,0 ,0 ,1 ,-3 ,3 ,4 ,-2 ,0 ,-3 ,-3 ,1 ,-1 ,-3 ,-1 ,0 ,-1 ,-3 ,-2 ,-2 ,1 ,4 ,-1 ,-4 ,
                    0 ,-1 ,-1 ,-1 ,-2 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-2 ,0 ,0 ,-2 ,-1 ,-1 ,-1 ,-1 ,-1 ,-4 ,
                    -4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,1
                    ]

        # add instrumentation
        stime = time.time()

        # initialize the driver
        all_results = adept.multiGPU(rseqs, qseqs, self.algorithm, sequence, self.cigar, self.MAX_REF_LEN, MAX_QUERY_LEN, score_matrix, self.gaps, int(batch_size))

        # separate out arrays
        ts = []
        rb = []
        re = []
        qb = []
        qe = []

        for i in all_results.results:
            # separate out arrays
            ts.append(i.top_scores())
            rb.append(i.ref_begin())
            re.append(i.ref_end())
            qb.append(i.query_begin())
            qe.append(i.query_end())

        # transpose the arrays for column major indexing
        ts = np.array(ts).T 
        rb = np.array(rb).T
        re = np.array(re).T - 1
        qb = np.array(qb).T
        qe = np.array(qe).T - 1

        print('\nProtein MultiGPU completed')
        print("--- Elapsed: %s seconds ---" % round((time.time() - stime), 4), flush=True)

        # create a dataframe from the output
        df = pd.DataFrame(zip(ts, rb, re, qb, qe), columns=['alignment_scores', 'reference_begin_location', 'reference_end_location', 'query_begin_location','query_end_location'], dtype=np.int16)

        # compare the dfs
        diff = pd.concat([df, dfref]).drop_duplicates(keep=False)

        # same results expected
        self.assertTrue(diff.empty)

    # ---------------------------------------------------------------------------------- #
    # test asynchronous AA run
    def test_multigpu_dna(self):
        """DNA MultiGPU"""

        MAX_QUERY_LEN  = 300
        sequence  = opts.SEQ_TYPE.DNA
        
        rseqs, qseqs = parseFASTAs(os.path.dirname(os.path.realpath(__file__)) + '/../../test-data/dna-reference.fasta', 
                                           os.path.dirname(os.path.realpath(__file__)) + '/../../test-data/dna-query.fasta',
                                           self.MAX_REF_LEN, MAX_QUERY_LEN)

        self.total_alignments = len(rseqs)

        dfref = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + '/../../test-data/expected256.algn', sep='\t')

        # at least 2 batches to test
        batch_size = int(len(rseqs) / 2 + 1)

        # DNA scoring matrix
        score_matrix = [self.MATCH, self.MISMATCH]

        # add instrumentation
        stime = time.time()

        # initialize the driver
        all_results = adept.multiGPU(rseqs, qseqs, self.algorithm, sequence, self.cigar, self.MAX_REF_LEN, MAX_QUERY_LEN, score_matrix, self.gaps, int(batch_size))

        # separate out arrays
        ts = []
        rb = []
        re = []
        qb = []
        qe = []

        for i in all_results.results:
            # separate out arrays
            ts.append(i.top_scores())
            rb.append(i.ref_begin())
            re.append(i.ref_end())
            qb.append(i.query_begin())
            qe.append(i.query_end())

        # transpose the arrays for column major indexing
        ts = np.array(ts).T 
        rb = np.array(rb).T
        re = np.array(re).T - 1
        qb = np.array(qb).T
        qe = np.array(qe).T - 1

        print('\nDNA MultiGPU completed')
        print("--- Elapsed: %s seconds ---" % round((time.time() - stime), 4), flush=True)

        # create a dataframe from the output
        df = pd.DataFrame(zip(ts, rb, re, qb, qe), columns=['alignment_scores', 'reference_begin_location', 'reference_end_location', 'query_begin_location','query_end_location'], dtype=np.int16)

        # compare the dfs
        diff = pd.concat([df, dfref]).drop_duplicates(keep=False)

        # same results expected
        self.assertTrue(diff.empty)

# ----------------------------- main test runner -------------------------------------- #
# main runner
def run():
    # run all tests
    unittest.main()

if __name__ == "__main__":
    run()
