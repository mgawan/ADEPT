#!@PYTHON_EXECUTABLE@
# MIT License
#
# Copyright (c) 2020, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of any
# required approvals from the U.S. Dept. of Energy).  All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the software is
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

import os
import sys
import time
import math
import argparse
import datetime
import numpy as np
import pandas as pd
import pyadept as adept
from pyadept import options as opts

MAX_REF_LEN    =      1200
MAX_QUERY_LEN  =       600
GPU_ID         =         0
DATA_SIZE      =  math.inf

GAP_OPEN       = -6
GAP_EXTEND     = -1

# --------------------------------------------------------------------------------------------------- #

def banner():
    print("\n------------------------------")
    print("     PYTHON ASYNC PROTEIN     ")
    print("------------------------------\n", flush=True)

# --------------------------------------------------------------------------------------------------- #

def parseFASTAs(rfile, qfile):
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

# --------------------------------------------------------------------------------------------------- #

#
# Main function
#
def main():

    # initialize arg parser
    parser = argparse.ArgumentParser(description='ADEPT Asynchronous Amino Acid Alignment Example')

    # reference file
    parser.add_argument('-r', '--ref', dest='rfile', type=str, required=True,
                        help='Path to reference sequences file (FASTA format)')

    # query file
    parser.add_argument('-q', '--query', dest='qfile', type=str, required=True,
                        help='Path to query sequences file (FASTA format)')

    # output file
    parser.add_argument('-o', '--out', dest='ofile', type=str, required=False,
                        help='Path to output file (TSV format)')

    # test file
    parser.add_argument('-t', '--test', dest='tfile', type=str, required=False,
                        help='Path to the TSV file against which the output will be compared')

    # parse arguments
    args = parser.parse_args()

    # reference seqs
    rfile = args.rfile.lstrip().rstrip()
    rfile = os.path.expanduser(rfile)

    # query seqs
    qfile = args.qfile.lstrip().rstrip()
    qfile = os.path.expanduser(qfile)

    # output file
    if args.ofile is not None:
        ofile = args.ofile.lstrip().rstrip()
        ofile = os.path.expanduser(ofile)
    else:
        ofile = os.path.dirname(os.path.realpath(__file__)) + '/aa.alignments.' + datetime.datetime.now().strftime("%d.%m.%Y.%H.%M.%S") + '.tsv'

    # check if both files exist
    if not os.path.exists(rfile) or not os.path.exists(qfile):
        print (f'ERROR: {rfile} or {qfile} does not exist\n')
        sys.exit (-1)

    # testing file
    if args.tfile is not None:
        tfile = args.tfile.lstrip().rstrip()
        tfile = os.path.expanduser(tfile)

        if not os.path.exists(tfile):
            print(f'{tfile} does not exist. Skipping correctness check...', flush=True)
            tfile = None
    else:
        tfile = None

    # parse FASTAs
    rseqs, qseqs = parseFASTAs(rfile, qfile)

    # instantiate a driver object
    drv = adept.driver()

    # blosum 62 scoring matrix for AA kernels
    score_matrix = adept.ShortList([4 ,-1 ,-2 ,-2 ,0 ,-1 ,-1 ,0 ,-2 ,-1 ,-1 ,-1 ,-1 ,-2 ,-1 ,1 ,0 ,-3 ,-2 ,0 ,-2 ,-1 ,0 ,-4 , -1 ,5 ,0 ,-2 ,
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
                                            ])

    # gap scores
    gaps = adept.gap_scores(GAP_OPEN, GAP_EXTEND)

    # get max batch size
    batch_size = adept.get_batch_size(GPU_ID, MAX_QUERY_LEN, MAX_REF_LEN, 100)

    total_alignments = len(rseqs)

    # initialize the driver
    itime = drv.initialize(score_matrix, gaps, opts.ALG_TYPE.SW, opts.SEQ_TYPE.AA, opts.CIGAR.YES, MAX_REF_LEN, MAX_QUERY_LEN, total_alignments, int(batch_size), int(GPU_ID))

    # print status
    print("STATUS: Launching driver", flush=True)

    # add instrumentation
    start_time = time.time()

    # launch the kernel
    drv.kernel_launch(rseqs, qseqs)

    work_cpu = 0

    while not drv.kernel_done():
        work_cpu += 1

    drv.mem_cpy_dth()

    while not drv.dth_done():
        work_cpu += 1

    # sync d2h transfers
    drv.dth_synch()

    # get results
    results = drv.get_alignments()

    print('\nSTATUS: PyADEPT Alignments completed')
    print("--- Elapsed: %s seconds ---" % round((time.time() - start_time), 4), flush=True)

    # separate out arrays
    ts = results.top_scores()
    rb = results.ref_begin()
    re = results.ref_end()
    re -= 1
    qb = results.query_begin()
    qe = results.query_end()
    qe -= 1

    # clean up the ADEPT driver
    drv.cleanup()

    # create a dataframe from the output
    dfr = pd.DataFrame(zip(ts, rb, re, qb, qe), columns=['alignment_scores', 'reference_begin_location', 'reference_end_location', 'query_begin_location','query_end_location'], dtype=np.int16)

    print("\nSTATUS: Writing results...")

    # save output file
    dfr.to_csv(ofile, sep='\t', index=False)

    print(' total CPU work (counts) done while GPU was busy:', work_cpu)


    # correctness check
    if tfile is not None:
        df2 = pd.read_csv(tfile, sep='\t')

        # compare the dfs
        diff = pd.concat([dfr,df2]).drop_duplicates(keep=False)

        if not diff.empty:
            # print diff
            diff.to_csv(ofile + '.diff', sep='\t')
            print(f'\nSTATUS: Correctness test failed. See: {ofile}.diff')
        else:
            print('\nSTATUS: Correctness test passed')
    else:
        print('\nINFO: Correctness test skipped...')

    print('\nSTATUS: Done')


# --------------------------------------------------------------------------------------------------- #

# The main function
if __name__ == '__main__':
    banner()
    main()