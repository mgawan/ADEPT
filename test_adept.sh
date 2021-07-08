#!/bin/bash -le

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

# print usage
function usage() {
    echo "USAGE: test_adept.sh <adept_build>"
    echo "adept_build: Path to ADEPT build directory. default: $PWD"
    echo ""
}

# set build path from the command line args
ADEPT=$1

# if not provided, set to $PWD
if [ -z "$1" ]; then
    echo "INFO: setting ADEPT build path to $PWD"
    ADEPT=$PWD
fi

# test the output
function test_output() {
    # if output was produced (adept ran successfully?)
    if [ -f "$2" ]; then
        DIFF=$(diff $1 $2);
    else 
        echo "ERROR: Output file does not exist";
        exit -1 ;
    fi

    # check if any diff?
    if [ "$DIFF" == "" ]; then
        printf "\nSUCCESS\n\n";
        echo "Removing $2" ; 
        rm $2 ;
    else
        echo "$DIFF" >> ./$2.diff ;
        echo "FAILED. Check $PWD/$2.diff" ;
        exit -2 ;
    fi
}

# cd to adept directory
pushd $ADEPT

# enable instrumentation if disabled
cmake .. -DADEPT_INSTR=ON -DBUILD_TESTS=ON

# make once
make clean
make install -j 16

#
# DNA examples
#

# set REF and ALN
REF=../test-data/expected256.algn
ALN=../test-data/dna-output.out

printf "\nRunning 1 out of 5\n\n";

# run simple sw example
./examples/simple_sw/simple_sw ../test-data/dna-reference.fasta ../test-data/dna-query.fasta $ALN $REF;

# test output
test_output "$REF" "$ALN"

printf "\nRunning 2 out of 5\n\n";

# run asynch_sw example
./examples/asynch_sw/asynch_sw ../test-data/dna-reference.fasta ../test-data/dna-query.fasta $ALN $REF;

# test output
test_output "$REF" "$ALN"

printf "\nRunning 3 out of 5\n\n";

# run multi_gpu example
./examples/multi_gpu/multi_gpu ../test-data/dna-reference.fasta ../test-data/dna-query.fasta $ALN $REF;

# test output
test_output "$REF" "$ALN"


#
# Protein examples
#

# set REF and ALN
REF=../test-data/protein_expected256.algn
ALN=../test-data/protein-output.out

printf "\nRunning 4 out of 5\n\n";

# run simple asynch_protein example
./examples/asynch_protein/asynch_protein ../test-data/protein-reference.fasta ../test-data/protein-query.fasta $ALN $REF;

printf "\nRunning 5 out of 5\n\n";

# run simple asynch_protein example
./examples/multigpu_protein/multigpu_protein ../test-data/protein-reference.fasta ../test-data/protein-query.fasta $ALN $REF;

# test output
test_output "$REF" "$ALN"