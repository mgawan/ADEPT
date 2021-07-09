#!@BASH_EXECUTABLE@

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
    echo "USAGE: run_datasets.sh <adept_build>"
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

# cd to adept directory
pushd $ADEPT

# make once
make clean
make install -j 8

# if output was produced (adept ran successfully?)
if [ -f "$ADEPT/log_SYCL.out" ]; then
    echo "Removing old $ADEPT/log_SYCL.out";
    rm $ADEPT/log_SYCL.out;
fi

# test DNA datasets
for i in $(seq 1 3); do
    printf "\nRunning dna dataset: $i out of 3\n\n"; 

    # Alignments
    REF=/global/cscratch1/sd/mhaseeb/sw-benchmarks/dna-testing/align_ds$i.out
    ALN=./dna_align_ds$i.out; 

    ./examples/multi_gpu/multi_gpu /global/cscratch1/sd/mhaseeb/sw-benchmarks/dna-testing/ref_set_$i.fasta /global/cscratch1/sd/mhaseeb/sw-benchmarks/dna-testing/read_set_$i.fasta  $ALN >> $ADEPT/log_SYCL.out 2>&1;

    # if output was produced (adept ran successfully?)
    if [ -f "$ALN" ]; then
        DIFF=$(diff $REF $ALN);
    else 
        echo "FAILURE: ADEPT failed for dataset $i";
        break ;
    fi

    # check if any diff?
    if [ "$DIFF" == "" ]; then
        printf "\nSUCCESS\n\n";
        # echo "Removing $ALN" ; rm $ALN ;
    else
        echo "$DIFF" >> ./$ALN.diff ;
    fi

done

# test protein datasets
for i in $(seq 1 3); do
    printf "\nRunning protein dataset: $i out of 3\n\n"; 

    # Alignments
    REF=/global/cscratch1/sd/mhaseeb/sw-benchmarks/protein-testing/align_ds$i.out
    ALN=./protein_align_ds$i.out; 

    ./examples/multigpu_protein/multigpu_protein /global/cscratch1/sd/mhaseeb/sw-benchmarks/protein-testing/ref_set_$i.fasta /global/cscratch1/sd/mhaseeb/sw-benchmarks/protein-testing/que_set_$i.fasta  $ALN >> $ADEPT/log_SYCL.out 2>&1;

    # if output was produced (adept ran successfully?)
    if [ -f "$ALN" ]; then
        DIFF=$(diff $REF $ALN);
    else 
        echo "FAILURE: ADEPT failed for dataset $i";
        break ;
    fi

    # check if any diff?
    if [ "$DIFF" == "" ]; then
        printf "\nSUCCESS\n\n";
        # echo "Removing $ALN" ; rm $ALN ;
    else
        echo "$DIFF" >> ./$ALN.diff ;
    fi

done

# go back to the directory
popd
