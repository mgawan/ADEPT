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
    echo "USAGE: test_adept.sh <num_iterations> <adept_build>"
    echo "num_iterations: Number of iterations to run. default: 5"
    echo "adept_build: Path to ADEPT build directory. default: $PWD"
    echo ""
}

# set number of iterations and build path from the command line args
R=$1
ADEPT=$2

# if not provided, set to 5
if [ -z "$1" ]; then
    usage
    echo "INFO: setting number of iters = 5"
    R=5
    echo "INFO: setting ADEPT build path to $PWD"
    ADEPT=$PWD
fi

# if not provided, set to $PWD
if [ -z "$2" ]; then
    echo "INFO: setting ADEPT build path to $PWD"
    ADEPT=$PWD
fi

# cd to adept directory
pushd $ADEPT

# make once
make install -j 16

# testing loop
for i in $(seq 1 $R); do
    printf "\nRunning $i out of $R\n\n"; 
    ./adept_test ../test-data/dna-reference.fasta ../test-data/dna-query.fasta ../test-data/dna-output-$i.out ;
    DIFF=$(diff ../test-data/expected256.algn ../test-data/dna-output-$i.out);

    if [ "$DIFF" == "" ]; then
        printf "\nSUCCESS\n"
    else
        echo "FAILED. Check $PWD/../test-data/dna-output$i.diff"
        echo "$DIFF" >> ../test-data/dna-output$i.diff
    fi

    echo "Removing $i" ; rm ../test-data/dna-output-$i.out ;
done

# go back to the directory
popd