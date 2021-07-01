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

# cd to adept directory
pushd $ADEPT

# make once
cmake .. -DADEPT_INSTR=OFF
make clean
make install -j 8

# test datasets
for i in $(seq 2 2); do
    printf "\nRunning dataset: $i out of 2\n\n"; 
    # ALIGNMENTS
    advisor --collect=roofline --profile-gpu --flop --trip-counts --project-dir=./roof_data_$i -- $ADEPT/examples/simple_sw/simple_sw /home/u75261/sw-benchmark/ref_set_$i.fasta /home/u75261/sw-benchmark/read_set_$i.fasta $ADEPT/align_roof$i.out 2>&1;
    
    advisor --report=roofline --with-stack --show-all-rows --display-callstack --top-down --gpu --memory-operation-type=all --data-type=int  --project-dir=./roof_data_$i --report-output=./roof_data_$i/roofline.html;

    advisor --report=summary --with-stack  --show-all-rows --display-callstack --top-down --gpu --memory-operation-type=all --data-type=int  --project-dir=./roof_data_$i --report-output=./roof_data_$i/summary;

    advisor --report=survey --with-stack  --show-all-rows --display-callstack --top-down --gpu --memory-operation-type=all --data-type=int  --project-dir=./roof_data_$i --report-output=./roof_data_$i/survey;

    advisor --report=top-down --with-stack  --show-all-rows --display-callstack --top-down --gpu --memory-operation-type=all --data-type=int  --project-dir=./roof_data_$i --report-output=./roof_data_$i/top-down;

    advisor --collect=hotspots --profile-gpu --flop --trip-counts --project-dir=./roof_data_$i -- $ADEPT/examples/simple_sw/simple_sw /home/u75261/sw-benchmark/ref_set_$i.fasta /home/u75261/sw-benchmark/read_set_$i.fasta $ADEPT/align_roof$i.out 2>&1;

    advisor --report=hotspots --with-stack  --show-all-rows --display-callstack --top-down --gpu --memory-operation-type=all --data-type=int  --project-dir=./roof_data_$i --report-output=./roof_data_$i/hotspots;

    advisor --collect=map --profile-gpu --flop --trip-counts --project-dir=./roof_data_$i -- $ADEPT/examples/simple_sw/simple_sw /home/u75261/sw-benchmark/ref_set_$i.fasta /home/u75261/sw-benchmark/read_set_$i.fasta $ADEPT/align_roof$i.out 2>&1;

    advisor --report=map --with-stack  --show-all-rows --display-callstack --top-down --gpu --memory-operation-type=all --data-type=int  --project-dir=./roof_data_$i --report-output=./roof_data_$i/map;

    advisor --collect=dependencies --profile-gpu --flop --trip-counts --project-dir=./roof_data_$i -- $ADEPT/examples/simple_sw/simple_sw /home/u75261/sw-benchmark/ref_set_$i.fasta /home/u75261/sw-benchmark/read_set_$i.fasta $ADEPT/align_roof$i.out 2>&1;

    advisor --report=map --with-stack  --show-all-rows --display-callstack --top-down --gpu --memory-operation-type=all --data-type=int  --project-dir=./roof_data_$i --report-output=./roof_data_$i/dependencies;

    advisor --report=joined --with-stack  --show-all-rows --display-callstack --top-down --gpu --memory-operation-type=all --data-type=int  --project-dir=./roof_data_$i --report-output=./roof_data_$i/dependencies;

done

# go back to the directory
popd
