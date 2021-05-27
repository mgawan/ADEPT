// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "adept/driver.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <bits/stdc++.h>

// constants
const int MAX_REF_LEN    =  1200;
const int MAX_QUERY_LEN  =   300;
const int BATCH_SIZE     = 30000;
const int GPU_ID         =     0;

// scores
const short MATCH          = 3;
const short MISMATCH       = -3;
const short GAP_OPEN       = -6;
const short GAP_EXTEND     = -1;

using namespace std;

// ------------------------------------------------------------------------------------ //

//
// main function
//
int
main(int argc, char* argv[])
{
    // parse cmd line args
    if (argc < 4)
    {
        cout << "USAGE: adept_test <reference_file> <query_file> <output_file>" << endl;
        exit(-1);
    }

    // command line arguments
    string refFile = argv[1];
    string queFile = argv[2];
    string outFile = argv[3];

    vector<string> ref_sequences, que_sequences;

    string   myInLine;

    ifstream ref_file(refFile);
    ifstream quer_file(queFile);

    unsigned largestA = 0, largestB = 0;

    int totSizeA = 0, totSizeB = 0;

    // ------------------------------------------------------------------------------------ //

    //
    // argparser
    //

    // extract reference sequences
    if(ref_file.is_open())
    {
        while(getline(ref_file, myInLine))
        {
            if(myInLine[0] == '>')
            {
                continue;
            }
            else
            {
                string seq = myInLine;
                ref_sequences.push_back(seq);
                totSizeA += seq.size();

                if(seq.size() > largestA)
                {
                    largestA = seq.size();
                }
            }
        }

        ref_file.close();
    }

    // extract query sequences
    if(quer_file.is_open())
    {
        while(getline(quer_file, myInLine))
        {

            if(myInLine[0] == '>')
            {
                continue;
            }
            else
            {
                string seq = myInLine;
                que_sequences.push_back(seq);
                totSizeB += seq.size();

                if(seq.size() > largestB)
                {
                    largestB = seq.size();
                }
            }
        }

        quer_file.close();
    }

    // sanity checks - not in release mode. - can use try-catch here if needed
    assert(largestA > 0);
    assert(largestB > 0);
    assert(que_sequences.size() > 0);
    assert(ref_sequences.size() > 0);

    // ------------------------------------------------------------------------------------ //

    //
    // ADEPT driver
    //

    // instantiate ADEPT driver
    ADEPT::driver sw_driver;

    // init scores array
    std::array<short, 4> scores = {MATCH, MISMATCH, GAP_OPEN, GAP_EXTEND};

    // initialize ADEPT driver
    sw_driver.initialize(scores.data(), ADEPT::ALG_TYPE::SW, ADEPT::SEQ_TYPE::DNA, ADEPT::CIGAR::YES, 
                        MAX_REF_LEN, MAX_QUERY_LEN, BATCH_SIZE, GPU_ID);

    // launch kernel
    sw_driver.kernel_launch(ref_sequences, que_sequences);

    // copy memory back from device to host
    sw_driver.mem_cpy_dth();

    // synchronize
    sw_driver.dth_synch();

    // cleanup kernel
    sw_driver.cleanup();

    // get alignment results
    auto results = sw_driver.get_alignments();

    // ------------------------------------------------------------------------------------ //

    //
    // print the results
    //

    ofstream results_file(outFile);

    // write the results header
    results_file << "alignment_scores\t"     << "reference_begin_location\t" << "reference_end_location\t" 
                 << "query_begin_location\t" << "query_end_location"         << endl;

    for(int k = 0; k < ref_sequences.size(); k++)
    {
        results_file << results.top_scores[k] << "\t" << results.ref_begin[k] << "\t" << results.ref_end[k] - 1 <<
        "\t" <<results.query_begin[k] << "\t" << results.query_end[k] - 1 << endl;
    }

    sw_driver.free_results();

    results_file.flush();
    results_file.close();

    return 0;
}
