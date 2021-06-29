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
#include <limits>
#include <bits/stdc++.h>
#include <functional>

// constants
constexpr int MAX_REF_LEN    =  1200;
constexpr int MAX_QUERY_LEN  =   300;
constexpr int BATCH_SIZE     = 50000;
constexpr int GPU_ID         =     0;

constexpr unsigned int DATA_SIZE = BATCH_SIZE; // std::numeric_limits<unsigned int>::max();

// scores
constexpr short MATCH          =  3;
constexpr short MISMATCH       = -3;
constexpr short GAP_OPEN       = -6;
constexpr short GAP_EXTEND     = -1;

using namespace std;

// ------------------------------------------------------------------------------------ //

//
// main function
//
int 
main(int argc, char* argv[])
{
    //
    // Print banner
    //
    std::cout <<                               std::endl;
    std::cout << "------------------------" << std::endl;
    std::cout << "       ADEPT SYCL       " << std::endl;
    std::cout << "------------------------" << std::endl;
    std::cout <<                               std::endl;

    //
    // argparser
    //

    // check command line arguments
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

    string lineR, lineQ;

    ifstream ref_file(refFile);
    ifstream quer_file(queFile);

    unsigned largestA = 0, largestB = 0;

    int totSizeA = 0, totSizeB = 0;

    // ------------------------------------------------------------------------------------ //

    //
    // File parser
    //

    // print status
    std::cout << "STATUS: Reading ref and query files" << std::endl;

    // extract reference sequences
    if(ref_file.is_open() && quer_file.is_open())
    {
        while(getline(ref_file, lineR))
        {
            getline(quer_file, lineQ);

            if(lineR[0] == '>')
            {
                if (lineR[0] == '>')
                    continue;
                else
                {
                    std::cout << "FATAL: Mismatch in lines" << std::endl;
                    exit(-2);
                }
            }
            else
            {
                if (lineR.length() <= MAX_REF_LEN && lineQ.length() <= MAX_QUERY_LEN)
                {
                    ref_sequences.push_back(lineR);
                    que_sequences.push_back(lineQ);

                    totSizeA += lineR.length();
                    totSizeB += lineQ.length();

                    if(lineR.length() > largestA)
                        largestA = lineR.length();

                    if(lineQ.length() > largestA)
                        largestB = lineQ.length();
                }
            }

            if (ref_sequences.size() == DATA_SIZE)
                break;
        }

        ref_file.close();
        quer_file.close();
    }

    if (ref_sequences.size() != que_sequences.size())
    {
        std::cerr << "FATAL: ref_sequences.size() != que_sequences.size()" << std::endl << std::flush;
        exit (-2);
    }

    // ------------------------------------------------------------------------------------ //

    //
    // run ADEPT on multiple GPUs
    //

    // print status
    std::cout << "STATUS: Launching driver" << std::endl << std::endl;

    // get batch size
    // auto gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    // size_t batch_size = ADEPT::get_batch_size(gpus[0], MAX_QUERY_LEN, MAX_REF_LEN, 100);

    std::array<short, 4> scores = { MATCH, MISMATCH, GAP_OPEN, GAP_EXTEND };

    // run on multi GPU
    auto all_results = ADEPT::multi_gpu(ref_sequences, que_sequences, ADEPT::ALG_TYPE::SW, ADEPT::SEQ_TYPE::DNA, ADEPT::CIGAR::YES, MAX_REF_LEN, MAX_QUERY_LEN, scores.data(), BATCH_SIZE);

    // ------------------------------------------------------------------------------------ //

    // print results from all GPUs

    // results
    ofstream results_file(outFile);
    int tot_gpus = all_results.gpus;

    std::cout << std::endl << "STATUS: Writing results..." << std::endl;

    // write the results header
    results_file << "alignment_scores\t"     << "reference_begin_location\t" << "reference_end_location\t" 
                 << "query_begin_location\t" << "query_end_location"         << endl;

    for(int gpus = 0; gpus < tot_gpus; gpus++)
    {
        int this_count = all_results.per_gpu;

        if(gpus == tot_gpus - 1)
            this_count += all_results.left_over;

        for(int k = 0; k < this_count; k++)
        {
            results_file<< all_results.results[gpus].top_scores[k] << "\t" << all_results.results[gpus].ref_begin[k] << "\t" << all_results.results[gpus].ref_end[k] - 1 << "\t"<< all_results.results[gpus].query_begin[k] << "\t" << all_results.results[gpus].query_end[k] - 1 << endl;
        }
    }

    // ------------------------------------------------------------------------------------ //

    //
    // Cleanup
    //

    // cleanup all_results
    for(int i = 0; i < tot_gpus; i++)
        all_results.results[i].free_results();
    
    // flush everything to stdout
    std::cout << "STATUS: Done" << std::endl << std::endl << std::flush;

    return 0;
}
