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

#include "driver.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <bits/stdc++.h>
#include <thread>
#include <functional>

constexpr int MAX_REF_LEN    =      1200;
constexpr int MAX_QUERY_LEN  =       600;
constexpr int GPU_ID         =         0;

constexpr unsigned int DATA_SIZE = std::numeric_limits<unsigned int>::max();

// scores
constexpr short MATCH          =  3;
constexpr short MISMATCH       = -3;
constexpr short GAP_OPEN       = -6;
constexpr short GAP_EXTEND     = -1;

using namespace std;

// ------------------------------------------------------------------------------------ //

//
// verify correctness
//
int verify_correctness(std::string file1, std::string file2);

// ------------------------------------------------------------------------------------ //

//
// main function
//
int 
main(int argc, char* argv[])
{
    std::cout <<                                 std::endl;
    std::cout << "--------------------------" << std::endl;
    std::cout << "     MULTIGPU PROTEIN     " << std::endl;
    std::cout << "--------------------------" << std::endl;
    std::cout <<                                 std::endl;

    // check command line arguments
    if (argc < 4)
    {
        cout << "USAGE: asynch_protein <reference_file> <query_file> <output_file> OPTIONAL: <expected_results_file>" << std::endl;
        exit(-1);
    }

    string refFile = argv[1];
    string queFile = argv[2];
    string out_file = argv[3];
    string expFile;

    if (argc == 5)
        expFile = argv[4];

    vector<string> ref_sequences, que_sequences;
    string   lineR, lineQ;
    ifstream ref_file(refFile);
    ifstream quer_file(queFile);

    unsigned largestA = 0, largestB = 0;

    int totSizeA = 0, totSizeB = 0;

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

    ADEPT::driver sw_driver;

    // blosum 62 score matrix
    std::array<short, 576> scores_matrix = {4 ,-1 ,-2 ,-2 ,0 ,-1 ,-1 ,0 ,-2 ,-1 ,-1 ,-1 ,-1 ,-2 ,-1 ,1 ,0 ,-3 ,-2 ,0 ,-2 ,-1 ,0 ,-4 , -1 ,5 ,0 ,-2 ,
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
                                            }; 

    // ------------------------------------------------------------------------------------ //

    //
    // run ADEPT on multiple GPUs
    //

    // get batch size
    auto gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    size_t batch_size = ADEPT::get_batch_size(GPU_ID, MAX_QUERY_LEN, MAX_REF_LEN, 100);

    ADEPT::gap_scores gaps(GAP_OPEN, GAP_EXTEND);

    std::cout << "STATUS: Launching driver" << std::endl << std::endl;

    // run on multi GPU
    auto all_results = ADEPT::multi_gpu(ref_sequences, que_sequences, ADEPT::ALG_TYPE::SW, ADEPT::SEQ_TYPE::AA, ADEPT::CIGAR::YES, MAX_REF_LEN, MAX_QUERY_LEN, scores_matrix.data(), gaps, batch_size);

    // ------------------------------------------------------------------------------------ //

    // print results from all GPUs

    // results
    ofstream results_file(out_file);
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
            results_file << all_results.results[gpus].top_scores[k] << "\t" << all_results.results[gpus].ref_begin[k] << "\t" << all_results.results[gpus].ref_end[k] - 1 << "\t"<< all_results.results[gpus].query_begin[k] << "\t" << all_results.results[gpus].query_end[k] - 1 << endl;
        }
    }

    results_file.flush();
    results_file.close();

    // cleanup all_results
    for(int i = 0; i < tot_gpus; i++)
        all_results.results[i].free_results();

    int status = 0;

    // if expected file is provided, then check for correctness, otherwise exit
    if (expFile != "")
    {
        std::cout << "\nSTATUS: Checking output against: " << expFile << std::endl << std::endl;
        status = verify_correctness(expFile, out_file);

        if (status)
            std::cout << "STATUS: Correctness test failed." << std::endl << std::endl;
        else
            std::cout << "STATUS: Correctness test passed." << std::endl << std::endl;
    }
    else
    {
        std::cout << "\nINFO: <expected_results_file> not provided. Skipping correctness check..." << std::endl << std::endl;
    }

    // flush everything to stdout
    std::cout << "STATUS: Done" << std::endl << std::endl << std::flush;

    return status;

}

//
// verify correctness
//
int verify_correctness(string file1, string file2)
{
    std::ifstream ref_file(file1);
    std::ifstream test_file(file2);

    string ref_line, test_line;

    int isSame = 0;

    // extract reference sequences
    if(ref_file.is_open() && test_file.is_open())
    {
        while(getline(ref_file, ref_line) && getline(test_file, test_line))
        {
            if(test_line != ref_line)
            {
                isSame = -1;
            }
        }

        if (getline(ref_file, ref_line) && test_line != "")
            isSame = -2;

        if (getline(test_file, test_line) && test_line != "")
            isSame = -3;

        ref_file.close();
        test_file.close();
    }
    else
    {
        std::cout << "ERROR: cannot open either " << file1 << " or " << file2 << std::endl;
        isSame = -4;
    }

    return isSame;
}
