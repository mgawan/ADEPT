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

#pragma once

// include headers
#include <vector>
#include <string>
#include <thread>
#include <CL/sycl.hpp>

// set shared memory setting to 48KB
const int SHMEM_BYTES = 48000;

// ------------------------------------------------------------------------------------ //

//
// namespace ADEPT
//
namespace ADEPT
{
//
// Enums
//
enum ALG_TYPE{SW, NW};
enum CIGAR{NO, YES};
enum SEQ_TYPE{DNA, AA};

//
// struct aln_results
//
struct aln_results{
    short *ref_begin, *ref_end, *query_begin, *query_end, *top_scores;
    aln_results() = default;
    void free_results();
};

struct all_alns
{
    aln_results* results;
    int per_gpu;
    int left_over;
    int gpus;

    all_alns(int count)
    {
        results = new aln_results[count];
        per_gpu = 0;
        left_over = 0;
        gpus = 0;
    }
};

//
// struct adept_stream
//
struct adept_stream;

//
// class driver
//
class driver
{
private:
    short match_score, mismatch_score, gap_start, gap_extend;
    sycl::device *device;
    ALG_TYPE algorithm;
    SEQ_TYPE sequence;
    CIGAR cigar_avail;
    adept_stream *curr_stream;

    int max_ref_size, max_que_size;
    char *ref_cstr, *que_cstr;
    int total_alignments, batch_size;
    int *offset_ref, *offset_que;
    int total_length_ref, total_length_que;
    short *ref_start_gpu, *ref_end_gpu, *query_start_gpu, *query_end_gpu, *scores_gpu;
    int *offset_ref_gpu, *offset_query_gpu;
    char *ref_cstr_d, *que_cstr_d;

    aln_results results;

    void allocate_gpu_mem();
    void dealloc_gpu_mem();
    void initialize_alignments();
    
    void mem_cpy_htd(int* offset_ref_gpu, int* offset_query_gpu, int* offsetA_h, int* offsetB_h, char* strA, char* strA_d, char* strB, char* strB_d, int totalLengthA, int totalLengthB);
    
    void mem_copies_dth(short* ref_start_gpu, short* alAbeg, short* query_start_gpu,short* alBbeg, short* scores_gpu , short* top_scores_cpu, int res_offset = 0);
    void mem_copies_dth_mid(short* ref_end_gpu, short* alAend, short* query_end_gpu, short* alBend, int res_offset = 0);
    int get_new_min_length(short* alAend, short* alBend, int blocksLaunched);

public:
    // default constructor
    driver() = default;

    void initialize(short scores[], ALG_TYPE _algorithm, SEQ_TYPE _sequence, CIGAR _cigar_avail, int _max_ref_size, int _max_query_size, int _batch_size, int _tot_alns, sycl::device *dev); // each adept_dna object will have a unique cuda stream
    std::array<double, 2>  kernel_launch(std::vector<std::string> ref_seqs, std::vector<std::string> query_seqs, int res_offset = 0);
    void mem_cpy_dth(int offset = 0);
    aln_results get_alignments();
    bool kernel_done();
    bool dth_done();
    void kernel_synch();
    void dth_synch();
    void cleanup();
};


aln_results thread_launch(std::vector<std::string> ref_vec, std::vector<std::string> que_vec, ADEPT::ALG_TYPE algorithm, ADEPT::SEQ_TYPE sequence, ADEPT::CIGAR cigar_avail, int max_ref_size, int max_que_size, int batch_size, sycl::device *device, short scores[], int thread_id);
    
all_alns multi_gpu(std::vector<std::string> ref_sequences, std::vector<std::string> que_sequences, ADEPT::ALG_TYPE algorithm, ADEPT::SEQ_TYPE sequence, ADEPT::CIGAR cigar_avail, int max_ref_size, int max_que_size, short scores[], int batch_size_ = -1);

size_t get_batch_size(const sycl::device &device, int max_q_size, int max_r_size, int per_gpu_mem = 100);
}
