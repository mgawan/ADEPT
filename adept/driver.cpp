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

#include <CL/sycl.hpp>
#include "kernel.hpp"
#include "driver.hpp"
#include <chrono>

using namespace sycl;
using namespace ADEPT;

// warp size
static constexpr size_t warpSize = 32;

// ------------------------------------------------------------------------------------ //

int 
getMaxLength (std::vector<std::string> v)
{
    int maxLength = 0;

    for(auto str : v)
    {
        if(maxLength < str.length())
            maxLength = str.length();
    }
    return maxLength;
}

// ------------------------------------------------------------------------------------ //

//
// struct ADEPT::stream
//
struct ADEPT::adept_stream
{
    // sycl queue
    sycl::queue stream;

    // use sycl default selector
    adept_stream()
    {
#if defined (DEBUG_ON_HOST)
        // select the host as device for debugging
        sycl::host_selector selector;
#else
        // automatically selects the best available device
        sycl::default_selector selector;

#endif // DEBUG_ON_HOST

        stream = sycl::queue(selector);
    }

    // use a specific device selector
    adept_stream(sycl::device_selector *selector)
    {
        stream = sycl::queue(*selector);
    }
};

// ------------------------------------------------------------------------------------ //

void 
driver::initialize(short scores[], ALG_TYPE _algorithm, SEQ_TYPE _sequence, CIGAR _cigar_avail, int _max_ref_size, int _max_query_size, int batch_size, int _device_id)
{
    algorithm = _algorithm, sequence = _sequence, cigar_avail = _cigar_avail;

    if(sequence == SEQ_TYPE::DNA)
    {
        match_score = scores[0], mismatch_score = scores[1], gap_start = scores[2], gap_extend = scores[3];
    }

    device_id = _device_id;

    sycl::gpu_selector selector;

    // using the GPU selector type here
    curr_stream = new adept_stream(&selector);

        // Print out the device information used for the kernel code.
    std::cout << "Running on device: "
              << curr_stream->stream.get_device().get_info<info::device::name>() << "\n";

    total_alignments = batch_size;
    max_ref_size = _max_ref_size;
    max_que_size = _max_query_size;

    //host pinned memory for offsets - memory pinning not explicit in SYCL (impl def)
    offset_ref = sycl::malloc_host<int>(total_alignments, curr_stream->stream);
    offset_que = sycl::malloc_host<int>(total_alignments, curr_stream->stream);


    //host pinned memory for sequences - memory pinning not explicit in SYCL (impl def)
    ref_cstr = sycl::malloc_host<char>(max_ref_size * total_alignments, curr_stream->stream);
    que_cstr = sycl::malloc_host<char>(max_que_size * total_alignments, curr_stream->stream);

    // host pinned memory for results
    initialize_alignments();

    //device memory for sequences
    ref_cstr_d = sycl::malloc_device<char>(max_ref_size * total_alignments, curr_stream->stream);
    que_cstr_d = sycl::malloc_device<char>(max_que_size * total_alignments, curr_stream->stream);

    //device memory for offsets and results
    allocate_gpu_mem();
}

// ------------------------------------------------------------------------------------ //

void 
driver::kernel_launch(std::vector<std::string> ref_seqs, std::vector<std::string> query_seqs)
{
    if(ref_seqs.size() != total_alignments)
        std::cerr << "INIT ERROR: driver was initialized with a batch size that does not match to the vector passed to kernel\n";

    //preparing offsets 
    int running_sum = 0;
    for(int i = 0; i < total_alignments; i++)
    {
        running_sum +=ref_seqs[i].size();
        offset_ref[i] = running_sum;
    }

    total_length_ref = offset_ref[total_alignments - 1];
    running_sum = 0;

    for(int i = 0; i < query_seqs.size(); i++)
    {
        running_sum += query_seqs[i].size();
        offset_que[i] = running_sum; 
    }

    total_length_que = offset_que[total_alignments - 1];

    //moving sequences from vector to cstrings
    int offsetSumA = 0;
    int offsetSumB = 0;

    for(int i = 0; i < ref_seqs.size(); i++)
    {
        char* seqptrA = ref_cstr + offsetSumA;
        memcpy(seqptrA, ref_seqs[i].c_str(), ref_seqs[i].size());

        char* seqptrB = que_cstr + offsetSumB;
        memcpy(seqptrB, query_seqs[i].c_str(), query_seqs[i].size());

        offsetSumA += ref_seqs[i].size();
        offsetSumB += query_seqs[i].size();
    }

    //move data asynchronously to GPU
    mem_cpy_htd(offset_ref_gpu, offset_query_gpu, offset_ref, offset_que, ref_cstr, ref_cstr_d, que_cstr, que_cstr_d, total_length_ref,  total_length_que); // 

    int minSize = (max_que_size < max_ref_size) ? max_que_size : max_ref_size;

    std::cout << "Work-group size = " << minSize << std::endl;

    int totShmem = 3 * (minSize + 1) * sizeof(short);
    int alignmentPad = 4 + (4 - totShmem % 4);
    size_t   ShmemBytes = totShmem + alignmentPad;

    // CUDA specific shared memory setting
#if defined(__NVCC__) || defined(__CUDACC__)
    if(ShmemBytes > SHMEM_BYTES)
        cudaFuncSetAttribute(Akernel::dna_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, ShmemBytes);

#endif // __NVCC__ or __CUDACC__

    // stream wait for data copy
    curr_stream->stream.wait_and_throw();

    // queue the forward Smith Waterman kernel
    auto f_kernel = curr_stream->stream.submit([&](sycl::handler &h)
    {
        //
        // shared memory accessors
        //

        // dynamic shared memory bytes
        sycl::accessor<char, 1, sycl::access::mode::read_write, 
                                sycl::access::target::local> 
            dyn_shmem(sycl::range<1>(ShmemBytes), h);

        // local Totals
        sycl::accessor<short, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            locTots(sycl::range(warpSize), h);

        // local Indices
        sycl::accessor<short, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            locInds(sycl::range(warpSize), h);

        // local indices2
        sycl::accessor<short, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            locInds2(sycl::range(warpSize), h);

        // sh_prev_E
        sycl::accessor<short, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            sh_prev_E(sycl::range(warpSize), h);

        // sh_prev_H
        sycl::accessor<short, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            sh_prev_H(sycl::range(warpSize), h);

        // sh_prev_prev_H
        sycl::accessor<short, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            sh_prev_prev_H(sycl::range(warpSize), h);

        // local_spill_prev_E
        sycl::accessor<short, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            local_spill_prev_E(sycl::range(1024), h);

        // local_spill_prev_H
        sycl::accessor<short, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            local_spill_prev_H(sycl::range(1024), h);

        // local_spill_prev_prev_H
        sycl::accessor<short, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            local_spill_prev_prev_H(sycl::range(1024), h);

        //
        // local variables inside the lambda for access
        //
        auto ref_cstr_d_loc = ref_cstr_d;
        auto que_cstr_d_loc = que_cstr_d;
        auto offset_ref_gpu_loc = offset_ref_gpu;
        auto offset_query_gpu_loc = offset_query_gpu;
        auto ref_start_gpu_loc = ref_start_gpu;
        auto ref_end_gpu_loc = ref_end_gpu;
        auto query_start_gpu_loc = query_start_gpu;
        auto query_end_gpu_loc = query_end_gpu;
        auto scores_gpu_loc = scores_gpu;
        auto match_score_loc = match_score;
        auto mismatch_score_loc = mismatch_score;
        auto gap_start_loc = gap_start;
        auto gap_extend_loc = gap_extend;

        // TODO: Check the nd_range
        h.parallel_for(sycl::nd_range<1>(total_alignments * minSize, minSize), [=](sycl::nd_item<1> item)[[intel::reqd_sub_group_size(warpSize)]]
        {
            Akernel::dna_kernel(ref_cstr_d_loc, que_cstr_d_loc, offset_ref_gpu_loc, offset_query_gpu_loc, ref_start_gpu_loc, ref_end_gpu_loc, query_start_gpu_loc, query_end_gpu_loc, scores_gpu_loc, match_score_loc, mismatch_score_loc, gap_start_loc, gap_extend_loc, false, 
            item, 
            dyn_shmem.get_pointer(), 
            sh_prev_E.get_pointer(),
            sh_prev_H.get_pointer(),
            sh_prev_prev_H.get_pointer(),
            local_spill_prev_E.get_pointer(),
            local_spill_prev_H.get_pointer(),
            local_spill_prev_prev_H.get_pointer(),
            locTots.get_pointer(),
            locInds.get_pointer(),
            locInds2.get_pointer()
            );
        });
    });

    // stream wait
    curr_stream->stream.wait_and_throw();

    std::cout << "Forward Kernel: DONE" << std::endl;

    // copy memory
    mem_copies_dth_mid(ref_end_gpu, results.ref_end , query_end_gpu, results.query_end);

    // stream wait
    curr_stream->stream.wait_and_throw();

    // new length?
    int new_length = get_new_min_length(results.ref_end, results.query_end, total_alignments);

    // queue the reverse Smith Waterman kernel
    auto r_kernel = curr_stream->stream.submit([&](sycl::handler &h)
    {
        //
        // shared memory accessors
        //

        // dynamic shared memory bytes
        sycl::accessor<char, 1, sycl::access::mode::read_write, 
                                sycl::access::target::local> 
            dyn_shmem(sycl::range<1>(ShmemBytes), h);

        // local Totals
        sycl::accessor<short, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            locTots(sycl::range(warpSize), h);

        // local Indices
        sycl::accessor<short, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            locInds(sycl::range(warpSize), h);

        // local indices2
        sycl::accessor<short, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            locInds2(sycl::range(warpSize), h);

        // sh_prev_E
        sycl::accessor<short, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            sh_prev_E(sycl::range(warpSize), h);

        // sh_prev_H
        sycl::accessor<short, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            sh_prev_H(sycl::range(warpSize), h);

        // sh_prev_prev_H
        sycl::accessor<short, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            sh_prev_prev_H(sycl::range(warpSize), h);

        // local_spill_prev_E
        sycl::accessor<short, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            local_spill_prev_E(sycl::range(1024), h);

        // local_spill_prev_H
        sycl::accessor<short, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            local_spill_prev_H(sycl::range(1024), h);

        // local_spill_prev_prev_H
        sycl::accessor<short, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            local_spill_prev_prev_H(sycl::range(1024), h);

        //
        // local variables inside the lambda for access
        //
        auto ref_cstr_d_loc = ref_cstr_d;
        auto que_cstr_d_loc = que_cstr_d;
        auto offset_ref_gpu_loc = offset_ref_gpu;
        auto offset_query_gpu_loc = offset_query_gpu;
        auto ref_start_gpu_loc = ref_start_gpu;
        auto ref_end_gpu_loc = ref_end_gpu;
        auto query_start_gpu_loc = query_start_gpu;
        auto query_end_gpu_loc = query_end_gpu;
        auto scores_gpu_loc = scores_gpu;
        auto match_score_loc = match_score;
        auto mismatch_score_loc = mismatch_score;
        auto gap_start_loc = gap_start;
        auto gap_extend_loc = gap_extend;
    
        //
        // DNA kernel
        //
        h.parallel_for(sycl::nd_range<1>(total_alignments * new_length, new_length), [=](sycl::nd_item<1> item)[[intel::reqd_sub_group_size(warpSize)]]
        {
            Akernel::dna_kernel(ref_cstr_d_loc, que_cstr_d_loc, offset_ref_gpu_loc, offset_query_gpu_loc, ref_start_gpu_loc, ref_end_gpu_loc, query_start_gpu_loc, query_end_gpu_loc, scores_gpu_loc, match_score_loc, mismatch_score_loc, gap_start_loc, gap_extend_loc, true, 
            item,
            dyn_shmem.get_pointer(),
            sh_prev_E.get_pointer(), 
            sh_prev_H.get_pointer(), 
            sh_prev_prev_H.get_pointer(), 
            local_spill_prev_E.get_pointer(), 
            local_spill_prev_H.get_pointer(), 
            local_spill_prev_prev_H.get_pointer(),
            locTots.get_pointer(), 
            locInds.get_pointer(), 
            locInds2.get_pointer()
            );
        });
    });

    // stream wait
    curr_stream->stream.wait_and_throw();

    std::cout << "Reverse Kernel: DONE" << std::endl;
}

// ------------------------------------------------------------------------------------ //

int 
driver::get_new_min_length(short* alAend, short* alBend, int blocksLaunched)
{
    int newMin = 1000;
    int maxA = 0;
    int maxB = 0;

    for(int i = 0; i < blocksLaunched; i++)
    {
        if(alBend[i] > maxB )
        {
            maxB = alBend[i];
        }

        if(alAend[i] > maxA)
        {
            maxA = alAend[i];
        }
    }

    newMin = (maxB > maxA)? maxA : maxB;

    return newMin;
}

// ------------------------------------------------------------------------------------ //

void 
driver::mem_cpy_htd(int* offset_ref_gpu, int* offset_query_gpu, int* offsetA_h, int* offsetB_h, char* strA, char* strA_d, char* strB, char* strB_d, int totalLengthA, int totalLengthB)
{
    curr_stream->stream.memcpy(offset_ref_gpu, offsetA_h, total_alignments * sizeof(int));
    curr_stream->stream.memcpy(offset_query_gpu, offsetB_h, total_alignments * sizeof(int));

    curr_stream->stream.memcpy(strA_d, strA, totalLengthA * sizeof(char));
    curr_stream->stream.memcpy(strB_d, strB, totalLengthB * sizeof(char));
}

// ------------------------------------------------------------------------------------ //

void 
driver::mem_copies_dth(short* ref_start_gpu, short* alAbeg, short* query_start_gpu, short* alBbeg, short* scores_gpu, short* top_scores_cpu)
{
    curr_stream->stream.memcpy(alAbeg, ref_start_gpu, total_alignments * sizeof(short));
    curr_stream->stream.memcpy(alBbeg, query_start_gpu, total_alignments * sizeof(short));
    curr_stream->stream.memcpy(top_scores_cpu, scores_gpu, total_alignments * sizeof(short));
}

// ------------------------------------------------------------------------------------ //

void 
driver::mem_copies_dth_mid(short* ref_end_gpu, short* alAend, short* query_end_gpu, short* alBend)
{
    curr_stream->stream.memcpy(alAend, ref_end_gpu, total_alignments * sizeof(short));
    curr_stream->stream.memcpy(alBend, query_end_gpu, total_alignments * sizeof(short));
}

// ------------------------------------------------------------------------------------ //

void 
driver::mem_cpy_dth()
{
    mem_copies_dth(ref_start_gpu, results.ref_begin, query_start_gpu, results.query_begin, scores_gpu , results.top_scores);
}

// ------------------------------------------------------------------------------------ //

void 
driver::initialize_alignments()
{
    results.ref_begin =   sycl::malloc_host<short> (total_alignments, curr_stream->stream);
    results.ref_end =     sycl::malloc_host<short> (total_alignments, curr_stream->stream);
    results.query_begin = sycl::malloc_host<short> (total_alignments, curr_stream->stream);
    results.query_end =   sycl::malloc_host<short> (total_alignments, curr_stream->stream);
    results.top_scores =  sycl::malloc_host<short> (total_alignments, curr_stream->stream);
}

// ------------------------------------------------------------------------------------ //

aln_results 
driver::get_alignments()
{
    return results;
}

// ------------------------------------------------------------------------------------ //

void 
driver::dealloc_gpu_mem()
{
    sycl::free(offset_ref_gpu, curr_stream->stream);
    sycl::free(offset_query_gpu, curr_stream->stream);
    sycl::free(ref_start_gpu, curr_stream->stream);
    sycl::free(ref_end_gpu, curr_stream->stream);
    sycl::free(query_start_gpu, curr_stream->stream);
    sycl::free(query_end_gpu, curr_stream->stream);
    sycl::free(ref_cstr_d, curr_stream->stream);
    sycl::free(que_cstr_d, curr_stream->stream);
}

// ------------------------------------------------------------------------------------ //

void 
driver::cleanup()
{
    sycl::free(offset_ref, curr_stream->stream);
    sycl::free(offset_que, curr_stream->stream);
    sycl::free(ref_cstr, curr_stream->stream);
    sycl::free(que_cstr, curr_stream->stream);

    dealloc_gpu_mem();

    curr_stream->stream.wait_and_throw();
}

// ------------------------------------------------------------------------------------ //

void 
driver::free_results()
{
    sycl::free(results.ref_begin, curr_stream->stream);
    sycl::free(results.ref_end, curr_stream->stream);
    sycl::free(results.query_begin, curr_stream->stream);
    sycl::free(results.query_end, curr_stream->stream);
    sycl::free(results.top_scores, curr_stream->stream);
}

// ------------------------------------------------------------------------------------ //

void 
driver::allocate_gpu_mem()
{
    offset_query_gpu = sycl::malloc_device<int> (total_alignments, curr_stream->stream);
    offset_ref_gpu =   sycl::malloc_device<int> (total_alignments, curr_stream->stream);
    ref_start_gpu =    sycl::malloc_device<short> (total_alignments, curr_stream->stream);
    ref_end_gpu =      sycl::malloc_device<short> (total_alignments, curr_stream->stream);
    query_end_gpu =    sycl::malloc_device<short> (total_alignments, curr_stream->stream);
    query_start_gpu =  sycl::malloc_device<short> (total_alignments, curr_stream->stream);
    scores_gpu =       sycl::malloc_device<short> (total_alignments, curr_stream->stream);
}

// ------------------------------------------------------------------------------------ //

size_t 
driver::get_batch_size(int device_id, int max_q_size, int max_r_size, int per_gpu_mem)
{
    // get the global memory size
    auto globalMem = curr_stream->stream.get_device().get_info<info::device::global_mem_size>();

    size_t gpu_mem_avail = (double)globalMem * (double)per_gpu_mem/100;
    size_t gpu_mem_per_align = max_q_size + max_r_size + 2 * sizeof(int) + 5 * sizeof(short);
    size_t max_concurr_aln = floor(((double)gpu_mem_avail)/gpu_mem_per_align);

    if (max_concurr_aln > 50000)
        return 50000;
    else
        return max_concurr_aln;
}

// ------------------------------------------------------------------------------------ //

bool 
driver::kernel_done()
{
    /*
    auto status = cudaEventQuery(curr_stream->kernel_event);
    if(status == cudaSuccess)
        return true;
    else
        return false; */

    curr_stream->stream.wait_and_throw();

    return true;
}

// ------------------------------------------------------------------------------------ //

bool 
driver::dth_done()
{
    /*
    auto status = cudaEventQuery(curr_stream->data_event);
    if(status == cudaSuccess)
        return true;
    else
        return false;
    */

    curr_stream->stream.wait_and_throw();

    return true;
}

// ------------------------------------------------------------------------------------ //

void driver::kernel_synch() { curr_stream->stream.wait_and_throw(); }

// ------------------------------------------------------------------------------------ //

void driver::dth_synch() { curr_stream->stream.wait_and_throw(); }
