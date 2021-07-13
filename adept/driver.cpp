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

#include "kernel.hpp"
#include "driver.hpp"
#include <chrono>
#include "instrument.hpp"


#define printVar(x)       std::cout << #x " = " << x << std::endl;

using namespace sycl;
using namespace ADEPT;

// warp size
static constexpr size_t warpSize = 32;

namespace DNA
{
/* Classes used to name the kernels for the runtime.
 * This must be done when the kernel is expressed as a lambda. */
class Adept_F;
class Adept_R;
}

namespace AA
{
/* Classes used to name the kernels for the runtime.
 * This must be done when the kernel is expressed as a lambda. */
class Adept_F;
class Adept_R;
}

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

    // use a specific device
    adept_stream(sycl::device *device)
    {
        stream = sycl::queue(*device);
    }

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



void 
ADEPT::aln_results::free_results()
{
    delete[] ref_begin;
    delete[] ref_end;
    delete[] query_begin;
    delete[] query_end;
    delete[] top_scores;
}

// ------------------------------------------------------------------------------------ //

double 
driver::initialize(short scores[], gap_scores g_scores, ALG_TYPE _algorithm, SEQ_TYPE _sequence, CIGAR _cigar_avail, int _max_ref_size, int _max_query_size, int _tot_alns, int _batch_size, sycl::device *dev)
{
    algorithm = _algorithm, sequence = _sequence, cigar_avail = _cigar_avail;

    device = dev;

    if (dev == nullptr)
    {
        sycl::gpu_selector selector;

        // using the GPU selector type here
        curr_stream = new adept_stream(&selector);
    }
    else
    {
        // using the device here
        curr_stream = new adept_stream(dev);
    }

    // set scores
    if(sequence == SEQ_TYPE::DNA)
    {
        match_score = scores[0];
        mismatch_score = scores[1]; 
    }
    else
    {
        scoring_matrix_cpu = scores;
        constexpr short temp_encode[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           23,0,0,0,0,0,0,0,0,0,0,0,0,0,
                           0,0,0,0,0,0,0,0,0,0,20,4,3,6,
                           13,7,8,9,0,11,10,12,2,0,14,5,
                           1,15,16,0,19,17,22,18,21};
        
        encoding_matrix = sycl::malloc_host<short>(ENCOD_MAT_SIZE, curr_stream->stream);

        for(int i = 0; i < ENCOD_MAT_SIZE; i++)
            encoding_matrix[i] = temp_encode[i];
    }

    // gap scores needed for both kernels
    gap_start = g_scores.open;
    gap_extend = g_scores.extend;

    // Print out the device information used for the kernel code.
    std::cout << "INFO: The device: "
              << curr_stream->stream.get_device().get_info<info::device::name>() << " is now ready!" << std::endl;

    total_alignments = _tot_alns;
    batch_size = _batch_size;
    max_ref_size = _max_ref_size;
    max_que_size = _max_query_size;

    // TODO: host pinned memory for offsets - memory pinning needs buffer/accessor model in SYCL
    offset_ref = sycl::malloc_host<int>(batch_size, curr_stream->stream);
    offset_que = sycl::malloc_host<int>(batch_size, curr_stream->stream);

    //host pinned memory for sequences - memory pinning needs buffer/accessor model in SYCL
    ref_cstr = sycl::malloc_host<char>(max_ref_size * batch_size, curr_stream->stream);
    que_cstr = sycl::malloc_host<char>(max_que_size * batch_size, curr_stream->stream);

    // host pinned memory for results - memory pinning needs buffer/accessor model in SYCL
    initialize_alignments();

    // measure memory allocation times
    MARK_START(init);

    //device memory for sequences
    ref_cstr_d = sycl::malloc_device<char>(max_ref_size * batch_size, curr_stream->stream);
    que_cstr_d = sycl::malloc_device<char>(max_que_size * batch_size, curr_stream->stream);

    //device memory for offsets and results
    allocate_gpu_mem();

    return ELAPSED_SECONDS_FROM(init);
}

// ------------------------------------------------------------------------------------ //

std::array<double, 4>
driver::kernel_launch(std::vector<std::string> &ref_seqs, std::vector<std::string> &query_seqs, int res_offset)
{
    if(ref_seqs.size() < batch_size)
        batch_size = ref_seqs.size();
    //    std::cerr << "INITIALIZATION ERROR: driver was initialized with wrong number of alignments\n";

    //preparing offsets
    int running_sum = 0;
    for(int i = 0; i < batch_size; i++)
    {
        running_sum += ref_seqs[i].size();
        offset_ref[i] = running_sum;
    }

    total_length_ref = offset_ref[batch_size - 1];
    running_sum = 0;

    for(int i = 0; i < query_seqs.size(); i++)
    {
        running_sum += query_seqs[i].size();
        offset_que[i] = running_sum; 
    }

    total_length_que = offset_que[batch_size - 1];

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

    //move data to GPU
    auto htd_time = mem_cpy_htd(offset_ref_gpu, offset_query_gpu, offset_ref, offset_que, ref_cstr, ref_cstr_d, que_cstr, que_cstr_d, total_length_ref,  total_length_que);

    int minSize = (max_que_size < max_ref_size) ? max_que_size : max_ref_size;

    int totShmem = 3 * (minSize + 1) * sizeof(short);
    int alignmentPad = 4 + (4 - totShmem % 4);
    size_t   ShmemBytes = totShmem + alignmentPad;

    // marker for forward kernel
    MARK_START(fwd_time);
    static thread_local double f_kernel_time = 0;


    // queue the forward kernel
    auto f_kernel = curr_stream->stream.submit([&](sycl::handler &h)
    {
        //
        // shared memory accessors
        //

        // dynamic shared memory bytes
        shmAccessor_t <char> dyn_shmem(sycl::range<1>(ShmemBytes), h);

        // local Totals
        shmAccessor_t <short> locTots(sycl::range(warpSize), h);

        // local Indices
        shmAccessor_t <short> locInds(sycl::range(warpSize), h);

        // local indices2
        shmAccessor_t <short> locInds2(sycl::range(warpSize), h);

        // sh_prev_E
        shmAccessor_t <short> sh_prev_E(sycl::range(warpSize), h);

        // sh_prev_H
        shmAccessor_t <short> sh_prev_H(sycl::range(warpSize), h);

        // sh_prev_prev_H
        shmAccessor_t <short> sh_prev_prev_H(sycl::range(warpSize), h);

        // local_spill_prev_E
        shmAccessor_t <short> local_spill_prev_E(sycl::range(1024), h);

        // local_spill_prev_H
        shmAccessor_t <short> local_spill_prev_H(sycl::range(1024), h);

        // local_spill_prev_prev_H
        shmAccessor_t <short> local_spill_prev_prev_H(sycl::range(1024), h);

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
        auto score_encode_matrix_gpu_loc = d_scoring_matrix;

        if (sequence == SEQ_TYPE::AA)
        {
            // sh_aa_encoding
            shmAccessor_t <short> sh_aa_encoding(sycl::range(ENCOD_MAT_SIZE), h);

            // sh_aa_scoring
            shmAccessor_t <short> sh_aa_scoring(sycl::range(SCORE_MAT_SIZE), h);

            //
            // Protein kernel forward
            //
            h.parallel_for<class AA::Adept_F>(sycl::nd_range<1>(batch_size * minSize, minSize), [=](sycl::nd_item<1> item)[[intel::reqd_sub_group_size(warpSize)]]
            {
                Akernel::aa_kernel(ref_cstr_d_loc, que_cstr_d_loc, offset_ref_gpu_loc, offset_query_gpu_loc, ref_start_gpu_loc, ref_end_gpu_loc, query_start_gpu_loc, query_end_gpu_loc, scores_gpu_loc, gap_start_loc, gap_extend_loc, score_encode_matrix_gpu_loc, 
                false, item,
                dyn_shmem.get_pointer(),
                sh_prev_E.get_pointer(), 
                sh_prev_H.get_pointer(), 
                sh_prev_prev_H.get_pointer(), 
                local_spill_prev_E.get_pointer(), 
                local_spill_prev_H.get_pointer(), 
                local_spill_prev_prev_H.get_pointer(),
                sh_aa_encoding.get_pointer(),
                sh_aa_scoring.get_pointer(),
                locTots.get_pointer(), 
                locInds.get_pointer(), 
                locInds2.get_pointer()
                );
            });
        }
        else
        {
            //
            // DNA kernel forward
            //
            h.parallel_for<class DNA::Adept_F>(sycl::nd_range<1>(batch_size * minSize, minSize), [=](sycl::nd_item<1> item)[[intel::reqd_sub_group_size(warpSize)]]
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
        }
    });

    // stream wait
    curr_stream->stream.wait_and_throw();

    // accumulate the forward kernel time
    f_kernel_time += ELAPSED_SECONDS_FROM(fwd_time);

    // copy memory
    auto dth_mid_time = mem_copies_dth_mid(ref_end_gpu, results.ref_end , query_end_gpu, results.query_end, res_offset);

    // compute new length
    int new_length = get_new_min_length(results.ref_end, results.query_end, batch_size);


    // marker for reverse kernel
    MARK_START(rev_time);
    static thread_local double r_kernel_time = 0;

    // queue the reverse kernel
    auto r_kernel = curr_stream->stream.submit([&](sycl::handler &h)
    {
        //
        // shared memory accessors
        //

        // dynamic shared memory bytes
        shmAccessor_t <char> dyn_shmem(sycl::range<1>(ShmemBytes), h);

        // local Totals
        shmAccessor_t <short> locTots(sycl::range(warpSize), h);

        // local Indices
        shmAccessor_t <short> locInds(sycl::range(warpSize), h);

        // local indices2
        shmAccessor_t <short> locInds2(sycl::range(warpSize), h);

        // sh_prev_E
        shmAccessor_t <short> sh_prev_E(sycl::range(warpSize), h);

        // sh_prev_H
        shmAccessor_t <short> sh_prev_H(sycl::range(warpSize), h);

        // sh_prev_prev_H
        shmAccessor_t <short> sh_prev_prev_H(sycl::range(warpSize), h);

        // local_spill_prev_E
        shmAccessor_t <short> local_spill_prev_E(sycl::range(1024), h);

        // local_spill_prev_H
        shmAccessor_t <short> local_spill_prev_H(sycl::range(1024), h);

        // local_spill_prev_prev_H
        shmAccessor_t <short> local_spill_prev_prev_H(sycl::range(1024), h);

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
        auto score_encode_matrix_gpu_loc = d_scoring_matrix;

        if (sequence == SEQ_TYPE::AA)
        {
            // sh_aa_encoding
            shmAccessor_t <short> sh_aa_encoding(sycl::range(ENCOD_MAT_SIZE), h);
    
            // sh_aa_scoring
            shmAccessor_t <short> sh_aa_scoring(sycl::range(SCORE_MAT_SIZE), h);

            //
            // Protein kernel reverse
            //
            h.parallel_for<class AA::Adept_R>(sycl::nd_range<1>(batch_size * new_length, new_length), [=](sycl::nd_item<1> item)[[intel::reqd_sub_group_size(warpSize)]]
            {
                Akernel::aa_kernel(ref_cstr_d_loc, que_cstr_d_loc, offset_ref_gpu_loc, offset_query_gpu_loc, ref_start_gpu_loc, ref_end_gpu_loc, query_start_gpu_loc, query_end_gpu_loc, scores_gpu_loc, gap_start_loc, gap_extend_loc, score_encode_matrix_gpu_loc, true,
                item,
                dyn_shmem.get_pointer(),
                sh_prev_E.get_pointer(), 
                sh_prev_H.get_pointer(), 
                sh_prev_prev_H.get_pointer(), 
                local_spill_prev_E.get_pointer(), 
                local_spill_prev_H.get_pointer(), 
                local_spill_prev_prev_H.get_pointer(),
                sh_aa_encoding.get_pointer(),
                sh_aa_scoring.get_pointer(),
                locTots.get_pointer(), 
                locInds.get_pointer(), 
                locInds2.get_pointer()
                );
            });
        }
        else
        {
            //
            // DNA kernel reverse
            //
            h.parallel_for<class DNA::Adept_R>(sycl::nd_range<1>(batch_size * new_length, new_length), [=](sycl::nd_item<1> item)[[intel::reqd_sub_group_size(warpSize)]]
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
        }
    });

    // stream wait
    curr_stream->stream.wait_and_throw();

    // accumulate the reverse kernel time
    r_kernel_time += ELAPSED_SECONDS_FROM(rev_time);

    // return cumulative times
    return std::array<double, 4>{f_kernel_time, r_kernel_time, htd_time, dth_mid_time};
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

double 
driver::mem_cpy_htd(int* offset_ref_gpu, int* offset_query_gpu, int* offsetA_h, int* offsetB_h, char* strA, char* strA_d, char* strB, char* strB_d, int totalLengthA, int totalLengthB)
{
    // marker for host 2 device data transfer
    MARK_START(htd);
    static thread_local double htd_time = 0;

    curr_stream->stream.memcpy(offset_ref_gpu, offsetA_h, batch_size * sizeof(int));
    curr_stream->stream.memcpy(offset_query_gpu, offsetB_h, batch_size * sizeof(int));

    curr_stream->stream.memcpy(strA_d, strA, totalLengthA * sizeof(char));
    curr_stream->stream.memcpy(strB_d, strB, totalLengthB * sizeof(char));

    if(sequence == SEQ_TYPE::AA)
    {
        curr_stream->stream.memcpy(d_encoding_matrix, encoding_matrix, ENCOD_MAT_SIZE * sizeof(short));
        curr_stream->stream.memcpy(d_scoring_matrix, scoring_matrix_cpu, SCORE_MAT_SIZE * sizeof(short));
    }

    curr_stream->stream.wait_and_throw();

    // accumulate the time
    htd_time += ELAPSED_SECONDS_FROM(htd);

    return htd_time;
}

// ------------------------------------------------------------------------------------ //

double 
driver::mem_copies_dth(short* ref_start_gpu, short* alAbeg, short* query_start_gpu, short* alBbeg, short* scores_gpu, short* top_scores_cpu, int res_offset)
{
    // marker for device 2 host data transfer
    MARK_START(dth);
    static thread_local double dth_time = 0;
    
    curr_stream->stream.memcpy(alAbeg + res_offset, ref_start_gpu, batch_size * sizeof(short));
    curr_stream->stream.memcpy(alBbeg + res_offset, query_start_gpu, batch_size * sizeof(short));
    curr_stream->stream.memcpy(top_scores_cpu + res_offset, scores_gpu, batch_size * sizeof(short));

    curr_stream->stream.wait_and_throw();

    // accumulate the time
    dth_time += ELAPSED_SECONDS_FROM(dth);

    return dth_time;
}

// ------------------------------------------------------------------------------------ //

double 
driver::mem_copies_dth_mid(short* ref_end_gpu, short* alAend, short* query_end_gpu, short* alBend, int res_offset)
{
    // marker for device 2 host mid data transfer
    MARK_START(dth_mid);
    static thread_local double dth_mid_time = 0;

    curr_stream->stream.memcpy(alAend + res_offset, ref_end_gpu, batch_size * sizeof(short));
    curr_stream->stream.memcpy(alBend + res_offset, query_end_gpu, batch_size * sizeof(short));

    curr_stream->stream.wait_and_throw();

    // accumulate the time
    dth_mid_time += ELAPSED_SECONDS_FROM(dth_mid);

    return dth_mid_time;
}

// ------------------------------------------------------------------------------------ //

double 
driver::mem_cpy_dth(int offset)
{
    return mem_copies_dth(ref_start_gpu, results.ref_begin, query_start_gpu, results.query_begin, scores_gpu , results.top_scores, offset);
}

// ------------------------------------------------------------------------------------ //

void 
driver::initialize_alignments()
{
    results.ref_begin =   new short[total_alignments];
    results.ref_end =     new short[total_alignments]; 
    results.query_begin = new short[total_alignments]; 
    results.query_end =   new short[total_alignments]; 
    results.top_scores =  new short[total_alignments]; 
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

    if(sequence == SEQ_TYPE::AA)
    {
        sycl::free(d_scoring_matrix, curr_stream->stream);
    }
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

    if(sequence == SEQ_TYPE::AA)
        sycl::free(encoding_matrix, curr_stream->stream);

}

// ------------------------------------------------------------------------------------ //

void 
driver::allocate_gpu_mem()
{
    offset_query_gpu = sycl::malloc_device<int> (batch_size, curr_stream->stream);
    offset_ref_gpu =   sycl::malloc_device<int> (batch_size, curr_stream->stream);
    ref_start_gpu =    sycl::malloc_device<short> (batch_size, curr_stream->stream);
    ref_end_gpu =      sycl::malloc_device<short> (batch_size, curr_stream->stream);
    query_end_gpu =    sycl::malloc_device<short> (batch_size, curr_stream->stream);
    query_start_gpu =  sycl::malloc_device<short> (batch_size, curr_stream->stream);
    scores_gpu =       sycl::malloc_device<short> (batch_size, curr_stream->stream);

    if(sequence == SEQ_TYPE::AA)
    {
        d_scoring_matrix = sycl::malloc_device<short>(SCORE_MAT_SIZE + ENCOD_MAT_SIZE, curr_stream->stream);
        d_encoding_matrix = d_scoring_matrix + SCORE_MAT_SIZE;

    }
}

// ------------------------------------------------------------------------------------ //

size_t 
ADEPT::get_batch_size(const sycl::device &dev0, int max_q_size, int max_r_size, int per_gpu_mem)
{
    // get the global memory size
    auto globalMem = dev0.get_info<info::device::global_mem_size>();

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
    curr_stream->stream.wait_and_throw();
    return true;
}

// ------------------------------------------------------------------------------------ //

bool 
driver::dth_done()
{
    curr_stream->stream.wait_and_throw();
    return true;
}

// ------------------------------------------------------------------------------------ //

void driver::kernel_synch() { curr_stream->stream.wait_and_throw(); }

// ------------------------------------------------------------------------------------ //

void driver::dth_synch() { curr_stream->stream.wait_and_throw(); }

// ------------------------------------------------------------------------------------ //

void driver::set_gap_scores(short _gap_open, short _gap_extend)
{
    gap_start = _gap_open;
    gap_extend = _gap_extend;
}

// ------------------------------------------------------------------------------------ //

aln_results ADEPT::thread_launch(std::vector<std::string> &ref_vec, std::vector<std::string> &que_vec, ADEPT::ALG_TYPE algorithm, ADEPT::SEQ_TYPE sequence, ADEPT::CIGAR cigar_avail, int max_ref_size, int max_que_size, int batch_size, sycl::device *device, short scores[], int thread_id, gap_scores gaps)
{
    int alns_this_gpu = ref_vec.size();
    int iterations = (alns_this_gpu + (batch_size-1))/batch_size;

    if(iterations == 0) 
        iterations = 1;

    int left_over = alns_this_gpu % batch_size;
    int batch_last_it = batch_size;

    // minimum 20 or 5% iterations
    int iter_20 = std::max(5, iterations/20);

    if(left_over > 0)
        batch_last_it = left_over;

    std::cout << "Thread # " << thread_id << " " << "now owns device: " << device->get_info<info::device::name>() << std::endl;

    std::cout << "Local # Alignments = " << alns_this_gpu << std::endl << std::endl << std::flush;

    // initialize the adept driver
    driver sw_driver_loc;

    auto init_time = sw_driver_loc.initialize(scores, gaps, algorithm, sequence, cigar_avail, max_ref_size, max_que_size, alns_this_gpu, batch_size, device);

    PRINT_ELAPSED(init_time);
    std::cout << std::endl;

    for(int i = 0; i < iterations ; i++)
    {
        std::vector<std::string>::const_iterator start_, end_;
        start_ = ref_vec.begin() + i * batch_size;

        if(i == iterations -1)
            end_ = ref_vec.begin() + i * batch_size + batch_last_it;
        else
            end_ = ref_vec.begin() + (i + 1) * batch_size;

        std::vector<std::string> temp_ref(start_, end_);

        start_ = que_vec.begin() + i * batch_size;

        if(i == iterations - 1)
            end_ = que_vec.begin() + i * batch_size + batch_last_it;
        else
            end_ = que_vec.begin() + (i + 1) * batch_size;

        std::vector<std::string> temp_que(start_, end_);

        // print progress every 5%
        if (i % iter_20 == 0 || i == iterations - 1)
            std::cout << "Thread # " << thread_id << " progress = " << i + 1 << "/" << iterations << std::endl << std::flush;

        // launch kernel
        auto&& ktimes = sw_driver_loc.kernel_launch(temp_ref, temp_que, i * batch_size);

        // copy results d2h
        auto d2h_time = sw_driver_loc.mem_cpy_dth(i * batch_size);

        // synchronize
        sw_driver_loc.dth_synch();

#if defined (ADEPT_INSTR)
        // print progress every 5%
        if (i % iter_20 == 0 || i == iterations - 1)
        {
            std::cout << "Cumulative Fkernel time: " << ktimes[0] << "s" << std::endl;
            std::cout << "Cumulative Rkernel time: " << ktimes[1] << "s" << std::endl;
            std::cout << "Cumulative H2D time: " << ktimes[2] << "s" << std::endl;
            std::cout << "Cumulative D2Hmid time: " << ktimes[3] << "s" << std::endl;
            std::cout << "Cumulative D2H time: " << d2h_time << "s" << std::endl; 
            std::cout << std::endl << std::flush;
        }
#endif // ADEPT_INSTR
    }

    // get all alignment results
    auto loc_results = sw_driver_loc.get_alignments();// results for all iterations are available now

    // cleanup
    sw_driver_loc.cleanup();

    // return local results
    return loc_results;
 }

// ------------------------------------------------------------------------------------ //

all_alns ADEPT::multi_gpu(std::vector<std::string> &ref_sequences, std::vector<std::string> &que_sequences, ADEPT::ALG_TYPE algorithm, ADEPT::SEQ_TYPE sequence, ADEPT::CIGAR cigar_avail, int max_ref_size, int max_que_size, short scores[], gap_scores gaps, int batch_size_)
{
    // get all the gpu devices
    auto gpus = sycl::device::get_devices(info::device_type::gpu);
    int num_gpus = gpus.size();

    // see if at least one GPU found
    if (num_gpus < 1)
    {
        std::cerr << "ABORT: No GPU device found on this platform" << std::endl << std::flush;
        exit(-1);
    }

    int total_alignments = ref_sequences.size();

    // adjust batch size if not provided
     if(batch_size_ == -1)
        batch_size_ = ADEPT::get_batch_size(gpus[0], max_que_size, max_ref_size, 100);

    int batch_size = batch_size_;

    //total_alignments = alns_per_batch;

    std::vector<std::vector<std::string>> ref_batch_gpu;
    std::vector<std::vector<std::string>> que_batch_gpu;

    int alns_per_gpu = total_alignments / num_gpus;
    int left_over = total_alignments % num_gpus;

    std::cout << "Batch Size = " << batch_size << std::endl;
    std::cout << "Total Alignments = " << total_alignments << std::endl;
    std::cout << "Total Devices = " << num_gpus << std::endl;
    std::cout << "Alns per GPU = " << alns_per_gpu << std::endl << std::endl << std::flush;

    // divide the workload across GPUs
    for(int i = 0; i < num_gpus ; i++)
    {
        std::vector<std::string>::const_iterator start_, end_;
        start_ = ref_sequences.begin() + i * alns_per_gpu;
        if(i == num_gpus -1)
            end_ = ref_sequences.begin() + (i + 1) * alns_per_gpu + left_over;
        else
            end_ = ref_sequences.begin() + (i + 1) * alns_per_gpu;

        std::vector<std::string> temp_ref(start_, end_);

        start_ = que_sequences.begin() + i * alns_per_gpu;
        if(i == num_gpus - 1)
            end_ = que_sequences.begin() + (i + 1) * alns_per_gpu + left_over;
        else
            end_ = que_sequences.begin() + (i + 1) * alns_per_gpu;

        std::vector<std::string> temp_que(start_, end_);

        ref_batch_gpu.push_back(std::move(temp_ref));
        que_batch_gpu.push_back(std::move(temp_que));
    }

    // alignment results from all GPUs
    all_alns global_results(num_gpus);

    global_results.per_gpu = alns_per_gpu;
    global_results.left_over = left_over;
    global_results.gpus = num_gpus;

    // create a thread pool for multiple GPUs
    std::vector<std::thread> thread_pool;

    // wrapper lambda function to provide thread_ids
    auto thread_launcher = [&](int thread_id)
    {
        global_results.results[thread_id] = ADEPT::thread_launch(ref_batch_gpu[thread_id], que_batch_gpu[thread_id], algorithm, sequence, cigar_avail, max_ref_size, max_que_size, batch_size, &gpus[thread_id], scores, thread_id, gaps);
    };

    // launch a thread for each GPU
    for (auto tid = 0; tid < num_gpus; tid++)
        thread_pool.push_back(std::move(std::thread(thread_launcher, tid)));

    // wait for threads
    for (auto& thd : thread_pool)
        thd.join();

    // clear the pool
    thread_pool.clear();

    return global_results;
}

// ------------------------------------------------------------------------------------ //