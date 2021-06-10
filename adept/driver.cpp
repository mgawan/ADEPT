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
#include "instrument.hpp"

using namespace sycl;
using namespace ADEPT;

// warp size
static constexpr size_t warpSize = 32;

// ------------------------------------------------------------------------------------ //

/* This is the class used to name the kernel for the runtime.
 * This must be done when the kernel is expressed as a lambda. */
class Adept_F;
class Adept_R;

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
ADEPT::aln_results::free_results()
{
    sycl::free(results.ref_begin, curr_stream->stream);
    sycl::free(results.ref_end, curr_stream->stream);
    sycl::free(results.query_begin, curr_stream->stream);
    sycl::free(results.query_end, curr_stream->stream);
    sycl::free(results.top_scores, curr_stream->stream);
}

// ------------------------------------------------------------------------------------ //

void 
driver::initialize(short scores[], ALG_TYPE _algorithm, SEQ_TYPE _sequence, CIGAR _cigar_avail, int _max_ref_size, int _max_query_size, int _tot_alns, int _batch_size, int _device_id)
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

	total_alignments = _tot_alns;
	batch_size = _batch_size;
    max_ref_size = _max_ref_size;
    max_que_size = _max_query_size;

    // TODO: host pinned memory for offsets - memory pinning needs buffer/accessor model in SYCL
    offset_ref = sycl::malloc_host<int>(total_alignments, curr_stream->stream);
    offset_que = sycl::malloc_host<int>(total_alignments, curr_stream->stream);


    //host pinned memory for sequences - memory pinning needs buffer/accessor model in SYCL
    ref_cstr = sycl::malloc_host<char>(max_ref_size * batch_size, curr_stream->stream);
    que_cstr = sycl::malloc_host<char>(max_que_size * batch_size, curr_stream->stream);

    // host pinned memory for results - memory pinning needs buffer/accessor model in SYCL
    initialize_alignments();

    //device memory for sequences
    ref_cstr_d = sycl::malloc_device<char>(max_ref_size * batch_size, curr_stream->stream);
    que_cstr_d = sycl::malloc_device<char>(max_que_size * batch_size, curr_stream->stream);

    //device memory for offsets and results
    allocate_gpu_mem();
}

// ------------------------------------------------------------------------------------ //

void 
driver::kernel_launch(std::vector<std::string> ref_seqs, std::vector<std::string> query_seqs, int res_offset)
{
	if(ref_seqs.size() < batch_size)
		batch_size = ref_seqs.size();
	//	std::cerr << "INITIALIZATION ERROR: driver was initialized with wrong number of alignments\n";

    //preparing offsets 
    int running_sum = 0;
    for(int i = 0; i < batch_size; i++)
    {
        running_sum +=ref_seqs[i].size();
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

    // marker for forward kernel
    MARK_START(fwd_time);

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

        //
        // DNA kernel forward
        //
        h.parallel_for<class Adept_F>(sycl::nd_range<1>(total_alignments * minSize, minSize), [=](sycl::nd_item<1> item)[[intel::reqd_sub_group_size(warpSize)]]
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

    // compute the time from initial marker
    auto f_kernel_time = ELAPSED_SECONDS_FROM(fwd_time);

    std::cout << "Forward Kernel: DONE" << std::endl;

    PRINT_ELAPSED(f_kernel_time);

    // copy memory
    mem_copies_dth_mid(ref_end_gpu, results.ref_end , query_end_gpu, results.query_end, res_offset);

    // stream wait
    curr_stream->stream.wait_and_throw();

    // new length?
    int new_length = get_new_min_length(results.ref_end, results.query_end, batch_size);

    // marker for reverse kernel
    MARK_START(rev_time);

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
        // DNA kernel reverse
        //
        h.parallel_for<class Adept_R>(sycl::nd_range<1>(total_alignments * new_length, new_length), [=](sycl::nd_item<1> item)[[intel::reqd_sub_group_size(warpSize)]]
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

    // compute the time from initial marker
    auto r_kernel_time = ELAPSED_SECONDS_FROM(rev_time);

    std::cout << "Reverse Kernel: DONE" << std::endl;

    PRINT_ELAPSED(r_kernel_time);

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
    curr_stream->stream.memcpy(offset_ref_gpu, offsetA_h, batch_size * sizeof(int));
    curr_stream->stream.memcpy(offset_query_gpu, offsetB_h, batch_size * sizeof(int));

    curr_stream->stream.memcpy(strA_d, strA, totalLengthA * sizeof(char));
    curr_stream->stream.memcpy(strB_d, strB, totalLengthB * sizeof(char));
}

// ------------------------------------------------------------------------------------ //

void 
driver::mem_copies_dth(short* ref_start_gpu, short* alAbeg, short* query_start_gpu, short* alBbeg, short* scores_gpu, short* top_scores_cpu, int res_offset)
{
    curr_stream->stream.memcpy(alAbeg + res_offset, ref_start_gpu, batch_size * sizeof(short));
    curr_stream->stream.memcpy(alBbeg + res_offset, query_start_gpu, batch_size * sizeof(short));
    curr_stream->stream.memcpy(top_scores_cpu + res_offset, scores_gpu, batch_size * sizeof(short));
}

// ------------------------------------------------------------------------------------ //

void 
driver::mem_copies_dth_mid(short* ref_end_gpu, short* alAend, short* query_end_gpu, short* alBend, int res_offset)
{
    curr_stream->stream.memcpy(alAend + res_offset, ref_end_gpu, batch_size * sizeof(short));
    curr_stream->stream.memcpy(alBend + res_offset, query_end_gpu, batch_size * sizeof(short));
}

// ------------------------------------------------------------------------------------ //

void 
driver::mem_cpy_dth(int offset)
{
    mem_copies_dth(ref_start_gpu, results.ref_begin, query_start_gpu, results.query_begin, scores_gpu , results.top_scores, offset);
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
driver::allocate_gpu_mem()
{
    offset_query_gpu = sycl::malloc_device<int> (batch_size, curr_stream->stream);
    offset_ref_gpu =   sycl::malloc_device<int> (batch_size, curr_stream->stream);
    ref_start_gpu =    sycl::malloc_device<short> (batch_size, curr_stream->stream);
    ref_end_gpu =      sycl::malloc_device<short> (batch_size, curr_stream->stream);
    query_end_gpu =    sycl::malloc_device<short> (batch_size, curr_stream->stream);
    query_start_gpu =  sycl::malloc_device<short> (batch_size, curr_stream->stream);
    scores_gpu =       sycl::malloc_device<short> (batch_size, curr_stream->stream);
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

// ------------------------------------------------------------------------------------ //

aln_results ADEPT::thread_launch(std::vector<std::string> ref_vec, std::vector<std::string> que_vec, ADEPT::ALG_TYPE algorithm, ADEPT::SEQ_TYPE sequence, ADEPT::CIGAR cigar_avail, int max_ref_size, int max_que_size, int batch_size, int dev_id, short scores[]){
	int alns_this_gpu = ref_vec.size();
	int iterations = (alns_this_gpu + (batch_size-1))/batch_size;
	if(iterations == 0) iterations = 1;
	int left_over = alns_this_gpu%batch_size;
	int batch_last_it = batch_size;
	if(left_over > 0)	batch_last_it = left_over;

	std::vector<std::vector<std::string>> its_ref_vecs;
	std::vector<std::vector<std::string>> its_que_vecs;
	int my_cpu_id = omp_get_thread_num();

	std::cout <<"total alignments:"<<alns_this_gpu<<" thread:"<<my_cpu_id<<"\n";
	for(int i = 0; i < iterations ; i++){
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

		its_ref_vecs.push_back(temp_ref);
		its_que_vecs.push_back(temp_que);
	}

	driver sw_driver_loc;
	sw_driver_loc.initialize(scores, algorithm, sequence, cigar_avail, max_ref_size, max_que_size, alns_this_gpu, batch_size, dev_id);
	for(int i = 0; i < iterations; i++){
		sw_driver_loc.kernel_launch(its_ref_vecs[i], its_que_vecs[i], i * batch_size);
		sw_driver_loc.mem_cpy_dth(i * batch_size);
		sw_driver_loc.dth_synch();
	}

	auto loc_results = sw_driver_loc.get_alignments();// results for all iterations are available now
	sw_driver_loc.cleanup();
	return loc_results;
 }

all_alns ADEPT::multi_gpu(std::vector<std::string> ref_sequences, std::vector<std::string> que_sequences, ADEPT::ALG_TYPE algorithm, ADEPT::SEQ_TYPE sequence, ADEPT::CIGAR cigar_avail, int max_ref_size, int max_que_size, short scores[], int batch_size_){
	if(batch_size_ == -1)
		batch_size_ = ADEPT::get_batch_size(0, max_que_size, max_ref_size, 100);
	int total_alignments = ref_sequences.size();
  	int num_gpus;
	cudaGetDeviceCount(&num_gpus);
	unsigned batch_size = batch_size_;
	
	//total_alignments = alns_per_batch;
	std::cout << "Batch Size:"<< batch_size<<"\n";
	std::cout << "Total Alignments:"<< total_alignments<<"\n";
    std::cout << "Total devices:"<< num_gpus<<"\n";

	std::vector<std::vector<std::string>> ref_batch_gpu;
	std::vector<std::vector<std::string>> que_batch_gpu;
	int alns_per_gpu = total_alignments/num_gpus;
	int left_over = total_alignments%num_gpus;

	std::cout<< "Alns per GPU:"<<alns_per_gpu<<"\n";
   // std::array<short, 4> scores = {3,-3,-6,-1};

	for(int i = 0; i < num_gpus ; i++){
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

		ref_batch_gpu.push_back(temp_ref);
		que_batch_gpu.push_back(temp_que);
	}
  omp_set_num_threads(num_gpus);
  all_alns global_results(num_gpus);
  global_results.per_gpu = alns_per_gpu;
  global_results.left_over = left_over;
  global_results.gpus = num_gpus;

  #pragma omp parallel
  {
    int my_cpu_id = omp_get_thread_num();
	global_results.results[my_cpu_id] = ADEPT::thread_launch(ref_batch_gpu[my_cpu_id], que_batch_gpu[my_cpu_id], algorithm, sequence, cigar_avail, max_ref_size, max_que_size, batch_size, my_cpu_id, scores);
    #pragma omp barrier
  }

return global_results;
}
