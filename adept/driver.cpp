#include "hip/hip_runtime.h"
#include "kernel.hpp"
#include "driver.hpp"
#include <thread>

#define errCheck(ans)                                                                  \
{                                                                                    \
    gpuAssert((ans), __FILE__, __LINE__);                                            \
}
inline void
gpuAssert(hipError_t code, const char* file, int line, bool abort = true){
    if(code != hipSuccess){
        fprintf(stderr, "GPUassert: %s %s %d cpu:%d\n", hipGetErrorString(code), file, line);
        if(abort)
            exit(code);
    }
}

using namespace ADEPT;

unsigned getMaxLength (std::vector<std::string> v){
  unsigned maxLength = 0;
  for(auto str : v){
    if(maxLength < str.length()){
      maxLength = str.length();
    }
  }
  return maxLength;
}

struct ADEPT::adept_stream{
	hipStream_t stream;
	hipEvent_t kernel_event;
	hipEvent_t data_event;
	
	adept_stream(int gpu){
		errCheck(hipSetDevice(gpu));
		errCheck(hipStreamCreate(&stream));
		errCheck(hipEventCreateWithFlags(&kernel_event, hipEventBlockingSync));
		errCheck(hipEventCreateWithFlags(&data_event, hipEventBlockingSync));
	}
};

void ADEPT::aln_results::free_results(){
	errCheck(hipHostFree(ref_begin));
    errCheck(hipHostFree(ref_end));
    errCheck(hipHostFree(query_begin));
    errCheck(hipHostFree(query_end));
    errCheck(hipHostFree(top_scores));

}
void driver::initialize(short scores[], ALG_TYPE _algorithm, SEQ_TYPE _sequence, CIGAR _cigar_avail, int _max_ref_size, int _max_query_size, int _tot_alns, int _batch_size,  int _gpu_id){
	algorithm = _algorithm, sequence = _sequence, cigar_avail = _cigar_avail;
	if(sequence == SEQ_TYPE::DNA){
		match_score = scores[0], mismatch_score = scores[1], gap_start = scores[2], gap_extend = scores[3];
	}
	gpu_id = _gpu_id;
	hipSetDevice(gpu_id);
	curr_stream = new adept_stream(gpu_id);

	total_alignments = _tot_alns;
	batch_size = _batch_size;

	max_ref_size = _max_ref_size;
	max_que_size = _max_query_size;
	//host pinned memory for offsets
	errCheck(hipHostMalloc(&offset_ref, sizeof(int) * batch_size));
	errCheck(hipHostMalloc(&offset_que, sizeof(int) * batch_size));
	//host pinned memory for sequences
	errCheck(hipHostMalloc(&ref_cstr, sizeof(char) * max_ref_size * batch_size));
	errCheck(hipHostMalloc(&que_cstr, sizeof(char) * max_que_size * batch_size));
	// host pinned memory for results
	initialize_alignments();
	//device memory for sequences
	errCheck(hipMalloc(&ref_cstr_d, sizeof(char) * max_ref_size * batch_size));
	errCheck(hipMalloc(&que_cstr_d,  sizeof(char)* max_que_size * batch_size));
	//device memory for offsets and results
	allocate_gpu_mem();

}

void driver::kernel_launch(std::vector<std::string> ref_seqs, std::vector<std::string> query_seqs, int res_offset){
	if(ref_seqs.size() < batch_size)
		batch_size = ref_seqs.size();
	//	std::cerr << "INITIALIZATION ERROR: driver was initialized with wrong number of alignments\n";
	//preparing offsets 
	unsigned running_sum = 0;
	for(int i = 0; i < batch_size; i++){
		running_sum +=ref_seqs[i].size();
		offset_ref[i] = running_sum;
	}
	total_length_ref = offset_ref[batch_size - 1];

	running_sum = 0;
	for(int i = 0; i < query_seqs.size(); i++){
		running_sum +=query_seqs[i].size();
		offset_que[i] = running_sum; 
	}
	total_length_que = offset_que[batch_size - 1];

	//moving sequences from vector to cstrings
	unsigned offsetSumA = 0;
	unsigned offsetSumB = 0;

 	for(int i = 0; i < ref_seqs.size(); i++){
		char* seqptrA = ref_cstr + offsetSumA;  
		memcpy(seqptrA, ref_seqs[i].c_str(), ref_seqs[i].size());
		char* seqptrB = que_cstr + offsetSumB;
		memcpy(seqptrB, query_seqs[i].c_str(), query_seqs[i].size());
		offsetSumA += ref_seqs[i].size();
		offsetSumB += query_seqs[i].size();
    }
    //move data asynchronously to GPU
    mem_cpy_htd(offset_ref_gpu, offset_query_gpu, offset_ref, offset_que, ref_cstr, ref_cstr_d, que_cstr, que_cstr_d, total_length_ref,  total_length_que); // TODO: add streams

	unsigned minSize = (max_que_size < max_ref_size) ? max_que_size : max_ref_size;
	unsigned totShmem = 3 * (minSize + 1) * sizeof(short);
	unsigned alignmentPad = 4 + (4 - totShmem % 4);
	size_t   ShmemBytes = totShmem + alignmentPad;
	//if(ShmemBytes > 48000)
     //   hipFuncSetAttribute(kernel::dna_kernel, hipFuncAttributeMaxDynamicSharedMemorySize, ShmemBytes);
	hipLaunchKernelGGL(kernel::dna_kernel, dim3(batch_size), dim3(minSize), ShmemBytes, curr_stream->stream, ref_cstr_d, que_cstr_d, offset_ref_gpu, offset_query_gpu, ref_start_gpu, ref_end_gpu, query_start_gpu, query_end_gpu, scores_gpu, match_score, mismatch_score, gap_start, gap_extend, false);
	mem_copies_dth_mid(ref_end_gpu, results.ref_end , query_end_gpu, results.query_end, res_offset);
	hipStreamSynchronize(curr_stream->stream);
	int new_length = get_new_min_length(results.ref_end, results.query_end, batch_size);
	hipLaunchKernelGGL(kernel::dna_kernel, dim3(batch_size), dim3(new_length), ShmemBytes, curr_stream->stream, ref_cstr_d, que_cstr_d, offset_ref_gpu, offset_query_gpu, ref_start_gpu, ref_end_gpu, query_start_gpu, query_end_gpu, scores_gpu, match_score, mismatch_score, gap_start, gap_extend, true);
	errCheck(hipEventRecord(curr_stream->kernel_event, curr_stream->stream));
}

int driver::get_new_min_length(short* alAend, short* alBend, int blocksLaunched){
        int newMin = 1000;
        int maxA = 0;
        int maxB = 0;
        for(int i = 0; i < blocksLaunched; i++){
          if(alBend[i] > maxB ){
              maxB = alBend[i];
          }
          if(alAend[i] > maxA){
            maxA = alAend[i];
          }
        }
        newMin = (maxB > maxA)? maxA : maxB;
        return newMin;
}

void driver::mem_cpy_htd(unsigned* offset_ref_gpu, unsigned* offset_query_gpu, unsigned* offsetA_h, unsigned* offsetB_h, char* strA, char* strA_d, char* strB, char* strB_d, unsigned totalLengthA, unsigned totalLengthB){
	errCheck(hipMemcpyAsync(offset_ref_gpu, offsetA_h, (batch_size) * sizeof(int), hipMemcpyHostToDevice, curr_stream->stream));
    errCheck(hipMemcpyAsync(offset_query_gpu, offsetB_h, (batch_size) * sizeof(int), hipMemcpyHostToDevice, curr_stream->stream));
    errCheck(hipMemcpyAsync(strA_d, strA, totalLengthA * sizeof(char), hipMemcpyHostToDevice, curr_stream->stream));
    errCheck(hipMemcpyAsync(strB_d, strB, totalLengthB * sizeof(char), hipMemcpyHostToDevice, curr_stream->stream));
}

void driver::mem_copies_dth(short* ref_start_gpu, short* alAbeg, short* query_start_gpu,short* alBbeg, short* scores_gpu ,short* top_scores_cpu, int res_offset){
    errCheck(hipMemcpyAsync(alAbeg + res_offset, ref_start_gpu, batch_size * sizeof(short), hipMemcpyDeviceToHost, curr_stream->stream));
	errCheck(hipMemcpyAsync(alBbeg + res_offset, query_start_gpu, batch_size * sizeof(short), hipMemcpyDeviceToHost, curr_stream->stream));
    errCheck(hipMemcpyAsync(top_scores_cpu + res_offset, scores_gpu, batch_size * sizeof(short), hipMemcpyDeviceToHost, curr_stream->stream));
	errCheck(hipEventRecord(curr_stream->data_event, curr_stream->stream));
}

void driver::mem_copies_dth_mid(short* ref_end_gpu, short* alAend, short* query_end_gpu, short* alBend, int res_offset){
    errCheck(hipMemcpyAsync(alAend + res_offset, ref_end_gpu, batch_size * sizeof(short), hipMemcpyDeviceToHost, curr_stream->stream));
    errCheck(hipMemcpyAsync(alBend + res_offset, query_end_gpu, batch_size * sizeof(short), hipMemcpyDeviceToHost, curr_stream->stream));
}

void driver::mem_cpy_dth(int offset){
	mem_copies_dth(ref_start_gpu, results.ref_begin, query_start_gpu, results.query_begin, scores_gpu , results.top_scores, offset);
}

void driver::initialize_alignments(){
	errCheck(hipHostMalloc(&(results.ref_begin), sizeof(short)*total_alignments));
	errCheck(hipHostMalloc(&(results.ref_end), sizeof(short)*total_alignments));
	errCheck(hipHostMalloc(&(results.query_begin), sizeof(short)*total_alignments));
	errCheck(hipHostMalloc(&(results.query_end), sizeof(short)*total_alignments));
	errCheck(hipHostMalloc(&(results.top_scores), sizeof(short)*total_alignments));
}

aln_results driver::get_alignments(){
	return results;
}

void driver::dealloc_gpu_mem(){
	errCheck(hipFree(offset_ref_gpu));
	errCheck(hipFree(offset_query_gpu));
	errCheck(hipFree(ref_start_gpu));
	errCheck(hipFree(ref_end_gpu));
	errCheck(hipFree(query_start_gpu));
	errCheck(hipFree(query_end_gpu));
	errCheck(hipFree(ref_cstr_d));
	errCheck(hipFree(que_cstr_d));
}

void driver::cleanup(){
	errCheck(hipHostFree(offset_ref));
	errCheck(hipHostFree(offset_que));
	errCheck(hipHostFree(ref_cstr));
	errCheck(hipHostFree(que_cstr));
	dealloc_gpu_mem();
	hipStreamDestroy(curr_stream->stream);
	hipEventDestroy(curr_stream->kernel_event);
	hipEventDestroy(curr_stream->data_event);
}

void driver::allocate_gpu_mem(){
    errCheck(hipMalloc(&offset_query_gpu, (batch_size) * sizeof(int)));
    errCheck(hipMalloc(&offset_ref_gpu, (batch_size) * sizeof(int)));
    errCheck(hipMalloc(&ref_start_gpu, (batch_size) * sizeof(short)));
    errCheck(hipMalloc(&ref_end_gpu, (batch_size) * sizeof(short)));
    errCheck(hipMalloc(&query_end_gpu, (batch_size) * sizeof(short)));
    errCheck(hipMalloc(&query_start_gpu, (batch_size) * sizeof(short)));
    errCheck(hipMalloc(&scores_gpu, (batch_size) * sizeof(short)));
}

size_t ADEPT::get_batch_size(int gpu_id, int max_q_size, int max_r_size, int per_gpu_mem){
	hipDeviceProp_t prop;
	errCheck(hipGetDeviceProperties(&prop, gpu_id));
	size_t gpu_mem_avail = (double)prop.totalGlobalMem * (double)per_gpu_mem/100;
	size_t gpu_mem_per_align = max_q_size + max_r_size + 2 * sizeof(int) + 5 * sizeof(short);
	size_t max_concurr_aln = floor(((double)gpu_mem_avail)/gpu_mem_per_align);

	if (max_concurr_aln > 50000)
		return 50000;
	else
		return max_concurr_aln;
}

bool driver::kernel_done(){
	auto status = hipEventQuery(curr_stream->kernel_event);
	if(status == hipSuccess)
		return true;
	else
		return false;
}

bool driver::dth_done(){
	auto status = hipEventQuery(curr_stream->data_event);
	if(status == hipSuccess)
		return true;
	else
		return false;
}

void driver::kernel_synch(){
	errCheck(hipEventSynchronize(curr_stream->kernel_event));
}

void driver::dth_synch(){
	errCheck(hipEventSynchronize(curr_stream->data_event));
}

aln_results ADEPT::thread_launch(std::vector<std::string> ref_vec, std::vector<std::string> que_vec, ADEPT::ALG_TYPE algorithm, ADEPT::SEQ_TYPE sequence, ADEPT::CIGAR cigar_avail, int max_ref_size, int max_que_size, int batch_size, int dev_id, short scores[]){
	int alns_this_gpu = ref_vec.size();
	int iterations = (alns_this_gpu + (batch_size-1))/batch_size;
	if(iterations == 0) iterations = 1;
	int left_over = alns_this_gpu%batch_size;
	int batch_last_it = batch_size;
	if(left_over > 0)	batch_last_it = left_over;

	std::vector<std::vector<std::string>> its_ref_vecs;
	std::vector<std::vector<std::string>> its_que_vecs;
	int my_cpu_id = dev_id;//omp_get_thread_num();

	std::cout <<"total alignments:"<<alns_this_gpu<<" thread:"<<my_cpu_id<<std::endl;
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
		std::cout<<"iteration:"<<i<<" on gpu:"<<dev_id<<std::endl;
		sw_driver_loc.kernel_launch(its_ref_vecs[i], its_que_vecs[i], i * batch_size);
		sw_driver_loc.mem_cpy_dth(i * batch_size);
		sw_driver_loc.dth_synch();
	}

	auto loc_results = sw_driver_loc.get_alignments();// results for all iterations are available now
	sw_driver_loc.cleanup();
	return loc_results;
 }

all_alns ADEPT::multi_gpu(std::vector<std::string> ref_sequences, std::vector<std::string> que_sequences, ADEPT::ALG_TYPE algorithm, ADEPT::SEQ_TYPE sequence, ADEPT::CIGAR cigar_avail, int max_ref_size, int max_que_size, short scores[], int dev_to_use, int batch_size_){
	if(batch_size_ == -1)
		batch_size_ = ADEPT::get_batch_size(0, max_que_size, max_ref_size, 100);
	int total_alignments = ref_sequences.size();
  	int num_gpus;
	hipGetDeviceCount(&num_gpus);
	std::cout << "Available "<< num_gpus<< " GPUs"<<std::endl;
	unsigned batch_size = batch_size_;
	if(num_gpus < dev_to_use)
		dev_to_use = num_gpus;
	else
		num_gpus = dev_to_use;
	
	std::cout << "Using "<< num_gpus<< " GPUs"<<std::endl;
	//total_alignments = alns_per_batch;
	std::cout << "Batch Size:"<< batch_size<<std::endl;
	std::cout << "Total Alignments:"<< total_alignments<<std::endl;
    std::cout << "Total devices:"<< num_gpus<<std::endl;

	std::vector<std::vector<std::string>> ref_batch_gpu;
	std::vector<std::vector<std::string>> que_batch_gpu;
	int alns_per_gpu = total_alignments/num_gpus;
	int left_over = total_alignments%num_gpus;

	std::cout<< "Alns per GPU:"<<alns_per_gpu<<std::endl;
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
		std::cout<<"gpu:"<<i<<" has alns:"<<temp_ref.size()<<std::endl;
	}
  //omp_set_num_threads(num_gpus);
  all_alns global_results(num_gpus);
  global_results.per_gpu = alns_per_gpu;
  global_results.left_over = left_over;
  global_results.gpus = num_gpus;

  std::vector<std::thread> threads;
  auto lambda_call = [&](int my_cpu_id){
	 global_results.results[my_cpu_id] = ADEPT::thread_launch(ref_batch_gpu[my_cpu_id], que_batch_gpu[my_cpu_id], algorithm, sequence, cigar_avail, max_ref_size, max_que_size, batch_size, my_cpu_id, scores); 
  };

  for(int tid = 0; tid < num_gpus; tid++){
	  threads.push_back(std::thread(lambda_call, tid));
  }

  for (auto &thrd : threads){
	  thrd.join();
  }

  threads.clear();
//   #pragma omp parallel
//   {
//     int my_cpu_id = omp_get_thread_num();
// 	global_results.results[my_cpu_id] = ADEPT::thread_launch(ref_batch_gpu[my_cpu_id], que_batch_gpu[my_cpu_id], algorithm, sequence, cigar_avail, max_ref_size, max_que_size, batch_size, my_cpu_id, scores);
//     #pragma omp barrier
//   }

return global_results;
}