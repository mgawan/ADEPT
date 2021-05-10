#include "kernel.hpp"
#include "driver.hpp"

#define cudaErrchk(ans)                                                                  \
{                                                                                    \
    gpuAssert((ans), __FILE__, __LINE__);                                            \
}
inline void
gpuAssert(cudaError_t code, const char* file, int line, bool abort = true){
    if(code != cudaSuccess){
        fprintf(stderr, "GPUassert: %s %s %d cpu:%d\n", cudaGetErrorString(code), file, line);
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
	cudaStream_t stream;
	cudaEvent_t kernel_event;
	cudaEvent_t data_event;
	
	adept_stream(int gpu){
		cudaErrchk(cudaSetDevice(gpu));
		cudaErrchk(cudaStreamCreate(&stream));
		cudaErrchk(cudaEventCreateWithFlags(&kernel_event, cudaEventBlockingSync));
		cudaErrchk(cudaEventCreateWithFlags(&data_event, cudaEventBlockingSync));
	}
};


void driver::initialize(short scores[], ALG_TYPE _algorithm, SEQ_TYPE _sequence, CIGAR _cigar_avail, int _max_ref_size, int _max_query_size, int batch_size, int _gpu_id){
	algorithm = _algorithm, sequence = _sequence, cigar_avail = _cigar_avail;
	if(sequence == SEQ_TYPE::DNA){
		match_score = scores[0], mismatch_score = scores[1], gap_start = scores[2], gap_extend = scores[3];
	}
	gpu_id = _gpu_id;
	curr_stream = new adept_stream(gpu_id);

	total_alignments = batch_size;
	max_ref_size = _max_ref_size;
	max_que_size = _max_query_size;
	//host pinned memory for offsets
	cudaErrchk(cudaMallocHost(&offset_ref, sizeof(int) * total_alignments));
	cudaErrchk(cudaMallocHost(&offset_que, sizeof(int) * total_alignments));
	//host pinned memory for sequences
	cudaErrchk(cudaMallocHost(&ref_cstr, sizeof(char) * max_ref_size * total_alignments));
	cudaErrchk(cudaMallocHost(&que_cstr, sizeof(char) * max_que_size * total_alignments));
	// host pinned memory for results
	initialize_alignments();
	//device memory for sequences
	cudaErrchk(cudaMalloc(&ref_cstr_d, sizeof(char) * max_ref_size * total_alignments));
	cudaErrchk(cudaMalloc(&que_cstr_d,  sizeof(char)* max_que_size * total_alignments));
	//device memory for offsets and results
	allocate_gpu_mem();

}

void driver::kernel_launch(std::vector<std::string> ref_seqs, std::vector<std::string> query_seqs){
	if(ref_seqs.size() != total_alignments)
		std::cerr << "INITIALIZATION ERROR: driver was initialized with a batch size that does not match to the vector passed to kernel\n";
	//preparing offsets 
	unsigned running_sum = 0;
	for(int i = 0; i < total_alignments; i++){
		running_sum +=ref_seqs[i].size();
		offset_ref[i] = running_sum;
	}
	total_length_ref = offset_ref[total_alignments - 1];

	running_sum = 0;
	for(int i = 0; i < query_seqs.size(); i++){
		running_sum +=query_seqs[i].size();
		offset_que[i] = running_sum; 
	}
	total_length_que = offset_que[total_alignments - 1];

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
	if(ShmemBytes > 48000)
        cudaFuncSetAttribute(kernel::dna_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, ShmemBytes);
	kernel::dna_kernel<<<total_alignments, minSize, ShmemBytes, curr_stream->stream>>>(ref_cstr_d, que_cstr_d, offset_ref_gpu, offset_query_gpu, ref_start_gpu, ref_end_gpu, query_start_gpu, query_end_gpu, scores_gpu, match_score, mismatch_score, gap_start, gap_extend, false);
	mem_copies_dth_mid(ref_end_gpu, results.ref_end , query_end_gpu, results.query_end);
	cudaStreamSynchronize(curr_stream->stream);
	int new_length = get_new_min_length(results.ref_end, results.query_end, total_alignments);
	kernel::dna_kernel<<<total_alignments, new_length, ShmemBytes, curr_stream->stream>>>(ref_cstr_d, que_cstr_d, offset_ref_gpu, offset_query_gpu, ref_start_gpu, ref_end_gpu, query_start_gpu, query_end_gpu, scores_gpu, match_score, mismatch_score, gap_start, gap_extend, true);
	cudaErrchk(cudaEventRecord(curr_stream->kernel_event, curr_stream->stream));
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
	cudaErrchk(cudaMemcpyAsync(offset_ref_gpu, offsetA_h, (total_alignments) * sizeof(int), cudaMemcpyHostToDevice, curr_stream->stream));
    cudaErrchk(cudaMemcpyAsync(offset_query_gpu, offsetB_h, (total_alignments) * sizeof(int), cudaMemcpyHostToDevice, curr_stream->stream));
    cudaErrchk(cudaMemcpyAsync(strA_d, strA, totalLengthA * sizeof(char), cudaMemcpyHostToDevice, curr_stream->stream));
    cudaErrchk(cudaMemcpyAsync(strB_d, strB, totalLengthB * sizeof(char), cudaMemcpyHostToDevice, curr_stream->stream));
}

void driver::mem_copies_dth(short* ref_start_gpu, short* alAbeg, short* query_start_gpu,short* alBbeg, short* scores_gpu ,short* top_scores_cpu){
    cudaErrchk(cudaMemcpyAsync(alAbeg, ref_start_gpu, total_alignments * sizeof(short), cudaMemcpyDeviceToHost, curr_stream->stream));
	cudaErrchk(cudaMemcpyAsync(alBbeg, query_start_gpu, total_alignments * sizeof(short), cudaMemcpyDeviceToHost, curr_stream->stream));
    cudaErrchk(cudaMemcpyAsync(top_scores_cpu, scores_gpu, total_alignments * sizeof(short), cudaMemcpyDeviceToHost, curr_stream->stream));
	cudaErrchk(cudaEventRecord(curr_stream->data_event, curr_stream->stream));
}

void driver::mem_copies_dth_mid(short* ref_end_gpu, short* alAend, short* query_end_gpu, short* alBend){
    cudaErrchk(cudaMemcpyAsync(alAend, ref_end_gpu, total_alignments * sizeof(short), cudaMemcpyDeviceToHost, curr_stream->stream));
    cudaErrchk(cudaMemcpyAsync(alBend, query_end_gpu, total_alignments * sizeof(short), cudaMemcpyDeviceToHost, curr_stream->stream));
}

void driver::mem_cpy_dth(){
	mem_copies_dth(ref_start_gpu, results.ref_begin, query_start_gpu, results.query_begin, scores_gpu , results.top_scores);
}

void driver::initialize_alignments(){
	cudaErrchk(cudaMallocHost(&(results.ref_begin), sizeof(short)*total_alignments));
	cudaErrchk(cudaMallocHost(&(results.ref_end), sizeof(short)*total_alignments));
	cudaErrchk(cudaMallocHost(&(results.query_begin), sizeof(short)*total_alignments));
	cudaErrchk(cudaMallocHost(&(results.query_end), sizeof(short)*total_alignments));
	cudaErrchk(cudaMallocHost(&(results.top_scores), sizeof(short)*total_alignments));
}

aln_results driver::get_alignments(){
	return results;
}

void driver::dealloc_gpu_mem(){
	cudaErrchk(cudaFree(offset_ref_gpu));
	cudaErrchk(cudaFree(offset_query_gpu));
	cudaErrchk(cudaFree(ref_start_gpu));
	cudaErrchk(cudaFree(ref_end_gpu));
	cudaErrchk(cudaFree(query_start_gpu));
	cudaErrchk(cudaFree(query_end_gpu));
	cudaErrchk(cudaFree(ref_cstr_d));
	cudaErrchk(cudaFree(que_cstr_d));
}

void driver::cleanup(){
	cudaErrchk(cudaFreeHost(offset_ref));
	cudaErrchk(cudaFreeHost(offset_que));
	cudaErrchk(cudaFreeHost(ref_cstr));
	cudaErrchk(cudaFreeHost(que_cstr));
	dealloc_gpu_mem();
	cudaStreamDestroy(curr_stream->stream);
	cudaEventDestroy(curr_stream->kernel_event);
	cudaEventDestroy(curr_stream->data_event);
}

void driver::free_results(){
    cudaErrchk(cudaFreeHost(results.ref_begin));
    cudaErrchk(cudaFreeHost(results.ref_end));
    cudaErrchk(cudaFreeHost(results.query_begin));
    cudaErrchk(cudaFreeHost(results.query_end));
    cudaErrchk(cudaFreeHost(results.top_scores));
}

void driver::allocate_gpu_mem(){
    cudaErrchk(cudaMalloc(&offset_query_gpu, (total_alignments) * sizeof(int)));
    cudaErrchk(cudaMalloc(&offset_ref_gpu, (total_alignments) * sizeof(int)));
    cudaErrchk(cudaMalloc(&ref_start_gpu, (total_alignments) * sizeof(short)));
    cudaErrchk(cudaMalloc(&ref_end_gpu, (total_alignments) * sizeof(short)));
    cudaErrchk(cudaMalloc(&query_end_gpu, (total_alignments) * sizeof(short)));
    cudaErrchk(cudaMalloc(&query_start_gpu, (total_alignments) * sizeof(short)));
    cudaErrchk(cudaMalloc(&scores_gpu, (total_alignments) * sizeof(short)));
}

size_t driver::get_batch_size(int gpu_id, int max_q_size, int max_r_size, int per_gpu_mem){
	cudaDeviceProp prop;
	cudaErrchk(cudaGetDeviceProperties(&prop, gpu_id));
	size_t gpu_mem_avail = (double)prop.totalGlobalMem * (double)per_gpu_mem/100;
	size_t gpu_mem_per_align = max_q_size + max_r_size + 2 * sizeof(int) + 5 * sizeof(short);
	size_t max_concurr_aln = floor(((double)gpu_mem_avail)/gpu_mem_per_align);

	if (max_concurr_aln > 50000)
		return 50000;
	else
		return max_concurr_aln;
}

bool driver::kernel_done(){
	auto status = cudaEventQuery(curr_stream->kernel_event);
	if(status == cudaSuccess)
		return true;
	else
		return false;
}

bool driver::dth_done(){
	auto status = cudaEventQuery(curr_stream->data_event);
	if(status == cudaSuccess)
		return true;
	else
		return false;
}

void driver::kernel_synch(){
	cudaErrchk(cudaEventSynchronize(curr_stream->kernel_event));
}

void driver::dth_synch(){
	cudaErrchk(cudaEventSynchronize(curr_stream->data_event));
}
