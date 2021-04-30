#include "kernel.hpp"

using namespace ADEPT;
unsigned getMaxLength (std::vector<std::string> v)
{
  unsigned maxLength = 0;
  for(auto str : v){
    if(maxLength < str.length()){
      maxLength = str.length();
    }
  }
  return maxLength;
}

struct adept_stream{
	cudaStream_t stream;
};

void driver::initialize(short scores[], ALG_TYPE _algorithm, SEQ_TYPE _sequence, CIGAR _cigar_avail, int _gpu_id, std::vector<std::string> ref_seqs, std::vector<std::string> query_seqs){
	algorithm = _algorithm, sequence = _sequence, cigar_avail = _cigar_avail;
	if(sequence == SEQ_TYPE::DNA){
		match_score = scores[0], mismatch_score = scores[1], gap_start = scores[2], gap_extend = scores[3];
	}

	gpu_id = _gpu_id;
    	cudaSetDevice(gpu_id);
        cudaStreamCreate(&(curr_stream.stream));

    	total_alignments = ref_seqs.size();
    	max_ref_size = getMaxLength(ref_seqs);
    	max_que_size = getMaxLength(query_seqs);
    	//host pinned memory for offsets
    	cudaMallocHost(&offset_ref, sizeof(int) * total_alignments);
    	cudaMallocHost(&offset_que, sizeof(int) * total_alignments);
    	//host pinned memory for sequences
    	cudaMallocHost(&ref_cstr, sizeof(char) * max_ref_size * total_alignments);
    	cudaMallocHost(&que_cstr, sizeof(char) * max_que_size * total_alignments);
    	// host pinned memory for results
   	initialize_alignments(results, total_alignments);
    	//device memory for sequences
    	cudaErrchk(cudaMalloc(&ref_cstr_d, sizeof(char) * max_ref_size * total_alignments));
    	cudaErrchk(cudaMalloc(&que_cstr_d,  sizeof(char)* max_que_size * total_alignments));
    	//device memory for offsets and results
    	allocate_gpu_mem(total_alignments);

    	//preparing offsets 
    	unsigned running_sum = 0;
    	for(int i = 0; i < total_alignments; i++)
        {
        	running_sum +=ref_seqs[i].size();
            	offset_ref[i] = running_sum;
        }
    	total_length_ref = offset_ref[total_alignments - 1];

    	running_sum = 0;
    	for(int i = 0; i < query_seqs.size(); i++)
    	{
        	running_sum +=query_seqs[i].size();
        	offset_que[i] = running_sum; 
    	}
    	total_length_que = offset_que[total_alignments - 1];

    	//moving sequences from vector to cstrings
    	unsigned offsetSumA = 0;
    	unsigned offsetSumB = 0;

 	for(int i = 0; i < ref.size(); i++)
    	{
        	char* seqptrA = ref_cstr + offsetSumA;  
        	memcpy(seqptrA, ref_seqs[i].c_str(), ref_seqs[i].size());
        	char* seqptrB = que_cstr + offsetSumB;
        	memcpy(seqptrB, query_seqs[i].c_str(), que_seqs[i].size());
        	offsetSumA += ref_seqs[i].size();
        	offsetSumB += que_seqs[i].size();
    	}

    	//move data asynchronously to GPU
    	mem_cpy_htd(offset_ref_gpu, offset_query_gpu, offset_ref, offset_que, ref_cstr, ref_cstr_d, que_cstr, que_cstr_d, total_length_ref,  total_length_que, total_alignments); // TODO: add streams
}

void driver::kernel_launch(){
	unsigned minSize = (max_que_size < max_ref_size) ? max_que_size : max_ref_size;
	unsigned totShmem = 3 * (minSize + 1) * sizeof(short);
	unsigned alignmentPad = 4 + (4 - totShmem % 4);
	size_t   ShmemBytes = totShmem + alignmentPad;
	if(ShmemBytes > 48000)
        	cudaFuncSetAttribute(gpu_bsw::sequence_dna_combo, cudaFuncAttributeMaxDynamicSharedMemorySize, ShmemBytes);
	kernel::dna_kernel<<<total_alignments, minSize, ShmemBytes>>>(ref_cstr_d, que_cstr_d, offset_ref_gpu, offset_query_gpu, ref_start_gpu, ref_end_gpu, query_start_gpu, query_end_gpu, scores_gpu, match_score, mismatch_score, start_gap, extend_gap);
}

void driver::mem_cpy_dth(){
	asynch_mem_copies_dth_mid(ref_end_gpu, ref_end , query_end_gpu, query_end, total_alignments);
	asynch_mem_copies_dth(ref_start_gpu, ref_begin, query_start_gpu, query_begin, scores_gpu , top_scores, total_alignments);
}

void driver::cleanup(){
	cudaErrchk(cudaFree(ref_cstr_d));
        cudaErrchk(cudaFree(que_cstr_d));
        cudaFreeHost(offset_ref);
        cudaFreeHost(offset_que);
        cudaFreeHost(ref_cstr);
        cudaFreeHost(que_cstr);
        dealloc_gpu_mem();
}
