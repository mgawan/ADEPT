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

void driver::initialize(short scores[], ALG_TYPE _algorithm, SEQ_TYPE _sequence, CIGAR _cigar_avail, int _gpu_id, std::vector<std::string> ref_seqs, std::vector<std::string> query_seqs){
    gpu_id = _gpu_id;
    cudaSetDevice(gpu_id);
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
    half_length_ref= 0; 
    half_length_que = 0;

    unsigned running_sum = 0;
    for(int i = 0; i < ref.size(); i++)
        {
            running_sum +=ref[i].size();
            offset_ref[i] = running_sum;//sequencesA[i].size();
        }
    total_length_ref = offset_ref[ref.size() - 1];

    running_sum = 0;
    for(int i = 0; i < que.size(); i++)
    {
        running_sum +=que[i].size();
        offset_que[i] = running_sum; //sequencesB[i].size();
    }
    total_length_que = offset_que[que.size() - 1];

    //moving sequences from vector to cstrings
    unsigned offsetSumA = 0;
    unsigned offsetSumB = 0;

    for(int i = 0; i < ref.size(); i++)
    {
        char* seqptrA = ref_cstr + offsetSumA;  
        memcpy(seqptrA, ref[i].c_str(), ref[i].size());
        char* seqptrB = que_cstr + offsetSumB;
        memcpy(seqptrB, que[i].c_str(), que[i].size());
        offsetSumA += ref[i].size();
        offsetSumB += que[i].size();
    }

    //move data asynchronously to GPU
    asynch_mem_copies_htd(offset_ref_gpu, offset_query_gpu, offset_ref, offset_que, ref_cstr, ref_cstr_d, que_cstr, que_cstr_d, total_length_ref,  total_length_que, ref.size()); // not using streams


}
