#include "adept/driver.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include<bits/stdc++.h>
#include <thread>
#include <functional>

using namespace std;

int main(int argc, char* argv[]){
  string refFile = argv[1];
  string queFile = argv[2];
  string out_file = argv[3];

  vector<string> ref_sequences, que_sequences;
  string   myInLine;
  ifstream ref_file(refFile);
  ifstream quer_file(queFile);
  unsigned largestA = 0, largestB = 0;

  int totSizeA = 0, totSizeB = 0;

  if(ref_file.is_open()){
    while(getline(ref_file, myInLine)){
      if(myInLine[0] == '>'){
        continue;
      }else{
        string seq = myInLine;
        ref_sequences.push_back(seq);
        totSizeA += seq.size();
        if(seq.size() > largestA){
            largestA = seq.size();
        }
      }
    }
    ref_file.close();
  }

  if(quer_file.is_open()){
    while(getline(quer_file, myInLine)){
      if(myInLine[0] == '>'){
        continue;
      }else{
        string seq = myInLine;
        que_sequences.push_back(seq);
        totSizeB += seq.size();
        if(seq.size() > largestB){
            largestB = seq.size();
        }
      }
    }
    quer_file.close();
  }

	// int total_alignments = ref_sequences.size();
  // int dev_count;
	// cudaGetDeviceCount(&dev_count);
	unsigned batch_size = 50000;//get_batch_size(0, max_que_size, max_ref_size, 100);// batch size per GPU
	
	//total_alignments = alns_per_batch;
	// std::cout << "Batch Size:"<< batch_size<<"\n";
	// std::cout << "Total Alignments:"<< total_alignments<<"\n";
  // std::cout << "Total devices:"<< dev_count<<"\n";

	// std::vector<std::vector<std::string>> ref_batch_gpu;
	// std::vector<std::vector<std::string>> que_batch_gpu;
	// int alns_per_gpu = total_alignments/num_gpus;
	// int left_over = total_alignments%num_gpus;

	// std::cout<< "Alns per GPU:"<<alns_per_gpu<<"\n";
  std::array<short, 4> scores = {3,-3,-6,-1};

 auto all_results = ADEPT::multi_gpu(ref_sequences, que_sequences, ADEPT::ALG_TYPE::SW, ADEPT::SEQ_TYPE::DNA, ADEPT::CIGAR::YES, 1200, 300, batch_size, scores.data());
 std::cout<<"returning from multi gpu\n";
 ofstream results_file(out_file);
 int tot_gpus = 1;
 for(int gpus = 0; gpus < tot_gpus; gpus++){
   int this_count = all_results.per_gpu;
   if(gpus == tot_gpus - 1) this_count += all_results.left_over;
  for(int k = 0; k < this_count; k++){
    results_file<<all_results.results[gpus].top_scores[k]<<"\t"<<all_results.results[gpus].ref_begin[k]<<"\t"<<all_results.results[gpus].ref_end[k] - 1<<
    "\t"<<all_results.results[gpus].query_begin[k]<<"\t"<<all_results.results[gpus].query_end[k] - 1<<endl;
  }
  }
	// for(int i = 0; i < num_gpus ; i++){
	// 	std::vector<std::string>::const_iterator start_, end_;
	// 	start_ = ref_seqs.begin() + i * alns_per_gpu;
	// 	if(i == num_gpus -1)
	// 		end_ = ref_seqs.begin() + (i + 1) * alns_per_gpu + left_over;
	// 	else
	// 		end_ = ref_seqs.begin() + (i + 1) * alns_per_gpu;

	// 	std::vector<std::string> temp_ref(start_, end_);

	// 	start_ = query_seqs.begin() + i * alns_per_gpu;
	// 	if(i == num_gpus - 1)
	// 		end_ = query_seqs.begin() + (i + 1) * alns_per_gpu + left_over;
	// 	else
	// 		end_ = query_seqs.begin() + (i + 1) * alns_per_gpu;

	// 	std::vector<std::string> temp_que(start_, end_);

	// 	ref_batch_gpu.push_back(temp_ref);
	// 	que_batch_gpu.push_back(temp_que);
	// }
  // omp_set_num_threads(deviceCount);
  // ADEPT::aln_results global_results[dev_count];

  // #pragma omp parallel
  // {
  //   int my_cpu_id = omp_get_thread_num();
  //   global_results[my_cpu_id] = ADEPT::thread_launch(ref_batch_gpu[my_cpu_id], que_batch_gpu[my_cpu_id], batch_size, my_cpu_id, scores.data());
  //   #pragma omp barrier
  // }



  


  // ADEPT::driver sw_driver;
  // std::array<short, 4> scores = {3,-3,-6,-1};
  // sw_driver.initialize(scores.data(), ADEPT::ALG_TYPE::SW, ADEPT::SEQ_TYPE::DNA, ADEPT::CIGAR::YES, 1200, 300, 30000, 30000, 0);
  // sw_driver.kernel_launch(ref_sequences, que_sequences);
  // sw_driver.mem_cpy_dth();
  // sw_driver.dth_synch();

  // auto results = sw_driver.get_alignments();

  // ofstream results_file(out_file);
  // for(int k = 0; k < ref_sequences.size(); k++){
  //   results_file<<results.top_scores[k]<<"\t"<<results.ref_begin[k]<<"\t"<<results.ref_end[k] - 1<<
  //   "\t"<<results.query_begin[k]<<"\t"<<results.query_end[k] - 1<<endl;
  // }


  // sw_driver.cleanup();
  // sw_driver.free_results();
  // results_file.flush();
  // results_file.close();

  return 0;
}
