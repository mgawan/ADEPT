#include "../../adept/driver.hpp"
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


	unsigned batch_size = ADEPT::get_batch_size(0, 300, 1200, 100);// batch size per GPU
  std::cout<<"batch:"<<batch_size<<"\n";
  std::array<short, 2> scores = {3,-3};
  ADEPT::gap_scores gaps(-6,-1);
 auto all_results = ADEPT::multi_gpu(ref_sequences, que_sequences, ADEPT::ALG_TYPE::SW, ADEPT::SEQ_TYPE::DNA, ADEPT::CIGAR::YES, 1200, 300, scores.data(), gaps, batch_size);
 
 ofstream results_file(out_file);
 int tot_gpus = all_results.gpus;
 for(int gpus = 0; gpus < tot_gpus; gpus++){
   int this_count = all_results.per_gpu;
   if(gpus == tot_gpus - 1) this_count += all_results.left_over;
  for(int k = 0; k < this_count; k++){
    results_file<<all_results.results[gpus].top_scores[k]<<"\t"<<all_results.results[gpus].ref_begin[k]<<"\t"<<all_results.results[gpus].ref_end[k] - 1<<
    "\t"<<all_results.results[gpus].query_begin[k]<<"\t"<<all_results.results[gpus].query_end[k] - 1<<endl;
  }
  }

  for(int i = 0; i < tot_gpus; i++)
    all_results.results[i].free_results();
	
  return 0;
}
