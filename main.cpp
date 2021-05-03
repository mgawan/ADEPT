#include "adept/driver.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include<bits/stdc++.h>

using namespace std;

int main(int argc, char* argv[])
{
  string refFile = argv[1];
  string queFile = argv[2];
  string out_file = argv[3];

  vector<string> ref_sequences, que_sequences;
  string   myInLine;
  ifstream ref_file(refFile);
  ifstream quer_file(queFile);
  unsigned largestA = 0, largestB = 0;

  int totSizeA = 0, totSizeB = 0;

  if(ref_file.is_open())
  {
      while(getline(ref_file, myInLine))
      {
          if(myInLine[0] == '>'){
            continue;
          }else{
            string seq = myInLine;
            ref_sequences.push_back(seq);
            totSizeA += seq.size();
            if(seq.size() > largestA)
            {
                largestA = seq.size();
            }

          }

      }
      ref_file.close();
  }

  if(quer_file.is_open())
  {
      while(getline(quer_file, myInLine))
      {
          if(myInLine[0] == '>'){
            continue;
          }else{
            string seq = myInLine;
            que_sequences.push_back(seq);
            totSizeB += seq.size();
            if(seq.size() > largestB)
            {
                largestB = seq.size();
            }

          }

      }
      quer_file.close();
  }

  //ADEPT::aln_results results_test;
  ADEPT::driver sw_driver;
  std::array<short, 4> scores = {3,-3,-6,-1};
  sw_driver.initialize(scores, ADEPT::ALG_TYPE::SW, ADEPT::SEQ_TYPE::DNA, ADEPT::CIGAR::YES, 0, 
				      ref_sequences, que_sequences);
  
  sw_driver.kernel_launch();
  sw_driver.mem_cpy_dth();
  sw_driver.cleanup();

  ofstream results_file(out_file);
  for(int k = 0; k < ref_sequences.size(); k++){
    results_file<<sw_driver.results.top_scores[k]<<"\t"<<sw_driver.results.ref_begin[k]<<"\t"<<sw_driver.results.ref_end[k]<<
    "\t"<<sw_driver.results.query_begin[k]<<"\t"<<sw_driver.results.query_end[k]<<endl;
  }

  sw_driver.free_results();

  results_file.flush();
  results_file.close();

  return 0;
}
