#include "driver.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include<bits/stdc++.h>
#include <thread>
#include <functional>

using namespace std;

constexpr int MAX_REF_LEN    =      1200;
constexpr int MAX_QUERY_LEN  =       300;
constexpr int GPU_ID         =         0;

constexpr unsigned int DATA_SIZE = std::numeric_limits<unsigned int>::max();;

// scores
constexpr short MATCH          =  3;
constexpr short MISMATCH       = -3;
constexpr short GAP_OPEN       = -6;
constexpr short GAP_EXTEND     = -1;

bool verify_correctness(string file1, string file2);

int main(int argc, char* argv[]){

  //
  // Print banner
  //
  std::cout <<                               std::endl;
  std::cout << "-----------------------" << std::endl;
  std::cout << "       SIMPLE DNA      " << std::endl;
  std::cout << "-----------------------" << std::endl;
  std::cout <<                               std::endl;

    if (argc < 5)
    {
        cout << "USAGE: asynch_sw <reference_file> <query_file> <output_file> <res_file>" << endl;
        exit(-1);
    }

  string refFile = argv[1];
  string queFile = argv[2];
  string out_file = argv[3];
  string res_file = argv[4];

  vector<string> ref_sequences, que_sequences;
  string   lineR, lineQ;

  ifstream ref_file(refFile);
  ifstream quer_file(queFile);

  unsigned largestA = 0, largestB = 0;

  int totSizeA = 0, totSizeB = 0;

  // extract reference sequences
  if(ref_file.is_open() && quer_file.is_open())
  {
    while(getline(ref_file, lineR))
    {
      getline(quer_file, lineQ);
      if(lineR[0] == '>')
      {
        if (lineR[0] == '>')
          continue;
        else
        {
          std::cout << "FATAL: Mismatch in lines" << std::endl;
          exit(-2);
        }
      }
      else
      {
        if (lineR.length() <= MAX_REF_LEN && lineQ.length() <= MAX_QUERY_LEN)
        {
          ref_sequences.push_back(lineR);
          que_sequences.push_back(lineQ);

          totSizeA += lineR.length();
          totSizeB += lineQ.length();

          if(lineR.length() > largestA)
            largestA = lineR.length();

          if(lineQ.length() > largestA)
            largestB = lineQ.length();
        }
      }
      if (ref_sequences.size() == DATA_SIZE)
          break;
    }

    ref_file.close();
    quer_file.close();
  }


	unsigned batch_size = ADEPT::get_batch_size(GPU_ID, MAX_QUERY_LEN, MAX_REF_LEN, 100);// batch size per GPU

  ADEPT::driver sw_driver;
  std::vector<short> scores = {MATCH, MISMATCH};
  ADEPT::gap_scores gaps(GAP_OPEN, GAP_EXTEND);

  int total_alignments = ref_sequences.size();
  auto kernel_sel = ADEPT::options::SCORING::ALNS_AND_SCORE;
  sw_driver.initialize(scores, gaps, ADEPT::options::ALG_TYPE::SW, ADEPT::options::SEQ_TYPE::DNA, ADEPT::options::CIGAR::YES, kernel_sel, MAX_REF_LEN, MAX_QUERY_LEN, total_alignments, batch_size, GPU_ID);

  std::cout << "STATUS: Launching driver" << std::endl << std::endl;

  sw_driver.kernel_launch(ref_sequences, que_sequences);
  sw_driver.mem_cpy_dth();
  sw_driver.dth_synch();


  auto results = sw_driver.get_alignments();

  ofstream results_file(out_file);

  std::cout << std::endl << "STATUS: Writing results..." << std::endl;

  // write the results header
  results_file << "alignment_scores\t"     << "reference_begin_location\t" << "reference_end_location\t" 
               << "query_begin_location\t" << "query_end_location"         << std::endl;

  for(int k = 0; k < ref_sequences.size(); k++){
    results_file<<results.top_scores[k]<<"\t"<<results.ref_begin[k]<<"\t"<<results.ref_end[k] - 1<<
    "\t"<<results.query_begin[k]<<"\t"<<results.query_end[k] - 1<<endl;
  }


  results.free_results(kernel_sel);
	sw_driver.cleanup();
  results_file.flush();
  results_file.close();

  int return_state = 0;
  if(!verify_correctness(res_file, out_file)) return_state = -1;
  
  if(return_state == 0){
    cout<< "correctness test passed"<<endl;
  }else{
    cout<< "correctness test failed"<<endl;
  }

  return return_state;
}

bool verify_correctness(string file1, string file2){
  ifstream ref_file(file1);
  ifstream test_file(file2);
  string ref_line, test_line;

  // extract reference sequences
  if(ref_file.is_open() && test_file.is_open())
  {
    while(getline(ref_file, ref_line) && getline(test_file, test_line)){
      if(test_line != ref_line){
        return false;
      }
    }
    ref_file.close();
    test_file.close();
  }

  return true;
}