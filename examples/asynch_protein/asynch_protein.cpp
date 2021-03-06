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
constexpr int MAX_QUERY_LEN  =       800;
constexpr int GPU_ID         =         0;

constexpr unsigned int DATA_SIZE = std::numeric_limits<unsigned int>::max();

// scores
constexpr short MATCH          =  3;
constexpr short MISMATCH       = -3;
constexpr short GAP_OPEN       = -6;
constexpr short GAP_EXTEND     = -1;

bool verify_correctness(string file1, string file2);


int main(int argc, char* argv[]){

  std::cout <<                               std::endl;
  std::cout << "-----------------------" << std::endl;
  std::cout << "     ASYNC PROTEIN     " << std::endl;
  std::cout << "-----------------------" << std::endl;
  std::cout <<                               std::endl;

  // check command line arguments
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

  std::cout << "STATUS: Reading ref and query files" << std::endl;

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

  unsigned batch_size = ADEPT::get_batch_size(GPU_ID, MAX_QUERY_LEN, MAX_REF_LEN, 100); // batch size per GPU

  int work_cpu = 0;

  ADEPT::driver sw_driver;

  std::vector<short> scores_matrix = {4 ,-1 ,-2 ,-2 ,0 ,-1 ,-1 ,0 ,-2 ,-1 ,-1 ,-1 ,-1 ,-2 ,-1 ,1 ,0 ,-3 ,-2 ,0 ,-2 ,-1 ,0 ,-4 , -1 ,5 ,0 ,-2 ,-3 ,1 ,0 ,-2 ,0 ,-3 ,-2 ,2 ,-1 ,-3 ,-2 ,-1 ,-1 ,-3 ,-2 ,-3 ,-1 ,0 ,-1 ,-4 ,
    -2 ,0 ,6 ,1 ,-3 ,0 ,0 ,0 ,1 ,-3 ,-3 ,0 ,-2 ,-3 ,-2 ,1 ,0 ,-4 ,-2 ,-3 ,3 ,0 ,-1 ,-4 ,
    -2 ,-2 ,1 ,6 ,-3 ,0 ,2 ,-1 ,-1 ,-3 ,-4 ,-1 ,-3 ,-3 ,-1 ,0 ,-1 ,-4 ,-3 ,-3 ,4 ,1 ,-1 ,-4 ,
    0 ,-3 ,-3 ,-3 ,9 ,-3 ,-4 ,-3 ,-3 ,-1 ,-1 ,-3 ,-1 ,-2 ,-3 ,-1 ,-1 ,-2 ,-2 ,-1 ,-3 ,-3 ,-2 ,-4 ,
    -1 ,1 ,0 ,0 ,-3 ,5 ,2 ,-2 ,0 ,-3 ,-2 ,1 ,0 ,-3 ,-1 ,0 ,-1 ,-2 ,-1 ,-2 ,0 ,3 ,-1 ,-4 ,
    -1 ,0 ,0 ,2 ,-4 ,2 ,5 ,-2 ,0 ,-3 ,-3 ,1 ,-2 ,-3 ,-1 ,0 ,-1 ,-3 ,-2 ,-2 ,1 ,4 ,-1 ,-4 ,
    0 ,-2 ,0 ,-1 ,-3 ,-2 ,-2 ,6 ,-2 ,-4 ,-4 ,-2 ,-3 ,-3 ,-2 ,0 ,-2 ,-2 ,-3 ,-3 ,-1 ,-2 ,-1 ,-4 ,
    -2 ,0 ,1 ,-1 ,-3 ,0 ,0 ,-2 ,8 ,-3 ,-3 ,-1 ,-2 ,-1 ,-2 ,-1 ,-2 ,-2 ,2 ,-3 ,0 ,0 ,-1 ,-4 ,
    -1 ,-3 ,-3 ,-3 ,-1 ,-3 ,-3 ,-4 ,-3 ,4 ,2 ,-3 ,1 ,0 ,-3 ,-2 ,-1 ,-3 ,-1 ,3 ,-3 ,-3 ,-1 ,-4 ,
    -1 ,-2 ,-3 ,-4 ,-1 ,-2 ,-3 ,-4 ,-3 ,2 ,4 ,-2 ,2 ,0 ,-3 ,-2 ,-1 ,-2 ,-1 ,1 ,-4 ,-3 ,-1 ,-4 ,
    -1 ,2 ,0 ,-1 ,-3 ,1 ,1 ,-2 ,-1 ,-3 ,-2 ,5 ,-1 ,-3 ,-1 ,0 ,-1 ,-3 ,-2 ,-2 ,0 ,1 ,-1 ,-4 ,
    -1 ,-1 ,-2 ,-3 ,-1 ,0 ,-2 ,-3 ,-2 ,1 ,2 ,-1 ,5 ,0 ,-2 ,-1 ,-1 ,-1 ,-1 ,1 ,-3 ,-1 ,-1 ,-4 ,
    -2 ,-3 ,-3 ,-3 ,-2 ,-3 ,-3 ,-3 ,-1 ,0 ,0 ,-3 ,0 ,6 ,-4 ,-2 ,-2 ,1 ,3 ,-1 ,-3 ,-3 ,-1 ,-4 ,
    -1 ,-2 ,-2 ,-1 ,-3 ,-1 ,-1 ,-2 ,-2 ,-3 ,-3 ,-1 ,-2 ,-4 ,7 ,-1 ,-1 ,-4 ,-3 ,-2 ,-2 ,-1 ,-2 ,-4 ,
    1 ,-1 ,1 ,0 ,-1 ,0 ,0 ,0 ,-1 ,-2 ,-2 ,0 ,-1 ,-2 ,-1 ,4 ,1 ,-3 ,-2 ,-2 ,0 ,0 ,0 ,-4 ,
    0 ,-1 ,0 ,-1 ,-1 ,-1 ,-1 ,-2 ,-2 ,-1 ,-1 ,-1 ,-1 ,-2 ,-1 ,1 ,5 ,-2 ,-2 ,0 ,-1 ,-1 ,0 ,-4 ,
    -3 ,-3 ,-4 ,-4 ,-2 ,-2 ,-3 ,-2 ,-2 ,-3 ,-2 ,-3 ,-1 ,1 ,-4 ,-3 ,-2 ,11 ,2 ,-3 ,-4 ,-3 ,-2 ,-4 ,
    -2 ,-2 ,-2 ,-3 ,-2 ,-1 ,-2 ,-3 ,2 ,-1 ,-1 ,-2 ,-1 ,3 ,-3 ,-2 ,-2 ,2 ,7 ,-1 ,-3 ,-2 ,-1 ,-4 ,
    0 ,-3 ,-3 ,-3 ,-1 ,-2 ,-2 ,-3 ,-3 ,3 ,1 ,-2 ,1 ,-1 ,-2 ,-2 ,0 ,-3 ,-1 ,4 ,-3 ,-2 ,-1 ,-4 ,
    -2 ,-1 ,3 ,4 ,-3 ,0 ,1 ,-1 ,0 ,-3 ,-4 ,0 ,-3 ,-3 ,-2 ,0 ,-1 ,-4 ,-3 ,-3 ,4 ,1 ,-1 ,-4 ,
    -1 ,0 ,0 ,1 ,-3 ,3 ,4 ,-2 ,0 ,-3 ,-3 ,1 ,-1 ,-3 ,-1 ,0 ,-1 ,-3 ,-2 ,-2 ,1 ,4 ,-1 ,-4 ,
    0 ,-1 ,-1 ,-1 ,-2 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-2 ,0 ,0 ,-2 ,-1 ,-1 ,-1 ,-1 ,-1 ,-4 ,
    -4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,1}; // blosum 62

  ADEPT::gap_scores gaps(GAP_OPEN, GAP_EXTEND);
  int total_alignments = ref_sequences.size();

  sw_driver.initialize(scores_matrix, gaps, ADEPT::options::ALG_TYPE::SW, ADEPT::options::SEQ_TYPE::AA, ADEPT::options::CIGAR::YES,  ADEPT::options::SCORING::ALNS_AND_SCORE, MAX_REF_LEN, MAX_QUERY_LEN, total_alignments, batch_size, GPU_ID);

  std::cout << "STATUS: Launching driver" << std::endl << std::endl;

  sw_driver.kernel_launch(ref_sequences, que_sequences);

  while(sw_driver.kernel_done() != true)
    work_cpu++;

  sw_driver.mem_cpy_dth();
  
  while(sw_driver.dth_done() != true)
    work_cpu++;

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

  std::cout <<" total CPU work (counts) done while GPU was busy:"<< work_cpu << "\n";

  results.free_results(ADEPT::options::SCORING::ALNS_AND_SCORE);
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