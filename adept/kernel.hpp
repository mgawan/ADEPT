#ifndef KERNEL_HPP
#define KERNEL_HPP
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sys/time.h>
//#include "driver.hpp"

#define NUM_OF_AA 21
#define ENCOD_MAT_SIZE 91
#define SCORE_MAT_SIZE 576
namespace kernel{
__device__ short
warpReduceMax_with_index(short val, short& myIndex, short& myIndex2, unsigned lengthSeqB, bool reverse);

__device__ short
warpReduceMax(short val, unsigned lengthSeqB);

__device__ short
blockShuffleReduce_with_index(short myVal, short& myIndex, short& myIndex2, unsigned lengthSeqB, bool reverse);

__device__ short
blockShuffleReduce(short val, unsigned lengthSeqB);

__device__ __host__ short
           findMax(short array[], int length, int* ind);

__device__ __host__ short
            findMaxFour(short first, short second, short third, short fourth);

__device__ void
traceBack(short current_i, short current_j, short* seqA_align_begin,
          short* seqB_align_begin, const char* seqA, const char* seqB, short* I_i,
          short* I_j, unsigned lengthSeqB, unsigned lengthSeqA, unsigned int* diagOffset);

__global__ void
dna_kernel(char* seqA_array, char* seqB_array, unsigned* prefix_lengthA,
                unsigned* prefix_lengthB, short* seqA_align_begin, short* seqA_align_end,
                short* seqB_align_begin, short* seqB_align_end, short* top_scores, short matchScore, short misMatchScore, short startGap, short extendGap, bool reverse);

__global__ void
aa_kernel(char* seqA_array, char* seqB_array, unsigned* prefix_lengthA,
                unsigned* prefix_lengthB, short* seqA_align_begin, short* seqA_align_end, 
                short* seqB_align_begin, short* seqB_align_end, short* top_scores, short startGap, short extendGap, short* scoring_matrix, short* encoding_matrix, bool reverse);
}
#endif
