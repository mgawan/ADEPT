// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

// include headers
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sys/time.h>

// macros for amino acids and matrix sizes
#define NUM_OF_AA                    21
#define ENCOD_MAT_SIZE               91
#define SCORE_MAT_SIZE               576

// ------------------------------------------------------------------------------------ //

//
// namespace kernel
//
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
sequence_aa_kernel(char* seqA_array, char* seqB_array, unsigned* prefix_lengthA,
                    unsigned* prefix_lengthB, short* seqA_align_begin, short* seqA_align_end,
                    short* seqB_align_begin, short* seqB_align_end, short* top_scores, short startGap, short extendGap, short* scoring_matrix, short* encoding_matrix);

__global__ void
sequence_aa_reverse(char* seqA_array, char* seqB_array, unsigned* prefix_lengthA,
                    unsigned* prefix_lengthB, short* seqA_align_begin, short* seqA_align_end,
                    short* seqB_align_begin, short* seqB_align_end, short* top_scores, short startGap, short extendGap, short* scoring_matrix, short* encoding_matrix);
}