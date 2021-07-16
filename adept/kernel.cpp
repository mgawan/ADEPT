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

#include <CL/sycl.hpp>
#include "kernel.hpp"

using namespace sycl;

// ------------------------------------------------------------------------------------ //

SYCL_EXTERNAL inline short 
Akernel::warpReduceMax_with_index(short val, short&myIndex, short&myIndex2, int lengthSeqB, bool reverse, sycl::nd_item<1> &item)
{
    short myMax    = val;
    short newInd   = 0;
    short newInd2  = 0;
    short ind      = myIndex;
    short ind2     = myIndex2;

    auto sg = item.get_sub_group();
    int warpSize = sg.get_local_range()[0];
    short laneId = sg.get_local_id();

    // unsigned mask  = __ballot_sync(0xffffffff, item.get_local_id(0)< lengthSeqB);  // item.get_local_range().get(2)

    int rem = sycl::min(lengthSeqB, warpSize);
    // unsigned newmask;

    for(int offset = rem/2; rem > 0; offset = sycl::max(1, rem/2))
    {
        rem -=offset;

        short tempVal = sg.shuffle_down(val, offset);
        newInd  = sg.shuffle_down(ind, offset);
        newInd2 = sg.shuffle_down(ind2, offset);

        // make sure all shuffles are done
        sg.barrier();

        val = sycl::max(val, tempVal);

#if defined (INTEL_GPU)
        // explicitly handle the undefined cases for Intel GPU
        if (laneId + offset >= warpSize)
        {
            val = 0;
            newInd = 0;
            newInd2 = 0;
        }
#endif // INTEL_GPU

        if(val > myMax)
        {
            ind   = newInd;
            ind2  = newInd2;
            myMax = val;
        }
         else if(val == tempVal) // this is kind of redundant and has been done purely to match the results
                                // with SSW to get the smallest alignment with highest score. Theoreticaly
                                // all the alignmnts with same score are same.
        {
            if(reverse == true)
            {
                if (newInd2 > ind2 || ((newInd2 == ind2) && (newInd > ind)))
                {
                    ind = newInd;
                    ind2 = newInd2;
                }
            }
            else
            {
                if (newInd < ind || ((newInd == ind) && (newInd2 < ind2)))
                {
                    ind = newInd;
                    ind2 = newInd2;
                }
            }
        }
    }

    myIndex  = ind;
    myIndex2 = ind2;
    val      = myMax;
    return val;
}

// ------------------------------------------------------------------------------------ //

SYCL_EXTERNAL short
Akernel::blockShuffleReduce_with_index(short myVal, short& myIndex, short& myIndex2, bool reverse, sycl::nd_item<1> &item, short* locTots, short *locInds, short *locInds2)
{
    auto sg      = item.get_sub_group();
    auto gp      = item.get_group();
    int warpSize = sg.get_local_range()[0];

    short laneId = sg.get_local_id();
    short warpId = sg.get_group_id();
    int threadId = item.get_local_linear_id();
    int blockSize = item.get_local_range(0);

    short myInd  = myIndex;
    short myInd2 = myIndex2;

    myVal  = warpReduceMax_with_index(myVal, myInd, myInd2, blockSize, reverse, item);

    if(laneId == 0)
        locTots[warpId] = myVal;
    if(laneId == 0)
        locInds[warpId] = myInd;
    if(laneId == 0)
        locInds2[warpId] = myInd2;

    BARRIER(gp);

    int nblocks = sg.get_group_range().size();

    if(threadId < nblocks)  /////******//////
    {
        myVal  = locTots[threadId];
        myInd  = locInds[threadId];
        myInd2 = locInds2[threadId];
    }
    else
    {
        myVal  = 0;
        myInd  = -1;
        myInd2 = -1;
    }

    BARRIER(gp);

    if(warpId == 0)
    {
        myVal    = warpReduceMax_with_index(myVal, myInd, myInd2, nblocks, reverse, item);
        myIndex  = myInd;
        myIndex2 = myInd2;
    }

    return myVal;
}

// ------------------------------------------------------------------------------------ //

 short
Akernel::findMaxFour(short first, short second, short third, short fourth)
{
    short maxScore = 0;

    maxScore = sycl::max(first,second);
    maxScore = sycl::max(maxScore, third);
    maxScore = sycl::max(maxScore, fourth);

    return maxScore;
}

// ------------------------------------------------------------------------------------ //

//
// dna kernel
//
SYCL_EXTERNAL void
Akernel::dna_kernel(char* seqA_array, char* seqB_array, int* prefix_lengthA,
                    int* prefix_lengthB, short* seqA_align_begin, short* seqA_align_end,
                    short* seqB_align_begin, short* seqB_align_end, short* top_scores, 
                    short matchScore, short misMatchScore, short startGap, short extendGap, 
                    bool reverse, sycl::nd_item<1> &item, 
                    char *is_valid_array,
                    short *sh_prev_E,
                    short *sh_prev_H,
                    short *sh_prev_prev_H,
                    short *local_spill_prev_E,
                    short *local_spill_prev_H,
                    short *local_spill_prev_prev_H,
                    short *locTots,
                    short *locInds,
                    short *locInds2)
{
    auto sg = item.get_sub_group();
    auto gp = item.get_group();

    int warpSize = sg.get_local_range()[0];
    int block_Id  = item.get_group_linear_id();
    int thread_Id = item.get_local_linear_id();

    short laneId = sg.get_local_id();
    short warpId = sg.get_group_id();

    int lengthSeqA;
    int lengthSeqB;
    
    // local pointers
    char*    seqA;
    char*    seqB;
    char* is_valid = &is_valid_array[0];
    char* longer_seq;

    // setting up block local sequences and their lengths.
    if(block_Id == 0)
    {
            lengthSeqA = prefix_lengthA[0];
            lengthSeqB = prefix_lengthB[0];
            seqA       = seqA_array;
            seqB       = seqB_array;
    }
    else
    {
            lengthSeqA = prefix_lengthA[block_Id] - prefix_lengthA[block_Id - 1];
            lengthSeqB = prefix_lengthB[block_Id] - prefix_lengthB[block_Id - 1];
            seqA       = seqA_array + prefix_lengthA[block_Id - 1];
            seqB       = seqB_array + prefix_lengthB[block_Id - 1];
    }

    if(reverse == true)
    {
        lengthSeqA = seqA_align_end[block_Id];
        lengthSeqB = seqB_align_end[block_Id];
    }

    int maxSize = lengthSeqA > lengthSeqB ? lengthSeqA : lengthSeqB;
    int minSize = lengthSeqA < lengthSeqB ? lengthSeqA : lengthSeqB;

    for(int p = thread_Id; p < minSize; p+=warpSize)
    {
        is_valid[p] = 0;
    }

    is_valid += minSize;

    for(int p = thread_Id; p < minSize; p+=warpSize)
    {
        is_valid[p] = 1;
    }

    is_valid += minSize;

    for(int p = thread_Id; p < minSize; p+=warpSize){
        is_valid[p] = 0;
    }

    char myColumnChar;

    // the shorter of the two strings is stored in thread registers
    if(lengthSeqA < lengthSeqB)
    {
        if(thread_Id < lengthSeqA)
        {
            if(reverse == true)
                myColumnChar = seqA[(lengthSeqA - 1) - thread_Id];
            else
                myColumnChar = seqA[thread_Id];  // read only once
            longer_seq = seqB;
        }
    }
    else
    {
        if(thread_Id < lengthSeqB)
        {
            if(reverse == true)
                myColumnChar = seqB[(lengthSeqB - 1) - thread_Id];
            else
                myColumnChar = seqB[thread_Id];
                longer_seq = seqA;
        }
    }

    BARRIER(gp); // this is required here so that complete sequence has been copied to shared memory

    int   i            = 1;
    short thread_max   = 0; // to maintain the thread max score
    short thread_max_i = 0; // to maintain the DP coordinate i for the longer string
    short thread_max_j = 0;// to maintain the DP cooirdinate j for the shorter string

    //initializing registers for storing diagonal values for three recent most diagonals (separate tables for
    //H, E and F)
    short _curr_H = 0, _curr_F = 0, _curr_E = 0;
    short _prev_H = 0, _prev_F = 0, _prev_E = 0;
    short _prev_prev_H = 0, _prev_prev_F = 0, _prev_prev_E = 0;
    short _temp_Val = 0;

    for(int diag = 0; diag < lengthSeqA + lengthSeqB - 1; diag++)
    {
        // iterate for the number of anti-diagonals
        is_valid = is_valid - (diag < minSize || diag >= maxSize); //move the pointer to left by 1 if cnd true

        _temp_Val = _prev_H; // value exchange happens here to setup registers for next iteration
        _prev_H = _curr_H;
        _curr_H = _prev_prev_H;
        _prev_prev_H = _temp_Val;
        _curr_H = 0;

        _temp_Val = _prev_E;
        _prev_E = _curr_E;
        _curr_E = _prev_prev_E;
        _prev_prev_E = _temp_Val;
        _curr_E = 0;

        _temp_Val = _prev_F;
        _prev_F = _curr_F;
        _curr_F = _prev_prev_F;
        _prev_prev_F = _temp_Val;
        _curr_F = 0;

        if(laneId == warpSize - 1)
        { 
            // if you are the last thread in your warp then spill your values to shmem

            sh_prev_E[warpId] = _prev_E;
            sh_prev_H[warpId] = _prev_H;
            sh_prev_prev_H[warpId] = _prev_prev_H;
        }

        if(diag >= maxSize)
        { 
            // if you are invalid in this iteration, spill your values to shmem

            local_spill_prev_E[thread_Id] = _prev_E;
            local_spill_prev_H[thread_Id] = _prev_H;
            local_spill_prev_prev_H[thread_Id] = _prev_prev_H;
        }

        BARRIER(gp); // this is needed so that all the shmem writes are completed.

        if (is_valid[thread_Id] && thread_Id < minSize)
        {
            auto sg = item.get_sub_group();
            // unsigned mask  = __ballot_sync(__activemask(), (is_valid[thread_Id] &&( thread_Id < minSize)));

            short fVal = _prev_F + extendGap;
            short hfVal = _prev_H + startGap;

            short valeShfl = sg.shuffle(_prev_E, laneId - 1);   //__shfl_sync(mask, _prev_E, laneId- 1, 32);
            short valheShfl =  sg.shuffle(_prev_H, laneId - 1); //__shfl_sync(mask, _prev_H, laneId - 1, 32);

            short eVal=0, heVal = 0;

            // when the previous thread has phased out, get value from shmem
            if(diag >= maxSize)
            {
                eVal = local_spill_prev_E[thread_Id - 1] + extendGap;
                heVal = local_spill_prev_H[thread_Id - 1]+ startGap;
            }
            else
            {
                eVal =((warpId !=0 && laneId == 0)?sh_prev_E[warpId-1]: valeShfl) + extendGap;
                heVal =((warpId !=0 && laneId == 0)?sh_prev_H[warpId-1]:valheShfl) + startGap;
            }
             // make sure that values for lane 0 in warp 0 is not undefined
            if(warpId == 0 && laneId == 0)
            {
                eVal = 0;
                heVal = 0;
            }

            _curr_F = (fVal > hfVal) ? fVal : hfVal;
            _curr_E = (eVal > heVal) ? eVal : heVal;
        
            short testShufll = sg.shuffle(_prev_prev_H, laneId - 1); //__shfl_sync(mask, _prev_prev_H, laneId - 1, 32);
            short final_prev_prev_H = 0;

            if(diag >= maxSize)
            {
                final_prev_prev_H = local_spill_prev_prev_H[thread_Id - 1];
            }
            else
            {
                final_prev_prev_H =(warpId !=0 && laneId == 0)?sh_prev_prev_H[warpId-1]:testShufll;
            }


            if(warpId == 0 && laneId == 0) final_prev_prev_H = 0;
            char to_comp = reverse == true ? longer_seq[maxSize -i] : longer_seq[i - 1];
            short diag_score = final_prev_prev_H + ((to_comp == myColumnChar)?matchScore:misMatchScore);
            _curr_H = findMaxFour(diag_score, _curr_F, _curr_E, 0);

            int _i = reverse == true ? maxSize - i : i;
            int _j = reverse == true ? minSize - thread_Id - 1: thread_Id + 1; 
            thread_max_i = (thread_max >= _curr_H) ? thread_max_i : _i;
            thread_max_j = (thread_max >= _curr_H) ? thread_max_j : _j;
            thread_max   = (thread_max >= _curr_H) ? thread_max : _curr_H;
            i++;
        }
        else
        {
            // we need these dummy shuffle operations for NVIDIA GPUs
            short valeShfl = sg.shuffle(_prev_E, laneId);   //__shfl_sync(mask, _prev_E, laneId- 1, 32);
            short valheShfl =  sg.shuffle(_prev_H, laneId); //__shfl_sync(mask, _prev_H, laneId - 1, 32);
            short testShufll = sg.shuffle(_prev_prev_H, laneId); //__shfl_sync(mask, _prev_prev_H, laneId - 1, 32);
        }

        BARRIER(gp); // why do I need this? commenting it out breaks it
    }

    thread_max = blockShuffleReduce_with_index(thread_max, thread_max_i, thread_max_j, reverse, item, locTots, locInds, locInds2);  // thread 0 will have the correct values

    if(reverse == true)
    {
        if(thread_Id == 0)
        {
            if(lengthSeqA < lengthSeqB)
            {
                seqB_align_begin[block_Id] = (thread_max_i);
                seqA_align_begin[block_Id] = (thread_max_j);
            }
            else
            {
                seqA_align_begin[block_Id] = (thread_max_i);
                seqB_align_begin[block_Id] = (thread_max_j);
            }
        }
    }
    else
    {  
        if(thread_Id == 0)
        {
            if(lengthSeqA < lengthSeqB)
            {
                seqB_align_end[block_Id] = thread_max_i;
                seqA_align_end[block_Id] = thread_max_j;
                top_scores[block_Id] = thread_max;
            }
            else
            {
                seqA_align_end[block_Id] = thread_max_i;
                seqB_align_end[block_Id] = thread_max_j;
                top_scores[block_Id] = thread_max;
            }
        }
    }
}

// ------------------------------------------------------------------------------------ //

//
// amino acid kernel
//
SYCL_EXTERNAL void
Akernel::aa_kernel(char* seqA_array, char* seqB_array, int* prefix_lengthA,
                                        int* prefix_lengthB, short* seqA_align_begin, short* seqA_align_end,
                                        short* seqB_align_begin, short* seqB_align_end, short* top_scores, short startGap, short extendGap, short* score_encode_matrix, bool reverse,
                                        sycl::nd_item<1> &item, 
                                        char *is_valid_array,
                                        short *sh_prev_E,
                                        short *sh_prev_H,
                                        short *sh_prev_prev_H,
                                        short *local_spill_prev_E,
                                        short *local_spill_prev_H,
                                        short *local_spill_prev_prev_H,
                                        short *sh_aa_encoding, 
                                        short *sh_aa_scoring,
                                        short *locTots,
                                        short *locInds,
                                        short *locInds2)
{

    auto sg = item.get_sub_group();
    auto gp = item.get_group();

    int block_Id  = item.get_group_linear_id();
    int thread_Id = item.get_local_linear_id();
    short laneId = sg.get_local_id();
    short warpId = sg.get_group_id();

    int warpSize = sg.get_local_range()[0];

    int lengthSeqA;
    int lengthSeqB;
    
    // local pointers
    char*    seqA;
    char*    seqB;
    char* longer_seq;

    char *is_valid = &is_valid_array[0];

    short *scoring_matrix = score_encode_matrix;
    short *encoding_matrix = score_encode_matrix + SCORE_MAT_SIZE;

    // setting up block local sequences and their lengths.
    if(block_Id == 0)
    {
        lengthSeqA = prefix_lengthA[0];
        lengthSeqB = prefix_lengthB[0];
        seqA       = seqA_array;
        seqB       = seqB_array;
    }
    else
    {
        lengthSeqA = prefix_lengthA[block_Id] - prefix_lengthA[block_Id - 1];
        lengthSeqB = prefix_lengthB[block_Id] - prefix_lengthB[block_Id - 1];
        seqA       = seqA_array + prefix_lengthA[block_Id - 1];
        seqB       = seqB_array + prefix_lengthB[block_Id - 1];
    }

    if(reverse == true)
    {
        lengthSeqA = seqA_align_end[block_Id];
        lengthSeqB = seqB_align_end[block_Id];
    }

    // what is the max length and what is the min length
    int maxSize = lengthSeqA > lengthSeqB ? lengthSeqA : lengthSeqB;
    int minSize = lengthSeqA < lengthSeqB ? lengthSeqA : lengthSeqB;

    // shared memory space for storing longer of the two strings
    for(int p = thread_Id; p < minSize; p+=warpSize)
        is_valid[p] = 0;

    is_valid += minSize;

    for(int p = thread_Id; p < minSize; p+=warpSize)
        is_valid[p] = 1;


    is_valid += minSize;
    for(int p = thread_Id; p < minSize; p+=warpSize)
        is_valid[p] = 0;


    char myColumnChar;

    // the shorter of the two strings is stored in thread registers
    if(lengthSeqA < lengthSeqB)
    {
        if(thread_Id < lengthSeqA)
        {
            if(reverse == true)
                myColumnChar = seqA[(lengthSeqA - 1) - thread_Id];
            else
                myColumnChar = seqA[thread_Id];  // read only once

            longer_seq = seqB;
        }
    }
    else
    {
        if(thread_Id < lengthSeqB)
        {
            if(reverse == true)
                myColumnChar = seqB[(lengthSeqB - 1) - thread_Id];
            else
                myColumnChar = seqB[thread_Id];

            longer_seq = seqA;
        }
    }

    BARRIER(gp); // this is required here so that complete sequence has been copied to shared memory

    int   i            = 1;
    short thread_max   = 0; // to maintain the thread max score
    short thread_max_i = 0; // to maintain the DP coordinate i for the longer string
    short thread_max_j = 0;// to maintain the DP cooirdinate j for the shorter string

    //initializing registers for storing diagonal values for three recent most diagonals (separate tables for H, E and F)

    short _curr_H = 0, _curr_F = 0, _curr_E = 0;
    short _prev_H = 0, _prev_F = 0, _prev_E = 0;
    short _prev_prev_H = 0, _prev_prev_F = 0, _prev_prev_E = 0;
    short _temp_Val = 0;

    int max_threads = item.get_local_range(0);

    for(int p = thread_Id; p < SCORE_MAT_SIZE; p+=max_threads)
        sh_aa_scoring[p] = scoring_matrix[p];

    for(int p = thread_Id; p < ENCOD_MAT_SIZE; p+=max_threads)
        sh_aa_encoding[p] = encoding_matrix[p];

    BARRIER(gp); // to make sure all shmem allocations have been initialized

    for(int diag = 0; diag < lengthSeqA + lengthSeqB - 1; diag++)
    {
        // iterate for the number of anti-diagonals
        is_valid = is_valid - (diag < minSize || diag >= maxSize); //move the pointer to left by 1 if cnd true

        _temp_Val = _prev_H; // value exchange happens here to setup registers for next iteration
        _prev_H = _curr_H;
        _curr_H = _prev_prev_H;
        _prev_prev_H = _temp_Val;
        _curr_H = 0;

        _temp_Val = _prev_E;
        _prev_E = _curr_E;
        _curr_E = _prev_prev_E;
        _prev_prev_E = _temp_Val;
        _curr_E = 0;

        _temp_Val = _prev_F;
        _prev_F = _curr_F;
        _curr_F = _prev_prev_F;
        _prev_prev_F = _temp_Val;
        _curr_F = 0;

        if(laneId == warpSize - 1)
        { 
            // if you are the last thread in your warp then spill your values to shmem
            sh_prev_E[warpId] = _prev_E;
            sh_prev_H[warpId] = _prev_H;
            sh_prev_prev_H[warpId] = _prev_prev_H;
        }

        if(diag >= maxSize)
        {
            // if you are invalid in this iteration, spill your values to shmem
            local_spill_prev_E[thread_Id] = _prev_E;
            local_spill_prev_H[thread_Id] = _prev_H;
            local_spill_prev_prev_H[thread_Id] = _prev_prev_H;
        }

        BARRIER(gp); // this is needed so that all the shmem writes are completed.

        if(is_valid[thread_Id] && thread_Id < minSize)
        {
            short fVal = _prev_F + extendGap;
            short hfVal = _prev_H + startGap;

            short valeShfl = sg.shuffle(_prev_E, laneId-1);
            short valheShfl = sg.shuffle(_prev_H, laneId-1);

            short eVal=0, heVal = 0;

            if(diag >= maxSize) // when the previous thread has phased out, get value from shmem
            {
                eVal = local_spill_prev_E[thread_Id - 1] + extendGap;
                heVal = local_spill_prev_H[thread_Id - 1]+ startGap;
            }
            else
            {
                eVal =((warpId !=0 && laneId == 0)?sh_prev_E[warpId-1]: valeShfl) + extendGap;
                heVal =((warpId !=0 && laneId == 0)?sh_prev_H[warpId-1]:valheShfl) + startGap;
            }

            if(warpId == 0 && laneId == 0) // make sure that values for lane 0 in warp 0 is not undefined
            {
                eVal = 0;
                heVal = 0;
            }

            _curr_F = (fVal > hfVal) ? fVal : hfVal;
            _curr_E = (eVal > heVal) ? eVal : heVal;

            short testShufll = sg.shuffle(_prev_prev_H, laneId - 1);

            short final_prev_prev_H = 0;

            if(diag >= maxSize)
            {
                final_prev_prev_H = local_spill_prev_prev_H[thread_Id - 1];
            }
            else
            {
                final_prev_prev_H =(warpId !=0 && laneId == 0)?sh_prev_prev_H[warpId-1]:testShufll;
            }

            if(warpId == 0 && laneId == 0) final_prev_prev_H = 0;

            char to_comp = reverse == true ? longer_seq[maxSize -i] : longer_seq[i - 1];
            short mat_index_q = sh_aa_encoding[(int)to_comp]; //encoding_matrix
            short mat_index_r = sh_aa_encoding[(int)myColumnChar];

            short add_score = sh_aa_scoring[mat_index_q*24 + mat_index_r]; // doesnt really matter in what order these indices are used, since the scoring table is symmetrical

            short diag_score = final_prev_prev_H + add_score;

            _curr_H = findMaxFour(diag_score, _curr_F, _curr_E, 0);

            int _i = reverse == true ? maxSize - i : i;
            int _j = reverse == true ? minSize - thread_Id - 1: thread_Id + 1;

            thread_max_i = (thread_max >= _curr_H) ? thread_max_i : _i;
            thread_max_j = (thread_max >= _curr_H) ? thread_max_j : _j;
            thread_max   = (thread_max >= _curr_H) ? thread_max : _curr_H;
            i++;
        }
        else
        {
            // we need these dummy shuffle operations for NVIDIA GPUs
            short valeShfl = sg.shuffle(_prev_E, laneId);   //__shfl_sync(mask, _prev_E, laneId- 1, 32);
            short valheShfl =  sg.shuffle(_prev_H, laneId); //__shfl_sync(mask, _prev_H, laneId - 1, 32);
            short testShufll = sg.shuffle(_prev_prev_H, laneId); //__shfl_sync(mask, _prev_prev_H, laneId - 1, 32);
        }

        BARRIER(gp); // why do I need this? commenting it out breaks it
    }

    thread_max = blockShuffleReduce_with_index(thread_max, thread_max_i, thread_max_j, reverse, item, locTots, locInds, locInds2);  // thread 0 will have the correct values

    if(reverse == true)
    {
        if(thread_Id == 0){
            if(lengthSeqA < lengthSeqB){
                seqB_align_begin[block_Id] = (thread_max_i);
                seqA_align_begin[block_Id] = (thread_max_j);
            }
            else{
                seqA_align_begin[block_Id] = (thread_max_i);
                seqB_align_begin[block_Id] = (thread_max_j);
            }
        }
    }
    else
    {  
        if(thread_Id == 0)
        {
            if(lengthSeqA < lengthSeqB)
            {
                seqB_align_end[block_Id] = thread_max_i;
                seqA_align_end[block_Id] = thread_max_j;
                top_scores[block_Id] = thread_max;
            }
            else
            {
                seqA_align_end[block_Id] = thread_max_i;
                seqB_align_end[block_Id] = thread_max_j;
                top_scores[block_Id] = thread_max;
            }
        }
    }
}

// ------------------------------------------------------------------------------------ //
