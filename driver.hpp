namespace ADEPT{
	enum ALG_TYPE{SW, NW};
	enum CIGAR{NO, YES};
	enum SEQ_TYPE{DNA, AA};
        
	struct aln_results{
		short ref_begin, ref_end, query_begin, query_end;
		aln_results();
	};

	struct adept_stream;
	class driver{
		private:
			short match_score, mismatch_score, gap_start, gap_extend;
			int gpu_id;
			ALG_TYPE algorithm;
			SEQ_TYPE sequence;
			CIGAR cigar_avail;
			aln_results *results;
			adept_stream curr_stream;

			unsigned max_ref_size, max_que_size;
        		char *ref_cstr, *que_cstr;
        		unsigned total_alignments;
        		unsigned *offset_ref, *offset_que;
        		unsigned total_length_ref, total_length_que;
			short *ref_start_gpu, *ref_end_gpu, *query_start_gpu, *query_end_gpu, *scores_gpu;
        		unsigned* offset_ref_gpu, *offset_query_gpu;
        		char *ref_cstr_d, *que_cstr_d;

			void allocate_gpu_mem(unsigned max_alignments);
			void dealloc_gpu_mem();
			void initialize_alignments();
			void free_alignments();
			void mem_copy_htd(unsigned* offset_ref_gpu, unsigned* offset_query_gpu, unsigned* offsetA_h, 
					unsigned* offsetB_h, char* strA, char* strA_d, char* strB, char* strB_d, 
					unsigned totalLengthA, unsigned totalLengthB, int sequences_per_stream);

		public:
			void initialize(short scores[], ALG_TYPE _algorithm, SEQ_TYPE _sequence, CIGAR _cigar_avail, int _gpu_id, 
				      std::vector<std::string> ref_seqs, std::vector<std::string>query_seqs);// each adept_dna object will have a unique cuda stream
			void kernel_launch();
			void mem_cpy_dth();
			aln_results get_results();
			bool kernel_done();
			void gpu_cleanup();
	};
}
