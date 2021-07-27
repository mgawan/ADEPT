#include <vector>
#include <string>
#include <array>


namespace ADEPT{

namespace options
{
	enum ALG_TYPE{SW, NW};
	enum CIGAR{NO, YES};
	enum SEQ_TYPE{DNA, AA};
}

	struct aln_results{
		short *ref_begin, *ref_end, *query_begin, *query_end, *top_scores;
		int size;
		void free_results();
	};

	struct gap_scores{
		short open;
		short extend;
		gap_scores(){
			open = 0;
			extend = 0;
		}
		gap_scores(short open_, short extend_){
			open = open_;
			extend = extend_;
		}

		void set_scores(short open_, short extend_)
		{
		    open = open_;
		    extend = extend_;
		}

		std::array<short, 2> get_scores()
		{
		    return {open, extend};
		}
	};

	struct all_alns{
		std::vector<aln_results> results;
		int per_gpu;
		int left_over;
		int gpus;
		all_alns(int count)
		{
			results.reserve(count);		
			// insert dummy aln_results here
			for (int i = 0; i < count; i++)
				results.push_back(aln_results());
			per_gpu = 0;
			left_over = 0;
			gpus = count;
		}
	};

	struct adept_stream;
	class driver{
		private:
			short match_score, mismatch_score, gap_start, gap_extend;
			int gpu_id;
			options::ALG_TYPE algorithm;
			options::SEQ_TYPE sequence;
			options::CIGAR cigar_avail;
			adept_stream *curr_stream = nullptr;

			unsigned max_ref_size, max_que_size;
			char *ref_cstr, *que_cstr;
			unsigned total_alignments, batch_size;
			unsigned *offset_ref, *offset_que;
			unsigned total_length_ref, total_length_que;
			short *ref_start_gpu, *ref_end_gpu, *query_start_gpu, *query_end_gpu, *scores_gpu;
			unsigned* offset_ref_gpu, *offset_query_gpu;
			char *ref_cstr_d, *que_cstr_d;
			aln_results results;
			short *d_encoding_matrix, *d_scoring_matrix, *scoring_matrix_cpu;
			short *encoding_matrix;

			void allocate_gpu_mem();
			void dealloc_gpu_mem();
			void initialize_alignments();
			void mem_cpy_htd(unsigned* offset_ref_gpu, unsigned* offset_query_gpu, unsigned* offsetA_h, unsigned* offsetB_h, char* strA, char* strA_d, char* strB, char* strB_d, unsigned totalLengthA, unsigned totalLengthB);
			void mem_copies_dth(short* ref_start_gpu, short* alAbeg, short* query_start_gpu,short* alBbeg, short* scores_gpu , short* top_scores_cpu, int res_offset = 0);
			void mem_copies_dth_mid(short* ref_end_gpu, short* alAend, short* query_end_gpu, short* alBend, int res_offset = 0);
			int get_new_min_length(short* alAend, short* alBend, int blocksLaunched);

		public:
			// default constructor
			driver() = default;
			void initialize(std::vector<short> &scores, gap_scores g_scores, options::ALG_TYPE _algorithm, options::SEQ_TYPE _sequence, options::CIGAR _cigar_avail, int _max_ref_size, int _max_query_size, int _batch_size, int _tot_alns, int _gpu_id = 0);// each adept_dna object will have a unique cuda stream

			void kernel_launch(std::vector<std::string> &ref_seqs, std::vector<std::string> &query_seqs, int res_offset = 0);
			void mem_cpy_dth(int offset=0);
			aln_results get_alignments();
			bool kernel_done();
			bool dth_done();
			void kernel_synch();
			void dth_synch();
			void cleanup();
			void set_gap_scores(short _gap_open, short _gap_extend);
	};

	aln_results thread_launch(std::vector<std::string> &ref_vec, std::vector<std::string> &que_vec, ADEPT::options::ALG_TYPE algorithm, ADEPT::options::SEQ_TYPE sequence, ADEPT::options::CIGAR cigar_avail, int max_ref_size, int max_que_size, int batch_size, int dev_id, std::vector<short> &scores, gap_scores gaps);

	all_alns multi_gpu(std::vector<std::string> &ref_sequences, std::vector<std::string> &que_sequences, ADEPT::options::ALG_TYPE algorithm, ADEPT::options::SEQ_TYPE sequence, ADEPT::options::CIGAR cigar_avail, int max_ref_size, int max_que_size, std::vector<short> &scores, gap_scores gaps, int batch_size_ = -1);
	size_t get_batch_size(int gpu_id, int max_q_size, int max_r_size, int per_gpu_mem = 100);
}
