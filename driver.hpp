namespace ADEPT{
	enum TYPE{SW, NW};
	enum CIGAR{NO, YES};
	class adept_dna{
		public:
			short match_score, mismatch_score, gap_start, gap_extend;
			int gpu_id;
			void initialize(short);
		
		

	}
}
