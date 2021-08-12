PLATFORM=amd
ifeq	($(HIP_PLATFORM),$(PLATFORM))
	FLAGS+=-DPLATFORM_AMD=true -O3
else
	FLAGS+=-arch=compute_70 -O3
endif
adept_test_dna: ./examples/multi_gpu_dna/dna_example.cpp adept/*
	hipcc $(FLAGS) ./examples/multi_gpu_dna/dna_example.cpp ./adept/driver.* ./adept/kernel.* -o adept_test_dna

adept_test_protein: ./examples/multi_gpu_protein/protein_example.cpp adept/*
	hipcc $(FLAGS) ./examples/multi_gpu_protein/protein_example.cpp ./adept/driver.* ./adept/kernel.* -o adept_test_protein
