PLATFORM=amd
ifeq	($(HIP_PLATFORM),$(PLATFORM))
	FLAGS+=-DPLATFORM_AMD=true
else
	FLAGS+=-arch=compute_70
endif
adept_test: main.cpp adept/*
	hipcc $(FLAGS) main.cpp ./adept/driver.* ./adept/kernel.* -o adept_test



