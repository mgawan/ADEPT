FLAGS=-arch=compute_70
ifeq	($(HIP_PLATFORM),"amd")
	FLAG+=" -DHIP_PLATFORM=true"
endif
adept_test: main.cpp adept/*
	hipcc $(FLAGS) main.cpp ./adept/driver.* ./adept/kernel.* -o adept_test



