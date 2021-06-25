adept_test: main.cpp adept/*
	hipcc -arch=compute_70 main.cpp ./adept/driver.* ./adept/kernel.* -o adept_test



