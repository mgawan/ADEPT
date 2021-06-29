# ADEPT SYCL (with multiGPU support)

[SYCL 1.2.1](https://www.khronos.org/files/sycl/sycl-121-reference-guide.pdf) + [SYCL 2020 Specifications](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html) based implementation of the ADEPT software code.    

## What's New?
Committing the working ADEPT SYCL code with multiGPU support tested on: 
1. ***Intel(R) UHD Graphics P630***       
2. ***NVIDIA Tesla V100-SXM2-16GB***       

## Build Instructions

### DevCloud
- SSH into the login nodes on Intel's oneAPI devcloud.     
- Get to a GPU node by running: `qsub -l nodes=1:gpu:ppn=2 -I`    

### Cori
- SSH into the login nodes on NERSC Cori.       
- Load the following modules on Cori: 

```bash
module use /global/cfs/cdirs/mpccc/dwdoerf/cori-gpu/modulefiles
module load dpc++
module load cgpu
module load cuda
```

### Instructions
- Use CMake to generate the Makefiles using the following command: `cmake $ADEPT_PATH -DCMAKE_CXX_COMPILER=<PATH_TO dpcpp|clang++> -DADEPT_GPU=<Intel|NVIDIA> (default: Intel) [OTHER CMAKE OPTIONS]`    
- Make (and) install the ADEPT by running the following command: `make install -j 8`   

## Test Instructions
- Navigate to: `cd $ADEPT_PATH/build`   
- Run on Cori: `srun --partition=gpu -C gpu -G 1 -t 00:10:00 ./test_adept [NUM_ITERATIONS]`    
- Run on DevCloud: `cd $ADEPT_PATH/build ; ./test_adept [NUM_ITERATIONS]`     


## Work In Progress
- Performance Analysis of the SYCL code vs the native CUDA code.
- Performance Analysis across GPUs from different vendors.
- The max block size needs to be looked into to handle `len(query)>256` on Intel GPUs.     
- The driver class may be revamped into `C++-17` style. For instance, there is no need to explicitly pass the member variables as arguments in the member functions of the driver class.   
- May need to switch to the buffer/accessor model to get pinned memory on host & avoid explicit and unnecessary memory management between the host and device.    
- Several other optimizations.     
