[![GitHub forks](https://img.shields.io/github/forks/mgawan/adept_revamp.svg?style=social&label=Fork&maxAge=2592000)](https://GitHub.com/mgawan/adept_revamp/network/) [![GitHub stars](https://img.shields.io/github/stars/mgawan/adept_revamp.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/mgawan/adept_revamp/stargazers/) [![GitHub contributors](https://img.shields.io/github/contributors/mgawan/adept_revamp.svg)](https://GitHub.com/mgawan/adept_revamp/graphs/contributors/) [![GitHub issues](https://img.shields.io/github/issues/mgawan/adept_revamp.svg)](https://GitHub.com/mgawan/adept_revamp/issues/) [![Github all releases](https://img.shields.io/github/downloads/mgawan/adept_revamp/total.svg)](https://GitHub.com/mgawan/adept_revamp/releases/)

# ADEPT SYCL
ADEPT is a GPU accelerated sequence alignment library for short DNA reads and protein sequences. This repository contains a [SYCL 1.2.1](https://www.khronos.org/files/sycl/sycl-121-reference-guide.pdf) + [SYCL 2020 Specifications](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html) based implementation of the ADEPT DPC++ code. Python bindings (PyADEPT package) are also compiled using [PyBind11](https://pybind11.readthedocs.io/en/stable/).    

###  Highlights
- SYCL + DPC++ ADEPT implementation tested on: Intel(R) UHD Graphics P630 & NVIDIA Tesla V100-SXM2-16GB     

- PyADEPT package binding all ADEPT C++ features in the Python domain.    


## Setup

We will assume that the ADEPT repository is located at: `$ADEPT_DIR`.

### Intel DevCloud
- SSH into the login nodes on Intel's oneAPI devcloud.     
- Allocate a GPU node by running: `qsub -l nodes=1:gpu:ppn=2 -I`    

### NERSC Cori
- SSH into the login nodes on NERSC Cori.       
- Load the following modules on Cori: 

```bash
module purge
module load esslurm
module use /global/common/software/m1759/llvm/modulefiles
module load llvm/nightly/20210706
module load cgpu
module unload cuda
module load cuda/10.1.243
```


## Build & Install
- Navigate to: `cd $ADEPT_DIR; mkdir build ; cd build`.
- Use CMake to generate the Makefiles using the following command: `cmake $ADEPT_DIR -DCMAKE_CXX_COMPILER=<path to: dpcpp|clang++> -DADEPT_GPU=<NVIDIA|INTEL> (default: NVIDIA) -DADEPT_INSTR=<ON|OFF> (default:ON) -DADEPT_USE_PYTHON=<ON|OFF> (default:OFF) -DBUILD_EXAMPLES=<ON|OFF> (default: ON) -DCMAKE_INSTALL_PREFIX=$ADEPT_DIR/install [OTHER CMAKE OPTIONS]`    
- Make (and) install the ADEPT by running the following command: `make install -j`     


## Examples

### ADEPT Examples
- Build examples: `cd $ADEPT_DIR/build; cmake .. -DBUILD_EXAMPLES=ON ; make install -j`      
- Run the ADEPT examples built in: `$ADEPT_DIR/build/examples`     
- Cori: `srun --partition=gpu -C gpu -G 1 -t 00:10:00 $ADEPT_DIR/build/examples/<example_name>/<example_name> /path/to/reference_seqs /path/to/query/seqs /path/to/output.tsv`    
- DevCloud: `./$ADEPT_DIR/build/examples/<example_name>/<example_name> /path/to/reference_seqs /path/to/query/seqs /path/to/output.tsv`     

### PyADEPT Examples
- Build examples and Python support: `cd $ADEPT_DIR/build; cmake .. -DBUILD_EXAMPLES=ON -ADEPT_USE_PYTHON=ON; make install -j`      
- Run the PyADEPT examples built in: `$ADEPT_DIR/build/examples/py_examples`     
- Cori: `srun --partition=gpu -C gpu -G 1 -t 00:10:00 python $ADEPT_DIR/build/examples/py_examples/<example_name> -r /path/to/reference_seqs -q /path/to/query/seqs -o /path/to/output.tsv`    
- DevCloud: `./$ADEPT_DIR/build/examples/py_examples/<example_name> -r /path/to/reference_seqs -q /path/to/query/seqs -o /path/to/output.tsv`     


## Testing
- Navigate to: `cd $ADEPT_DIR/build`  
- If built with `ADEPT_USE_PYTHON=ON`, add PyADEPT to PYTHONPATH: `export PYTHONPATH=$ADEPT_DIR/build:$PYTHONPATH`   
- Cori: `srun --partition=gpu -C gpu -G 1 -t 00:10:00 ctest`
- DevCloud: `ctest`

```bash 
$ ctest

Test project /global/homes/m/mhaseeb/repos/mhaseeb/adept_revamp/build
      Start  1: simple_dna
 1/10 Test  #1: simple_dna .......................   Passed    1.05 sec
      Start  2: async_dna
 2/10 Test  #2: async_dna ........................   Passed    0.91 sec
      Start  3: async_protein
 3/10 Test  #3: async_protein ....................   Passed    1.04 sec
      Start  4: multiGPU_dna
 4/10 Test  #4: multiGPU_dna .....................   Passed    0.98 sec
      Start  5: multiGPU_aa
 5/10 Test  #5: multiGPU_aa ......................   Passed    1.03 sec
      Start  6: py_simple_dna
 6/10 Test  #6: py_simple_dna ....................   Passed    1.38 sec
      Start  7: py_async_dna
 7/10 Test  #7: py_async_dna .....................   Passed    1.43 sec
      Start  8: py_multiGPU_dna
 8/10 Test  #8: py_multiGPU_dna ..................   Passed    1.63 sec
      Start  9: py_async_protein
 9/10 Test  #9: py_async_protein .................   Passed    1.53 sec
      Start 10: py_multiGPU_aa
10/10 Test #10: py_multiGPU_aa ...................   Passed    1.74 sec

100% tests passed, 0 tests failed out of 10

Total Test time (real) =  12.73 sec

```

### PyADEPT Unittests

- Build ADEPT with Python: `cd $ADEPT_DIR/build; cmake .. -DADEPT_USE_PYTHON=ON ; make install -j`
- Add PyADEPT to PYTHONPATH: `export PYTHONPATH=$ADEPT_DIR/build:$PYTHONPATH`    
- Navigate to: `cd $ADEPT_DIR/pyadept`    
- Cori: `srun --partition=gpu -C gpu -G 1 -t 00:10:00 python -m unittest -v`    
- DevCloud: `python -m unittest -v`     
- Read more about PyADEPT unittests [here](./pyadept/test/README.md#pyADEPT-unit-testing).

## Roofline Analysis

### NERSC Cori (NVIDIA V100)
- Please follow the instructions and steps documented [here](https://github.com/mhaseeb123/Instruction_roofline_scripts/#instruction-roofline-for-adept) to obtain the instruction roofline. Note that the conventional roofline (FLOP/byte) is not applicable to ADEPT as it performs all `integer` operations.

### Intel DevCloud (Intel UHD Graphics P630)
- Navigate to `cd $ADEPT_DIR/build`
- Run on DevCloud: `./roofline_intel`
- Copy the `roof_data_*` folder to local computer.
- Open the Intel(R) Advisor to view, analyze the collected metrics and the roofline.

## Limitations (Known Issues)
- Maximum query sequence length on Intel UHD Graphics P630 GPU must be `<=256`.
- Please change the constant variable: `MAX_QUERY_LEN = 256` in all ADEPT and PyADEPT examples before running them on the Intel DevCloud.

## Planned Future Updates
- Backtracking algorithm for Smith-Waterman.    
- Needleman-Wunsch alignment algorithm.    
- Support for long-sequence alignments: `len(query_seq) > 1024` on NVIDIA GPUs and `len(query_seq) > 256` on Intel GPUs.     
- The driver class may be revamped into `C++-17` style. For instance, there is no need to explicitly pass the member variables as arguments in the member functions of the driver class.   
- Memory model may need to switch to the buffer/accessor model to get pinned memory on host & avoid explicit memory management.    

## Credits
1. [Muaaz Awan](https://www.nersc.gov/about/nersc-staff/application-performance/muaaz-awan/) [![Twitter](https://flat.badgen.net/twitter/follow/MuaazGul?icon=twitter)](https://twitter.com/MuaazGul)     
2. [Muhammad Haseeb](https://sites.google.com/fiu.edu/muhammadhaseeb) [![Twitter](https://flat.badgen.net/twitter/follow/iHaseebM?icon=twitter)](https://twitter.com/iHaseebM)      

## License
Please read the license [here](./LICENSE).

## Citation
*Awan, M.G., Deslippe, J., Buluc, A. et al. ADEPT: a domain independent sequence alignment strategy for gpu architectures. BMC Bioinformatics 21, 406 (2020). https://doi.org/10.1186/s12859-020-03720-1*

## Contact
If you need help modifying the library to match your specific use-case or for other issues and bug reports please open an issue or reach out at: mgawan@lbl.gov


