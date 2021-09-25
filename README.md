
# ADEPT
ADEPT is a GPU accelerated sequence alignment library for short DNA reads and protein sequences. It provides API calls for asychronously performing sequence alignments on GPUs while keeping track of progress on GPUs so that work can be performed on CPUs. Different capabilities of ADEPT API are explored using examples in the `examples` folder.

## Dependencies
ADEPT can be built using CUDA 9.0 or later and a version of GCC compatible with the CUDA version that is used. 

## Building
To build ADEPT library along with the C++ and Python examples, first move into the top level directory of the repo and follow the below steps:

### To build:
```bash
mkdir build
cd build
cmake -DADEPT_USE_PYTHON=ON ../
make
```
### To run
To run the installed examples (examples are also used as test):
```bash
make test
```
If you do not need python module of ADEPT, simply use:
```bash
mkdir build
cd build
cmake ../
make
```
and follow the above steps to run the tests.

## Usage
ADEPT provides Smith Waterman sequence alignment support for Protein and DNA sequences. The library provides options of using multiple GPUs and asynchronous support for performing CPU work while GPUs are busy with ADEPT alignments. To understand different use cases please refer to the samples contained in Examples folder.

### Contact
If you need help modifying the library to match your specific use-case or for other issues and bug reports please open an issue or reach out at mgawan@lbl.gov

### Citation
*Awan, M.G., Deslippe, J., Buluc, A. et al. ADEPT: a domain independent sequence alignment strategy for gpu architectures. BMC Bioinformatics 21, 406 (2020). https://doi.org/10.1186/s12859-020-03720-1*


### Credits
1. [Muaaz Awan](https://www.nersc.gov/about/nersc-staff/application-performance/muaaz-awan/) [![Twitter](https://flat.badgen.net/twitter/follow/MuaazGul?icon=twitter)](https://twitter.com/MuaazGul)     
2. [Muhammad Haseeb](https://sites.google.com/fiu.edu/muhammadhaseeb) [![Twitter](https://flat.badgen.net/twitter/follow/iHaseebM?icon=twitter)](https://twitter.com/iHaseebM)      

### License

ADEPT: a domain independent sequence alignment strategy for GPU architectures Copyright (c) 2019, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.

NOTICE. This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights. As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit other to do so.
