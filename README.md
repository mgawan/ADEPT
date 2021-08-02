[![GitHub forks](https://img.shields.io/github/forks/mgawan/adept_revamp.svg?style=social&label=Fork&maxAge=2592000)](https://GitHub.com/mgawan/adept_revamp/network/) [![GitHub stars](https://img.shields.io/github/stars/mgawan/adept_revamp.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/mgawan/adept_revamp/stargazers/) [![GitHub contributors](https://img.shields.io/github/contributors/mgawan/adept_revamp.svg)](https://GitHub.com/mgawan/adept_revamp/graphs/contributors/) [![GitHub issues](https://img.shields.io/github/issues/mgawan/adept_revamp.svg)](https://GitHub.com/mgawan/adept_revamp/issues/) [![Github all releases](https://img.shields.io/github/downloads/mgawan/adept_revamp/total.svg)](https://GitHub.com/mgawan/adept_revamp/releases/)

# ADEPT
ADEPT is a GPU accelerated sequence alignment library for short DNA reads and protein sequences. It provides API calls for asychronously performing sequence alignments on GPUs while keeping track of progress on GPUs so that work can be performed on CPUs. Different capabilities of ADEPT API are explored using examples in the `examples` folder.

## Dependencies
ADEPT can built using CUDA 9.0 or later and a version of GCC compatible with the CUDA version that is used. 

## Examples
To build the examples cd into the directory containing the <example>.cpp file and use below instructions to build and run:

To build:
```bash
mkdir build
cd build
cmake ../
make
```
To run:
```bash
./<example> ../../../test-data/dna-reference.fasta ../../../test-data/dna-query.fasta ./results
```

### Contact
If you need help modifying the library to match your specific use-case or for other issues and bug reports please open an issue or reach out at mgawan@lbl.gov

### Citation
*Awan, M.G., Deslippe, J., Buluc, A. et al. ADEPT: a domain independent sequence alignment strategy for gpu architectures. BMC Bioinformatics 21, 406 (2020). https://doi.org/10.1186/s12859-020-03720-1*


### Credits
1. [Muaaz Awan](https://www.nersc.gov/about/nersc-staff/application-performance/muaaz-awan/) [![Twitter](https://flat.badgen.net/twitter/follow/MuaazGul?icon=twitter)](https://twitter.com/MuaazGul)     
2. [Muhammad Haseeb](https://sites.google.com/fiu.edu/muhammadhaseeb) [![Twitter](https://flat.badgen.net/twitter/follow/iHaseebM?icon=twitter)](https://twitter.com/iHaseebM)      

### License
Please read the license [here](./LICENSE).