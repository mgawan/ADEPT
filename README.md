# ADEPT
ADEPT is a GPU accelerated sequence alignment library for short DNA reads and protein sequences. It provides API calls for asychronously performing sequence alignments on GPUs while keeping track of progress on GPUs so that work can be performed on CPUs. Different capabilities of ADEPT API are explored using examples in the `example folder`.

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
srun ./<example> ../../../test-data/dna-reference.fasta ../../../test-data/dna-query.fasta ./results
```
### Contact
If you need help modifying the library to match your specific use-case or for other issues and bug reports please open an issue or reach out at mgawan@lbl.gov


### Citation
*Awan, M.G., Deslippe, J., Buluc, A. et al. ADEPT: a domain independent sequence alignment strategy for gpu architectures. BMC Bioinformatics 21, 406 (2020). https://doi.org/10.1186/s12859-020-03720-1*

### License:
        
**GPU accelerated Smith-Waterman for performing batch alignments (GPU-BSW) Copyright (c) 2019, The
Regents of the University of California, through Lawrence Berkeley National
Laboratory (subject to receipt of any required approvals from the U.S.
Dept. of Energy).  All rights reserved.**

**If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.**

**NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative
works, and perform publicly and display publicly, and to permit other to do
so.**