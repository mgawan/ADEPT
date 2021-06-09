# adept_revamp
revamping adept from scratch to make more usable in library form

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
