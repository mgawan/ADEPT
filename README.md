# adept_revamp_hip
revamping adept from scratch to make more usable in library form

## Examples
To build test cases for protein and dna launches first load the rocm module and make sure that `HIP_PLATFORM` is set to `amd `.

To build protein test case:
```bash
make adept_test_protein

```
To run:
```bash
./adept_test_protein ./test-data/ref_50k_aa.fasta ./test-data/que_50k_aa.fasta ./results_protein
```

To build DNA test case:
```bash
make adept_test_dna

```
To run:
```bash
./adept_test_dna ./test-data/ref_50k_dna.fasta ./test-data/read_50k_dna.fasta ./results_dna
```
