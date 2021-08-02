# PyADEPT Unit testing
This folder contains unit tests for the Python ADEPT (PyADEPT) module. The unit tests have been written using the Python's [unittest](https://docs.python.org/3/library/unittest.html) framework.

## Usage

```bash
## ssh to a GPU node
$ export PYTHONPATH=$PYADEPT_PYBIND11_LIB_PATH:$PYTHONPATH
$ cd $PYADEPT_MODULE_DIR
$ srun python -m unittest -v

test_async_dna (test.test_dna.PyAdeptDNATests)
attempted relative import beyond top-level package
attempted relative import beyond top-level package

DNA async completed
--- Elapsed: 0.0844 seconds ---
DNA async ... ok
test_simple_dna (test.test_dna.PyAdeptDNATests)

DNA simple completed
--- Elapsed: 0.0768 seconds ---
DNA simple ... ok
test_multigpu_aa (test.test_multigpu.PyAdeptMultiGPUTests)
Batch Size:14958
Total Alignments:29914
Total devices:1
Alns per GPU:29914
gpu:0 has alns:0
total alignments:29914 thread:0
GPU: 0 progress = 1/2
GPU: 0 progress = 2/2

Protein MultiGPU completed
--- Elapsed: 0.151 seconds ---
Protein MultiGPU ... ok
test_multigpu_dna (test.test_multigpu.PyAdeptMultiGPUTests)
Batch Size:15001
Total Alignments:30000
Total devices:1
Alns per GPU:30000
gpu:0 has alns:0
total alignments:30000 thread:0
GPU: 0 progress = 1/2
GPU: 0 progress = 2/2

DNA MultiGPU completed
--- Elapsed: 0.1193 seconds ---
DNA MultiGPU ... ok
test_async_aa (test.test_protein.PyAdeptAATests)

Protein async completed
--- Elapsed: 0.1009 seconds ---
Protein async ... ok
test_simple_aa (test.test_protein.PyAdeptAATests)

Protein simple completed
--- Elapsed: 0.1009 seconds ---
Protein simple ... ok

----------------------------------------------------------------------
Ran 6 tests in 1.924s

OK
```