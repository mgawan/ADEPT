# PyADEPT Unit testing
This folder contains unit tests for the Python ADEPT (PyADEPT) module. The unit tests have been written using the Python's [unittest](https://docs.python.org/3/library/unittest.html) framework.

## Usage

```bash
## ssh to a GPU node
$ export PYTHONPATH=$PYADEPT_PYBIND11_LIB_PATH:$PYTHONPATH
$ cd $PYADEPT_MODULE_DIR
$ srun python -m unittest -v

test_async_dna (test.test_dna.PyAdeptDNATests)
INFO: The device: Tesla V100-SXM2-16GB is now ready!
attempted relative import beyond top-level package
attempted relative import beyond top-level package

DNA async completed
--- Elapsed: 0.1913 seconds ---
DNA async ... ok
test_simple_dna (test.test_dna.PyAdeptDNATests)
INFO: The device: Tesla V100-SXM2-16GB is now ready!

DNA simple completed
--- Elapsed: 0.1705 seconds ---
DNA simple ... ok
test_multigpu_aa (test.test_multigpu.PyAdeptMultiGPUTests)
Batch Size = 14958
Total Alignments = 29914
Total Devices = 1
Alns per GPU = 29914

Thread 0 now owns device: Tesla V100-SXM2-16GB
Local Alignments = 29914

INFO: The device: Tesla V100-SXM2-16GB is now ready!
Elapsed Time: 0.00112366s

Thread 0 progress = 1/2
Cumulative Fkernel time: 0.0897211s
Cumulative Rkernel time: 0.0318555s
Cumulative H2D time: 0.000703908s
Cumulative D2Hmid time: 5.5841e-05s
Cumulative D2H time: 6.7699e-05s

Thread 0 progress = 2/2
Cumulative Fkernel time: 0.180076s
Cumulative Rkernel time: 0.0640948s
Cumulative H2D time: 0.00128898s
Cumulative D2Hmid time: 0.000105177s
Cumulative D2H time: 0.000131843s


Protein MultiGPU completed
--- Elapsed: 0.7519 seconds ---
Protein MultiGPU ... ok
test_multigpu_dna (test.test_multigpu.PyAdeptMultiGPUTests)
Batch Size = 15001
Total Alignments = 30000
Total Devices = 1
Alns per GPU = 30000

Thread 0 now owns device: Tesla V100-SXM2-16GB
Local Alignments = 30000

INFO: The device: Tesla V100-SXM2-16GB is now ready!
Elapsed Time: 0.00107591s

Thread 0 progress = 1/2
Cumulative Fkernel time: 0.0574921s
Cumulative Rkernel time: 0.0174068s
Cumulative H2D time: 0.000725063s
Cumulative D2Hmid time: 5.7457e-05s
Cumulative D2H time: 6.4767e-05s

Thread 0 progress = 2/2
Cumulative Fkernel time: 0.11577s
Cumulative Rkernel time: 0.0351384s
Cumulative H2D time: 0.00156843s
Cumulative D2Hmid time: 0.000105221s
Cumulative D2H time: 0.000139145s


DNA MultiGPU completed
--- Elapsed: 0.667 seconds ---
DNA MultiGPU ... ok
test_async_aa (test.test_protein.PyAdeptAATests)
INFO: The device: Tesla V100-SXM2-16GB is now ready!

Protein async completed
--- Elapsed: 0.2576 seconds ---
Protein async ... ok
test_simple_aa (test.test_protein.PyAdeptAATests)
INFO: The device: Tesla V100-SXM2-16GB is now ready!

Protein simple completed
--- Elapsed: 0.2577 seconds ---
Protein simple ... ok

----------------------------------------------------------------------
Ran 6 tests in 4.875s

OK
```