# ADEPT SYCL
[SYCL* 1.2.1](https://www.khronos.org/files/sycl/sycl-121-reference-guide.pdf) + [SYCL 2020 Specifications](https://www.khronos.org/registry/SYCL/specs/sycl-2020/pdf/sycl-2020.pdf) based implementation of the ADEPT software code.

## What's New?
Committing the working ADEPT SYCL code tested on ***Intel(R) UHD Graphics P630***.

- Tested with the sample data only (may need more testing for correctness).
- To CMake with DPC++ compiler use: `cmake .. -DCMAKE_CXX_COMPILER=dpcpp [OTHER OPTIONS]`
- Currently only tested in `-DCMAKE_BUILD_TYPE=Debug`. Will do Release testing after a few optimizations.
- Changed the name of kernel namespace to Akernel to avoid the mix with the class kernel defined in `<cl/sycl.hpp>`

## What still needs to be done?
- The max block size needs to be looked into to handle queries >256
- The driver class needs massive revamping into at least `C++-17` (required by SYCL) style. For instance, there is no need to explicitly pass the member variables as arguments in the member functions of the driver class. This creates a big overhead.
- We may need to switch to the buffer/accessor model to avoid the need to explicitly manage memory on both the host and device?
- There are almost 20 arguments being passed to the functions in the kernel namespace which is absurd.
- The `subgroup` (warp) size may need to be set to the optimum value by the compiler depending on the device. We need to remove the hardcoded value 32 and all the operations in the kernel that assume it to be `32`.
