# ADEPT for HIP
This branch is a work in progress, this contained HIP port of ADEPT library. Currently full funtionality is available for AMD devices. Follow the instructions below for building and running for AMD devices.

## To Build
To build and run tests make sure that rocm is available on your system and env variable `HIP_PLATFORM` is set to `amd ` for AMD devices then follow:

```bash
mkdir build
cd build
cmake ../
make
```
To run all the tests/examples:
```bash
make test
```
## Contact
contact at mgawan@lbl.gov for further support
