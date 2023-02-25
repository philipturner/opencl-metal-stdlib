# OpenCL Metal Stdlib

(Very tentative) We may be able to access SIMD-group reductions through OpenCL kernels. If this turns out true, I will personally ensure OpenCL reaches fully parity with Metal, allowing us to bypass Apple's restriction on OpenCL. This would mean we can utilize `simdgroup_matrix` from OpenCL too, reaching 80% ALU utilization in matmul -> AI/ML. All of this will be made possible by one C header, which you insert into OpenCL kernel code.
- TODO: Outline how OpenCL driver has better scheduling overhead than naive Metal code.
- TODO: Explain why Open SYCL is still the future, but OpenCL is a good near-term stopgap.
- TODO: Integrate this into VkFFT, tinygrad, DLPrimitives.

## Usage

```opencl
// OpenCL code
#include "metal_stdlib"
// cannot replicate "using namespace metal;" unfortunately - OpenCL C is not C++

__kernel void vector_add(__global const int *A, __global const int *B, __global int *C) {
  
     // Get the index of the current element to be processed
     int i = get_global_id(0);
  
     // Do the operation
     C[i] = A[i] + B[i];
}
```

## Previous Attempts

https://github.com/philipturner/MoltenCL
