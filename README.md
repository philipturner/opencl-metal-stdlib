# OpenCL Metal Stdlib

> Note: This is a work in progress, please don't use the header in production code yet.

(Very tentative) We may be able to access SIMD-group reductions through OpenCL kernels. If this turns out true, I will ensure OpenCL reaches performance parity with Metal, allowing developers to bypass Apple's restriction on OpenCL. This would mean OpenCL can utilize `simdgroup_matrix`, jumping from 25% to 80% ALU utilization in matmul; AI/ML becomes viable. All of this will be made possible by one C header, which you insert into OpenCL kernel code. The header also implements partial conformance to `cl_khr_subgroups` and other subgroup extensions.
<!--
- TODO: Integrate this into VkFFT, tinygrad, DLPrimitives.
-->

### Why you don't need Metal

- OpenCL only permits 256 threads/threadgroup instead of 1024. That's fine, because anything above 256 threads seriously deterioriates performance.
- OpenCL does not support `half` precision. That's fine, because the M1 GPU architecture doesn't either.
- OpenCL doesn't allow access to the `MTLCommandBuffer`. That's fine, because it internally bunches up `clEnqueueXXX` calls into command buffers. And probably more optimally than you will.
- OpenCL is especially fast at AI/ML applications that [dispatch several small operations](https://github.com/philipturner/metal-experiment-1). It should have much better sequential throughput than PyTorch.
- [SYCL](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#introduction) will hopefully have a backend for Metal in the future. That means you can use another standardized Khronos API soon. If you're planning to invest time and money migrating OpenCL applications to Metal, the port may become obsolete soon. Note that this is speculative, and not professional advice.

### What you need to watch out for

- OpenCL does not run on iOS. But SYCL will.
- OpenCL provides no direct way to query a thread's lane ID within a subgroup. This is a quirk of how the AIR workaround is implemented. Therefore, the subgroup functions aren't fully OpenCL 2.0 compliant. To work around this, always make threadgroup sizes a multiple of 32, then take `get_local_id(0) % 32`.
- For the same reasons as the previous note, clustered subgroup operations fail unless cluster size is 1, 4, or 32. The remaining sizes could technically be implemented on Apple 8, which has hardware support ("simd shuffle and fill"). However, it may not be possible to determine the GPU architecture inside OpenCL code. The Metal Standard Library functions for "simd shuffle and fill" are exposed, so just use those if needed.
- OpenCL events will infect any commands in their vicinity. After making any `clEnqueueXXX` call that signals a `cl_event`, flush the queue. Do the same immediately before waiting on any `cl_event`.
- OpenCL profiling is buggy. It reports time spans as 3/125 times their actual value, because it treats `mach_timebase` ticks like nanoseconds.
- OpenCL seems to not support pre-compiled binary functions - I could not get it to work. Use Metal if startup overhead is mission critical (e.g. real-time rendering). Note that Apple's JIT shader compiler harnesses the system Metal Shader Cache, and is quite fast.
- OpenCL commands probably cannot be captured in Metal Frame Capture. I'm not 100% sure; I just tested OpenMM which is a massive code base. You can still test execution time of each kernel. Just use the OpenCL kernel profiling API. For lower-level profiling within the kernel, read [metal-benchmarks](https://github.com/philipturner/metal-benchmarks), use your best judgment, and use trial and error.

## Features

The biggest motivating factor behind this library was inaccessibility of SIMD-scoped operations. However, this library can expose unrelated Metal functionality in the future.

OpenCL (from extension specification):
- cl_khr_subgroups - without the [work-item functions](https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_Ext.html#cl_khr_subgroups-additions-to-section-6.13.1-work-item-functions)
- cl_khr_subgroup_extended_types
- cl_khr_subgroup_non_uniform_vote
- cl_khr_subgroup_ballot
- cl_khr_subgroup_non_uniform_arithmetic
- cl_khr_subgroup_shuffle
- cl_khr_subgroup_shuffle_relative
- cl_khr_subgroup_clustered_reduce - with compile-time failure for clusters of 2, 8, or 16

Metal (from feature set tables):
- Quad-scoped permute operations
- SIMD-scoped permute operations
- SIMD-scoped reduction operations
- SIMD-scoped matrix multiply operations
- SIMD shift and fill

## Usage

```opencl
// OpenCL code
#include <metal_stdlib>
// unfortunately cannot replicate "using namespace metal;" - OpenCL C is not C++

__kernel void vector_add(__global const int *A, __global const int *B, __global int *C) {
  
     // Get the index of the current element to be processed
     int i = get_global_id(0);
  
     // Do the operation
     C[i] = A[i] + B[i];
}
```

## Previous Attempts

https://github.com/philipturner/MoltenCL
