# OpenCL Metal Stdlib

> Note: The first release will come soon; please don't use the header in production code yet.

A header for accessing functions from the Metal Standard Library inside OpenCL code.

The Apple GPU was designed to have slower communication between SIMD groups in a threadgroup, but faster communication within a single SIMD. This tradeoff improves power efficiency and requires modifying algorithms to utilize SIMD-scoped operations. In OpenCL, Apple does not expose such operations to the shading language. These would be possible under OpenCL 2.0 or the `cl_khr_subgroups` extension, but Apple has deprecated OpenCL. The instructions are only exposed to Metal. This restriction fundamentally makes some OpenCL code bases slower than if written in Metal, including molecular dynamics code. It also makes matrix multiplication drop to 1/3 of maximum performance, becoming no faster than the CPU's AMX. Third-party code cannot harness the Apple GPU for AI/ML without using the `simdgroup_matrix` instruction set.

This repository is a solution to the problem. In Apple's M1 OpenCL driver, the `__asm` keyword lowers down to AIR (Apple Intermediate Representation). Certain OpenCL Standard Library functions are implemented directly through bindings to AIR. Any function, including SIMD-scoped operations, can be exposed this way. As long as it is a callable AIR function. This imposes some unusual constraints and prevents full conformance to the `cl_khr_subgroups` extension. Client code can still harness SIMD-scoped operations to reach maximum performance, just with a minor restriction to how threads are dispatched.

### Why you don't need to use Metal directly

- OpenCL only permits 256 threads/threadgroup instead of 1024. That's fine, because anything above 256 threads significantly deterioriates performance for memory-heavy workloads.
- OpenCL does not support `half` precision. That's fine, because the M1 GPU architecture doesn't either. The M1 and A15 made FP32 just as fast as FP16; half-precision only remains to decrease register pressure and register bandwidth.
- OpenCL doesn't allow access to the `MTLCommandBuffer`. That's fine, because it internally bunches up `clEnqueueXXX` calls into command buffers. It does this more optimally than many manual solutions.
- OpenCL is especially fast at AI/ML applications that [dispatch several small operations](https://github.com/philipturner/metal-experiment-1). It should have much better sequential throughput than PyTorch.
- [SYCL](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#introduction) will hopefully have a backend for Metal in the future. That means you can use another standardized Khronos API soon. If you're planning to invest time and money migrating OpenCL applications to Metal, the port may become obsolete soon. Note that this is speculative, and not professional advice.

### What you need to watch out for

- OpenCL does not run on iOS. But SYCL will.
- The OpenCL Metal Stdlib header may take several seconds to compile, the first time. This could cause a major bottleneck if multiple independent shader programs import it. To work around this, fuse several kernels into a single OpenCL program.
- Do not use this header on x86 macOS OpenCL backends. Use a conditional compilation macro to only do SIMD reductions on M1. Other vendors have better communication between SIMDs in a threadgroup, so this isn't as much of a concern as with M1.
- OpenCL provides no direct way to query a thread's lane ID within a subgroup. This is a quirk of how the AIR workaround is implemented. Therefore, the subgroup functions aren't fully OpenCL 2.0 compliant. To work around this, always make threadgroup sizes a multiple of 32, then take `get_local_id(0) % 32`.
- For the same reasons as the previous note, clustered subgroup operations fail unless cluster size is 4 or 32. The remaining sizes could technically be implemented on Apple 8, which has hardware support ("simd shuffle and fill"). However, it may not be possible to determine the GPU architecture inside OpenCL code. The Metal Standard Library functions for "simd shuffle and fill" are exposed, so just use those if needed.
- Metal only provides SIMD-scoped operations for 8, 16, and 32-bit types. Therefore, this header cannot expose functions for 64-bit integers. Emulating 64-bit operations is a non-trivial task, so I instead opted to violate Khronos conformance.
- OpenCL events will trap any commands in their vicinity into the same `MTLCommandBuffer`. After making any `clEnqueueXXX` call that signals a `cl_event`, flush the queue. Do the same immediately before waiting on any `cl_event`.
- OpenCL queue profiling is not Khronos conformant. It reports time spans as 3/125 times their actual value, because it treats `mach_timebase` ticks like nanoseconds.
- OpenCL seems to not support pre-compiled binary functions - I could not get it to work. Use Metal if startup overhead is mission critical (e.g. real-time rendering). Note that Apple's JIT shader compiler harnesses the system Metal Shader Cache, and is quite fast.
- OpenCL commands probably cannot be captured in Metal Frame Capture. I'm not 100% sure; I just tested OpenMM which is a massive code base. You can still test execution time of each kernel. Just use the OpenCL queue profiling API. For lower-level profiling within the kernel, read [metal-benchmarks](https://github.com/philipturner/metal-benchmarks), use your best judgment, and use trial and error.

## Features

The biggest motivating factor behind this header-only library was inaccessibility of SIMD-scoped operations. However, this library has exposed other Metal functionality, such as `ctz`.

OpenCL (from extension specification):
- cl_khr_subgroups - without the [work-item functions](https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_Ext.html#cl_khr_subgroups-additions-to-section-6.13.1-work-item-functions), without the overload of `sub_group_barrier` that takes a memory scope argument
- cl_khr_subgroup_extended_types<sup>1</sup> - without 64-bit types
- cl_khr_subgroup_non_uniform_vote - without `sub_group_non_uniform_all_equal`
- cl_khr_subgroup_ballot - only `sub_group_non_uniform_broadcast`, `sub_group_broadcast_first`, and `sub_group_ballot`
- cl_khr_subgroup_non_uniform_arithmetic - without the prefix forms of min, max, bitwise, and logical reductions
- cl_khr_subgroup_shuffle
- cl_khr_subgroup_shuffle_relative
- cl_khr_subgroup_clustered_reduce - with compile-time failure for clusters of 1, 2, 8, or 16
- `ctz` from OpenCL 2.0

Metal (from feature set tables):
- Quad-scoped permute operations
- SIMD-scoped permute operations
- SIMD-scoped reduction operations
- SIMD-scoped matrix multiply operations - not implemented yet
- SIMD shift and fill - not implemented yet
- SIMD-group async copy operations - not implemented yet

<sup>1</sup>When `cl_khr_subgroup_extended_types` is enabled, the compiler spends significant time processing function definitions for each type. There are 36 possible types with the extension enabled, and 3 without. The compiler often spends several seconds parsing these extra types. None of them are hardware-accelerated on M1, except `char4` and `short2` shuffle, which can be masked as `int` shuffle. For the sake of out of the box performance, the extension and associated Metal Stdlib bindings are not exposed by default. To expose them, set the following compiler flag:

```bash
# Method 1: pass argument to OpenCL compiler
-DOPENCL_USE_SUBGROUP_EXTENDED_TYPES=1
```

```c
// Method 2: embed into source code
#define OPENCL_USE_SUBGROUP_EXTENDED_TYPES 1
#include "metal_stdlib.h"
#undef OPENCL_USE_SUBGROUP_EXTENDED_TYPES
```

## Usage

To use the header, either include its file path in shader code, or paste the header's contents directly. The header currently only works with Apple's M1 OpenCL backend, so check for the Apple architecture before including it. The example below uses pre-defined macros for each distinct vendor. Your source code may use a different method to activate this code path, as long as it only activates on Apple silicon GPUs.

```opencl
// OpenCL code
#if __VENDOR_APPLE__
#include "metal_stdlib.h"
#endif

__kernel void vector_add(__global const int *A, __global const int *B, __global int *C) {
  
     // Get the index of the current element to be processed
     int i = get_global_id(0);
  
     // Do the operation
#if __VENDOR_APPLE__
     // You can also use the function defined by the OpenCL subgroups extension.
     // This would make the line of code runnable on Windows.
     C[i] = A[i] + B[i] + simd_prefix_inclusive_sum(i);
#else
     __local int scratch_memory[__VENDOR_SIMD_WIDTH__];
     // ... (perform parallel prefix sum, the slow way)
     C[i] = A[i] + B[i] + scratch_memory[get_local_id(0) % __VENDOR_SIMD_WIDTH__];
#endif
}
```

To test the header, you can run the test script either from the command line or from Xcode. To run from the command-line, first download `Tests.swift` from this repository. Either place `metal_stdlib.h` in the same directory as the script, or specify an alternative parent folder (see usage below). Right-click the folder containing the script in Finder. Click <b>New Terminal at Folder</b> and enter the following command.

```bash
swift Tests.swift --header-directory . --use-subgroup-extended-types
# Usage: swift Tests.swift \
#   [--header-directory <path>] \
#   [--use-subgroup-extended-types]
```

To run from Xcode, create a new Xcode project with the template <b>macOS > Command Line Tool</b>. Replace the Swift file with `Tests.swift` from this repository. Next, copy the header file into the project folder. Add the header to the Xcode target using the following steps:
- Navigate to `Project Settings` -> `TARGETS` -> This Xcode Project -> `Build Phases` -> `Copy Files`
- `Destination`: change to `Resources`
- `Subpath`: change to nothing (erase the contents of the text box)
- De-select `copy only when installing`
- Click the `+` button, locate `metal_stdlib.h`, and add it to the build phase.

Finally, click <b>Menu Bar > Product > Run</b> to execute the test. The script will automatically skip the tests for `cl_khr_subgroup_extended_types`. To enable them, search for `usingSubgroupExtendedTypes` using `Ctrl + F`. Replace the default value of `false` with `true`.

```swift
var usingSubgroupExtendedTypes = false // before
var usingSubgroupExtendedTypes = true // after
```

## Licensing

If your project needs to depend on a specific version of this repository, check the GitHub releases. The newest release includes `metal_stdlib.h` and a separate ZIP folder containing other source files from this repository. You can obtain the MIT license from the zipped source folder. To refer to this repository, you may use the informal titles "(the) OpenCL Metal Standard Library" or "metal_stdlib.h". Note that Apple owns the trademarks "OpenCL" and "Metal".

## Related Work

https://github.com/philipturner/MoltenCL - a previous attempt to bypass restrictions on using the M1 GPU from OpenCL
