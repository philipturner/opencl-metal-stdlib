# OpenCL Metal Stdlib

(Very tentative) We may be able to access SIMD-group reductions through OpenCL kernels. If this turns out true, I will personally ensure OpenCL reaches fully parity with Metal, allowing us to bypass Apple's restriction on OpenCL. This would mean we can utilize `simdgroup_matrix` from OpenCL too, reaching 80% ALU utilization in matmul -> AI/ML. All of this will be made possible by one C++ header, which you insert into OpenCL kernel code.
- TODO: Outline how OpenCL driver has better scheduling overhead than naive Metal code.
- TODO: Integrate this into tinygrad, DLPrimitives?
