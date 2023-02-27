//
//  Tests.swift
//  OpenCL Metal Stdlib
//
//  Created by Philip Turner on 2/24/23.
//

import Metal
import OpenCL

// Either specify header from command-line or package inside Xcode app bundle.
//
// TODO: Transfer these instructions to README
//
// Command-Line:
// swift Tests.swift --headers-directory .
//
// Xcode:
// Project Settings -> TARGETS -> This Xcode Project -> Build Phases
// -> Copy Files
// - Destination: change to `Resources`
// - Subpath: change do nothing
// - Delect "copy only when installing"
// - Add "metal_stdlib.h" using the `+` button

var headerURL: URL
if let headersDirectoryFlagIndex = CommandLine.arguments.firstIndex(
  of: "--headers-directory") {
  let headerDirectoryIndex = headersDirectoryFlagIndex + 1
  guard CommandLine.arguments.count > headerDirectoryIndex else {
    fatalError("Usage: swift Tests.swift [--headers-directory =<path>]")
  }
  let headersDirectoryPath = CommandLine.arguments[headerDirectoryIndex]
  let directoryURL = URL(
    filePath: headersDirectoryPath, directoryHint: .isDirectory)
  headerURL = directoryURL.appending(
    component: "metal_stdlib.h", directoryHint: .notDirectory)
} else {
  guard let _headerURL = Bundle.main.url(
    forResource: "metal_stdlib", withExtension: "h") else {
    fatalError("Could not locate header in Xcode app bundle.")
  }
  headerURL = _headerURL
}
guard let headerData = FileManager.default.contents(
  atPath: headerURL.absoluteString) else {
  fatalError("Invalid header path: \(headerURL.absoluteString)")
}
guard let headerString = String(data: headerData, encoding: .utf8) else {
  fatalError("Malformatted header: \(headerURL.absoluteString)")
}

var defaultTypes = ["char", "short", "int", "long"]
defaultTypes += defaultTypes.map { "u" + $0 }
defaultTypes.append("float")

protocol ShaderRepresentable {
  static var shaderType: String { get }
}

extension Bool: ShaderRepresentable { static let shaderType = "bool" }
extension Int8: ShaderRepresentable { static let shaderType = "char" }
extension Int16: ShaderRepresentable { static let shaderType = "short" }
extension Int32: ShaderRepresentable { static let shaderType = "int" }
extension Int64: ShaderRepresentable { static let shaderType = "long" }
extension UInt8: ShaderRepresentable { static let shaderType = "uchar" }
extension UInt16: ShaderRepresentable { static let shaderType = "ushort" }
extension UInt32: ShaderRepresentable { static let shaderType = "uint" }
extension UInt64: ShaderRepresentable { static let shaderType = "ulong" }
extension Float: ShaderRepresentable { static let shaderType = "float" }

extension SIMD2: ShaderRepresentable where Scalar: ShaderRepresentable {
  static var shaderType: String {
    "\(Scalar.shaderType)2"
  }
}
extension SIMD3: ShaderRepresentable where Scalar: ShaderRepresentable {
  static var shaderType: String {
    "\(Scalar.shaderType)3"
  }
}
extension SIMD4: ShaderRepresentable where Scalar: ShaderRepresentable {
  static var shaderType: String {
    "\(Scalar.shaderType)4"
  }
}

// Use `auto` keyword to make the body type-generic.
// Returns a dictionary matching types to their respective kernel.
func generateKernelSources(
  body: String, types: [String] = defaultTypes
) -> [String: String] {
  var output: [String: String] = [:]
  for type in types {
    output[type] = """
    #if __METAL__
    #include <metal_stdlib>
    using namespace metal;
    
    #define KERNEL kernel
    #define GLOBAL device
    #define BUFFER_BINDING(index) [[buffer(index)]]
    #else
    \(headerString)
    
    #define KERNEL __kernel
    #define GLOBAL __global
    #define BUFFER_BINDING(index)
    #endif
    
    KERNEL void vectorOperation(
      GLOBAL \(type)* a BUFFER_BINDING(0),
      GLOBAL \(type)* b BUFFER_BINDING(1),
      GLOBAL \(type)* c BUFFER_BINDING(2)
    #if __METAL__
      , uint tid [[thread_position_in_grid]]
    ) {
    #else
    ) {
      uint tid = uint(get_global_id(0));
    #endif
      \(body)
    }
    """
  }
  return output
}

// Set of protocols to unify Metal and OpenCL tests. This also makes C-friendly
// OpenCL easier to harness from Swift. These will probably be `class` objects
// so the deinitializer can do OpenCL reference counting.

protocol GPUBackend {
  associatedtype Device: GPUDevice
  associatedtype Library: GPULibrary
  associatedtype Kernel: GPUKernel
  associatedtype Buffer: GPUBuffer
  associatedtype CommandQueue: GPUCommandQueue
}

protocol GPUDevice {
  
}

protocol GPULibrary {
  
}

protocol GPUKernel {
  
}

protocol GPUBuffer {
  
}

protocol GPUCommandQueue {
  associatedtype Backend: GPUBackend
  
  // Call this explicitly before encoding new commands.
  func startCommandBuffer()
  
  // Sets the kernel object for the current command.
  func setKernel(_ kernel: Backend.Kernel)
  
  // Set raw data argument for the kernel.
  func setBytes<T>(_ bytes: Array<T>, index: Int)
  
  // Set API buffer argument for the kernel.
  func setBuffer(_ buffer: Backend.Buffer, index: Int)
  
  // Call this explicitly before waiting for commands to finish.
  func submitCommandBuffer()
  
  // Halts until most recent command buffer has completed.
  func finishCommands()
}

#if false

strcpy(source_str, """
kernel void vector_add(global int4* a, global int4 *b, global int4* c) {
//char c1 = 1;
//char c2 = convert_char_sat(c1);
    float c5 = 2.0;
    float c6 = float(sub_group_scan_inclusive_add(uint(c5)));

    size_t i = get_global_id(0);
    c[i] = a[i] + b[i] + int(c6);// + int(char(simd_mefix_inclusive_sum(uchar(char(2)))));
}
""")



var source_size = strlen(source_str)
//var source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp)
//fclose(fp)

// Get platform and device information
var platform_id: cl_platform_id? = nil
var device_id: cl_device_id? = nil
var ret_num_devices: cl_uint = 0
var ret_num_platforms: cl_uint = 0
var ret: cl_int = clGetPlatformIDs(1, &platform_id, &ret_num_platforms)
ret = clGetDeviceIDs(platform_id, cl_device_type(CL_DEVICE_TYPE_DEFAULT), 1,
  &device_id, &ret_num_devices)

// Create OpenCL context
let context: cl_context = clCreateContext(nil, 1, &device_id, nil, nil, &ret)

// Create a command queue
let command_queue: cl_command_queue = clCreateCommandQueue(context, device_id, 0,
  &ret)

// Create memory buffers on the device for each vector
var a_mem_obj: cl_mem = clCreateBuffer(context, cl_mem_flags(CL_MEM_READ_ONLY),
  LIST_SIZE * MemoryLayout<Int32>.stride, nil, &ret)
var b_mem_obj: cl_mem = clCreateBuffer(context, cl_mem_flags(CL_MEM_READ_ONLY),
  LIST_SIZE * MemoryLayout<Int32>.stride, nil, &ret)
var c_mem_obj: cl_mem = clCreateBuffer(context, cl_mem_flags(CL_MEM_READ_ONLY),
  LIST_SIZE * MemoryLayout<Int32>.stride, nil, &ret)

// Copy the lists A and B to their respective memory buffers
ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, cl_bool(CL_TRUE), 0,
  LIST_SIZE * MemoryLayout<Int32>.stride, A, 0, nil, nil)
ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, cl_bool(CL_TRUE), 0,
  LIST_SIZE * MemoryLayout<Int32>.stride, B, 0, nil, nil)

// Create a program from the kernel source
var source_str_casted = unsafeBitCast(source_str, to: UnsafePointer<CChar>?.self)
let program: cl_program = clCreateProgramWithSource(context, 1,
  &source_str_casted, &source_size, &ret)

//var devices: [cl_device_id?] = [device_id]
//ret = clCompileProgram(program, 1, nil, nil, 0, nil, nil, nil, nil)

// Build the program
//ret = clBuildProgram(program, 1, &device_id, nil, nil, nil)
//print("Error", ret)

//var build_info_size: Int = 0
//clGetProgramBuildInfo(program, device_id, UInt32(CL_PROGRAM_BUILD_LOG), 0, nil, &build_info_size)
//print(build_info_size)
//
//var build_info = [UInt8](repeating: 0, count: build_info_size)
//clGetProgramBuildInfo(program, device_id, UInt32(CL_PROGRAM_BUILD_LOG), build_info_size, &build_info, nil)
//print(String(cString: build_info))
//
//var binary_type: cl_program_binary_type = 0
//clGetProgramBuildInfo(program, device_id, cl_program_build_info(CL_PROGRAM_BINARY_TYPE), cl_program_binary_type.bitWidth, nil, nil)
//print(binary_type == CL_PROGRAM_BINARY_TYPE_NONE)
//print(binary_type == CL_PROGRAM_BINARY_TYPE_LIBRARY)
//print(binary_type == CL_PROGRAM_BINARY_TYPE_EXECUTABLE)
//print(binary_type == CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT)
//
//var binary_sizes = [Int](repeating: 0, count: 10)
//clGetProgramInfo(program, cl_program_info(CL_PROGRAM_BINARY_SIZES), 8, &binary_sizes, nil)
//print(binary_sizes)

//var binary = [UInt8](repeating: 0, count: binary_sizes[0])
//var binary_size: Int = 0
//var output: UnsafeMutablePointer<UInt8>?
//clGetProgramInfo(program, cl_program_info(CL_PROGRAM_BINARIES), 8, &output, nil)
//print(output)

print("Test")
//exit(0)

// Build the program
ret = clBuildProgram(program, 1, &device_id, nil, nil, nil)

// Create the OpenCL kernel
let kernel: cl_kernel = clCreateKernel(program, "vector_add", &ret)

// Set the arguments of the kernel
let size_of_cl_mem = MemoryLayout<cl_mem>.stride
ret = clSetKernelArg(kernel, 0, size_of_cl_mem, &a_mem_obj)
ret = clSetKernelArg(kernel, 1, size_of_cl_mem, &b_mem_obj)
ret = clSetKernelArg(kernel, 2, size_of_cl_mem, &c_mem_obj)

// Execute the OpenCL kernel on the list
var global_item_size = LIST_SIZE
var local_item_size = 64
ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, nil, &global_item_size,
  &local_item_size, 0, nil, nil)

// Read the memory buffer C on the device to the local variable C
let C = UnsafeMutablePointer<Int32>.allocate(capacity: LIST_SIZE)
ret = clEnqueueReadBuffer(command_queue, c_mem_obj, cl_bool(CL_TRUE), 0,
  LIST_SIZE * MemoryLayout<Int32>.stride, C, 0, nil, nil)

// Display the result to the screen
for i in 0..<LIST_SIZE {
  print("\(A[i]) + \(B[i]) = \(C[i])")
  
  /*
   Output:
   
   0 + 1024 = 1024
   1 + 1023 = 1024
   2 + 1022 = 1024
   3 + 1021 = 1024
   4 + 1020 = 1024
   5 + 1019 = 1024
   6 + 1018 = 1024
   ...
   1019 + 5 = 1024
   1020 + 4 = 1024
   1021 + 3 = 1024
   1022 + 2 = 1024
   1023 + 1 = 1024
   */
}

// Clean up
ret = clFlush(command_queue)
ret = clFinish(command_queue)
ret = clReleaseKernel(kernel)
ret = clReleaseProgram(program)
ret = clReleaseMemObject(a_mem_obj)
ret = clReleaseMemObject(b_mem_obj)
ret = clReleaseMemObject(c_mem_obj)
ret = clReleaseCommandQueue(command_queue)
ret = clReleaseContext(context)
free(A)
free(B)
free(C)
exit(0)
#endif
