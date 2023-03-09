//
//  Tests.swift
//  OpenCL Metal Stdlib
//
//  Created by Philip Turner on 2/24/23.
//

import Metal
import OpenCL

func showUsage() -> Never {
  fatalError("""
    Usage: swift Tests.swift [--headers-directory <path>] \
    [--use-subgroup-extended-types]
    """)
}

var headerURL: URL
if let headersDirectoryFlagIndex = CommandLine.arguments.firstIndex(
  of: "--headers-directory") {
  let headerDirectoryIndex = headersDirectoryFlagIndex + 1
  guard CommandLine.arguments.count > headerDirectoryIndex else {
    showUsage()
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
  atPath: headerURL.relativePath) else {
  fatalError("Invalid header path: \(headerURL.relativePath)")
}

var usingSubgroupExtendedTypes = false
if CommandLine.arguments.contains("--use-subgroup-extended-types") {
  usingSubgroupExtendedTypes = true
}

if CommandLine.arguments.contains(where: { argument in
  argument.starts(with: "--")
  && argument != "--headers-directory"
  && argument != "--use-subgroup-extended-types"
}) {
  showUsage()
}

// Workaround for Swift semantic capture scope issue.
var headerString: String?
headerString = String(data: headerData, encoding: .utf8)
guard headerString != nil else {
  fatalError("Malformatted header: \(headerURL.relativePath)")
}

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

typealias AutogeneratibleScalar =
  ShaderRepresentable & Numeric & SIMDScalar

typealias AutogeneratibleVector =
  ShaderRepresentable & SIMD<AutogeneratibleScalar>

let defaultTypeGroups: [any TypeGroupProtocol.Type] =
usingSubgroupExtendedTypes ? [
  TypeGroup<Int8>.self,
  TypeGroup<Int16>.self,
  TypeGroup<Int32>.self,
  TypeGroup<UInt8>.self,
  TypeGroup<UInt16>.self,
  TypeGroup<UInt32>.self,
  TypeGroup<Float>.self,
] : [
  TypeGroup<Int32>.self,
  TypeGroup<UInt32>.self,
  TypeGroup<Float>.self,
]

let integerTypeGroups: [any TypeGroupProtocol.Type] =
usingSubgroupExtendedTypes ? [
  TypeGroup<Int8>.self,
  TypeGroup<Int16>.self,
  TypeGroup<Int32>.self,
  TypeGroup<UInt8>.self,
  TypeGroup<UInt16>.self,
  TypeGroup<UInt32>.self,
] : [
  TypeGroup<Int32>.self,
  TypeGroup<UInt32>.self,
]

// Initialize vectors by first initializing the scalar, then initializing the
// vector through `init(repeating:)`.
struct TypeGroup<T: AutogeneratibleScalar>: TypeGroupProtocol {
  static var scalar: T.Type { T.self }
  static var vector2: SIMD2<T>.Type { SIMD2<T>.self }
  static var vector3: SIMD3<T>.Type { SIMD3<T>.self }
  static var vector4: SIMD4<T>.Type { SIMD4<T>.self }
}

protocol TypeGroupProtocol {
  associatedtype T: AutogeneratibleScalar
  static var scalar: T.Type { get }
  static var vector2: SIMD2<T>.Type { get }
  static var vector3: SIMD3<T>.Type { get }
  static var vector4: SIMD4<T>.Type { get }
}


let defaultScalarTypeStrings = defaultTypeGroups.map {
  $0.self.scalar.shaderType
}
var defaultTypeStrings = defaultScalarTypeStrings
if usingSubgroupExtendedTypes {
  defaultTypeStrings += defaultScalarTypeStrings.map { $0 + "2" }
  defaultTypeStrings += defaultScalarTypeStrings.map { $0 + "3" }
  defaultTypeStrings += defaultScalarTypeStrings.map { $0 + "4" }
}

let integerScalarTypeStrings = integerTypeGroups.map {
  $0.self.scalar.shaderType
}
var integerTypeStrings = integerScalarTypeStrings
if usingSubgroupExtendedTypes {
  integerTypeStrings += integerScalarTypeStrings.map { $0 + "2" }
  integerTypeStrings += integerScalarTypeStrings.map { $0 + "3" }
  integerTypeStrings += integerScalarTypeStrings.map { $0 + "4" }
}

// Returns a group of shader permutations for each type.
// Use `auto` keyword to make the body type-generic.
func generateSource(
  body: String,
  types: [String] = defaultTypeStrings
) -> String {
  var output: String = ""
  
  if usingSubgroupExtendedTypes {
    output += """
      #define OPENCL_USE_SUBGROUP_EXTENDED_TYPES 1
      
      """
  }
  
  output += """
    #if __METAL__
    #include <metal_stdlib>
    using namespace metal;
    
    #define KERNEL kernel
    #define GLOBAL device
    #define LOCAL threadgroup
    #define BUFFER_BINDING(index) [[buffer(index)]]
    #else
    \(headerString!)
    
    #define KERNEL __kernel
    #define GLOBAL __global
    #define LOCAL __local
    #define BUFFER_BINDING(index)
    #endif
    
    """
  
  for type in types {
    output.append("""
      KERNEL void vectorOperation_\(type)(
        GLOBAL \(type)* a BUFFER_BINDING(0),
        GLOBAL \(type)* b BUFFER_BINDING(1),
        GLOBAL \(type)* c BUFFER_BINDING(2)
      #if __METAL__
        ,
        uint tid [[thread_position_in_grid]],
        ushort lane_id [[thread_index_in_simdgroup]]
      ) {
      #else
      ) {
        uint tid = uint(get_global_id(0));
      
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wunused-variable"
        ushort lane_id = ushort(get_local_id(0)) % 32;
        #pragma clang diagnostic pop
      #endif
        \(body)
      }
      
      """)
  }
  
  return output
}

// MARK: - Protocol Definitions

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
  associatedtype Backend: GPUBackend
  
  // Only one GPU device on an Apple silicon system.
  init()
  
  func createLibrary(source: String) -> Backend.Library
  
  func createBuffer(length: Int) -> Backend.Buffer
  
  func createBuffer<T>(_ bytes: Array<T>) -> Backend.Buffer
  
  func createCommandQueue() -> Backend.CommandQueue
}

protocol GPULibrary {
  associatedtype Backend: GPUBackend
  
  func createKernel(name: String) -> Backend.Kernel
}

protocol GPUKernel {
  associatedtype Backend: GPUBackend
}

protocol GPUBuffer {
  associatedtype Backend: GPUBackend
  
  // Must finish the queue before calling this.
  func setElements<T>(_ bytes: Array<T>)
  
  // Must finish the queue before calling this.
  func getElements<T>() -> Array<T>
}

protocol GPUCommandQueue {
  associatedtype Backend: GPUBackend
  
  // Call this explicitly before encoding new commands.
  func startCommandBuffer()
  
  // Call this before binding any arguments.
  func setKernel(_ kernel: Backend.Kernel)
  
  // Set API buffer argument for the kernel.
  func setBuffer(_ buffer: Backend.Buffer, index: Int)
  
  // Execute the kernel.
  func dispatchThreads(_ threads: Int, threadgroupSize: Int)
  
  // Call this explicitly before waiting for commands to finish.
  func submitCommandBuffer()
  
  // Halts until most recent command buffer has completed.
  func finishCommands()
}

extension GPUCommandQueue {
  // Allows for specifying the kernel objects without generic constrants.
  func setKernel(_ kernel: any GPUKernel) {
    setKernel(kernel as! Backend.Kernel)
  }
  
  // Allows for specifying the buffer objects without generic constrants.
  func setBuffer(_ buffer: any GPUBuffer, index: Int) {
    setBuffer(buffer as! Backend.Buffer, index: index)
  }
}

// MARK: - Metal Types

class MetalBackend: GPUBackend {
  typealias Device = MetalDevice
  typealias Library = MetalLibrary
  typealias Kernel = MetalKernel
  typealias Buffer = MetalBuffer
  typealias CommandQueue = MetalCommandQueue
}

class MetalDevice: GPUDevice {
  typealias Backend = MetalBackend
  
  var device: MTLDevice
  
  required init() {
    self.device = MTLCopyAllDevices().first!
  }
  
  func createLibrary(source: String) -> MetalLibrary {
    let library = try! device.makeLibrary(source: source, options: nil)
    return MetalLibrary(library: library)
  }
  
  func createBuffer(length: Int) -> MetalBuffer {
    // Shared storage mode by default.
    let buffer = device.makeBuffer(length: length)!
    return MetalBuffer(buffer: buffer)
  }
  
  func createBuffer<T>(_ bytes: Array<T>) -> MetalBuffer {
    // Shared storage mode by default.
    let numBytes = bytes.count * MemoryLayout<T>.stride
    let buffer = device.makeBuffer(bytes: bytes, length: numBytes)!
    return MetalBuffer(buffer: buffer)
  }
  
  func createCommandQueue() -> MetalCommandQueue {
    let commandQueue = device.makeCommandQueue()!
    return MetalCommandQueue(commandQueue: commandQueue)
  }
}

class MetalLibrary: GPULibrary {
  typealias Backend = MetalBackend
  
  var library: MTLLibrary
  
  init(library: MTLLibrary) {
    self.library = library
  }
  
  func createKernel(name: String) -> MetalKernel {
    let function = library.makeFunction(name: name)!
    let pipeline = try! library.device
      .makeComputePipelineState(function: function)
    return MetalKernel(pipeline: pipeline)
  }
}

class MetalKernel: GPUKernel {
  typealias Backend = MetalBackend
  
  var pipeline: MTLComputePipelineState
  
  init(pipeline: MTLComputePipelineState) {
    self.pipeline = pipeline
  }
}

class MetalBuffer: GPUBuffer {
  typealias Backend = MetalBackend
  
  var buffer: MTLBuffer
  
  init(buffer: MTLBuffer) {
    self.buffer = buffer
  }
  
  func setElements<T>(_ bytes: Array<T>) {
    bytes.withUnsafeBufferPointer { pointer in
      let numBytes = pointer.count * MemoryLayout<T>.stride
      memcpy(buffer.contents(), pointer.baseAddress, numBytes)
    }
  }
  
  func getElements<T>() -> Array<T> {
    let arraySize = buffer.length / MemoryLayout<T>.stride
    precondition(
      buffer.length % MemoryLayout<T>.stride == 0,
      "Buffer not evenly divisible into elements of type '\(T.self)'.")
    
    // Using unsafe array initializer to avoid unnecessary copy.
    return Array(unsafeUninitializedCapacity: arraySize) { pointer, count in
      memcpy(pointer.baseAddress, buffer.contents(), buffer.length)
      count = arraySize
    }
  }
}

class MetalCommandQueue: GPUCommandQueue {
  typealias Backend = MetalBackend
  
  var commandQueue: MTLCommandQueue
  var commandBuffer: MTLCommandBuffer!
  var encoder: MTLComputeCommandEncoder!
  
  init(commandQueue: MTLCommandQueue) {
    self.commandQueue = commandQueue
  }
  
  func startCommandBuffer() {
    precondition(
      commandBuffer == nil ||
      commandBuffer?.status == .committed ||
      commandBuffer?.status == .scheduled ||
      commandBuffer?.status == .completed,
      "Called `startCommandBuffer` before submitting the current command buffer.")
    commandBuffer = commandQueue.makeCommandBuffer()!
    encoder = commandBuffer.makeComputeCommandEncoder()
  }
  
  func setKernel(_ kernel: MetalKernel) {
    encoder.setComputePipelineState(kernel.pipeline)
  }
  
  func setBuffer(_ buffer: MetalBuffer, index: Int) {
    encoder.setBuffer(buffer.buffer, offset: 0, index: index)
  }
  
  func dispatchThreads(_ threads: Int, threadgroupSize: Int) {
    encoder.dispatchThreads(
      MTLSizeMake(threads, 1, 1),
      threadsPerThreadgroup: MTLSizeMake(threadgroupSize, 1, 1))
  }
  
  func submitCommandBuffer() {
    encoder.endEncoding()
    commandBuffer.commit()
  }
  
  func finishCommands() {
    precondition(
      commandBuffer.status == .committed,
      "Called `finishCommands` before submitting the current command buffer.")
    commandBuffer.waitUntilCompleted()
  }
}

// MARK: - OpenCL Types

func checkOpenCLError(
  _ ret: Int32,
  file: StaticString = #file,
  line: UInt = #line
) {
  if _slowPath(ret != CL_SUCCESS) {
    fatalError("Encountered OpenCL error code \(ret).", file: file, line: line)
  }
}

class OpenCLBackend: GPUBackend {
  typealias Device = OpenCLDevice
  typealias Library = OpenCLLibrary
  typealias Kernel = OpenCLKernel
  typealias Buffer = OpenCLBuffer
  typealias CommandQueue = OpenCLCommandQueue
}

class OpenCLDevice: GPUDevice {
  typealias Backend = OpenCLBackend
  
  var platform: cl_platform_id
  var device: cl_device_id
  var context: cl_context
  private var mappingCommandQueue: OpenCLCommandQueue!
  
  required init() {
    var platform: cl_platform_id? = nil
    var ret_num_platforms: cl_uint = 0
    checkOpenCLError(clGetPlatformIDs(1, &platform, &ret_num_platforms))
    self.platform = platform!
    
    var device: cl_device_id? = nil
    var ret_num_devices: cl_uint = 0
    checkOpenCLError(clGetDeviceIDs(
      platform, cl_device_type(CL_DEVICE_TYPE_DEFAULT), 1, &device,
      &ret_num_devices))
    self.device = device!
    
    var ret: cl_int = 0
    self.context = clCreateContext(nil, 1, &device, nil, nil, &ret)
    checkOpenCLError(ret)
    
    self.mappingCommandQueue = createCommandQueue()
  }
  
  deinit {
    clReleaseContext(context)
  }
  
  func createLibrary(source: String) -> OpenCLLibrary {
    var ret: cl_int = 0
    var source_count = source.count
    let program = source.withCString { source_ptr in
      var source_ptr_copy: Optional = source_ptr
      return clCreateProgramWithSource(
        context, 1, &source_ptr_copy, &source_count, &ret)
    }
    checkOpenCLError(ret)
    
    var device_copy: Optional = device
    ret = clBuildProgram(program, 1, &device_copy, nil, nil, nil)
    
    // Diagnose errors when the shader won't compile.
    if ret != 0 {
      var build_status: cl_build_status = 0
      var ret2 = clGetProgramBuildInfo(
        program, device, cl_program_build_info(CL_PROGRAM_BUILD_STATUS), 4,
        &build_status, nil)
      precondition(ret2 == 0, "Failed unexpectedly.")
      print("Build status: \(build_status)")
      
      var build_log_size: Int = 0
      ret2 = clGetProgramBuildInfo(
        program, device, cl_program_build_info(CL_PROGRAM_BUILD_LOG), 0, nil,
        &build_log_size)
      precondition(ret2 == 0, "Failed unexpectedly.")
      print("Build log size: \(build_log_size)")
      
      var build_log = [UInt8](repeating: 0, count: build_log_size)
      ret2 = clGetProgramBuildInfo(
        program, device, cl_program_build_info(CL_PROGRAM_BUILD_LOG),
        build_log_size, &build_log, nil)
      precondition(ret2 == 0, "Failed unexpectedly.")
      let logString = String(cString: build_log)
      print("Build log:\n\(logString)")
    }
    
    checkOpenCLError(ret)
    return OpenCLLibrary(program: program!)
  }

  func createBuffer(length: Int) -> OpenCLBuffer {
    var ret: cl_int = 0
    let flags = CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR
    let buffer = clCreateBuffer(
      context, cl_mem_flags(flags), length, nil, &ret)
    checkOpenCLError(ret)
    
    let contents = extractHostPointer(buffer: buffer!, length: length)
    return OpenCLBuffer(buffer: buffer!, contents: contents, length: length)
  }

  func createBuffer<T>(_ bytes: Array<T>) -> OpenCLBuffer {
    var ret: cl_int = 0
    let flags = CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR
    var length = 0
    let buffer = bytes.withUnsafeBytes { pointer in
      length = pointer.count
      let casted = unsafeBitCast(
        pointer.baseAddress, to: UnsafeMutableRawPointer?.self)
      return clCreateBuffer(
        context, cl_mem_flags(flags), length, casted, &ret)
      
    }
    checkOpenCLError(ret)
    
    let contents = extractHostPointer(buffer: buffer!, length: length)
    return OpenCLBuffer(buffer: buffer!, contents: contents, length: length)
  }

  func createCommandQueue() -> OpenCLCommandQueue {
    var ret: cl_int = 0
    let commandQueue = clCreateCommandQueue(context, device, 0, &ret)
    checkOpenCLError(ret)
    return OpenCLCommandQueue(commandQueue: commandQueue!)
  }
  
  private func extractHostPointer(
    buffer: cl_mem, length: Int
  ) -> UnsafeMutableRawPointer {
    var ret: cl_int = 0
    let hostPointer = clEnqueueMapBuffer(
      mappingCommandQueue.commandQueue, buffer, cl_bool(CL_TRUE),
      cl_map_flags(CL_MAP_READ | CL_MAP_WRITE), 0, length, 0, nil, nil, &ret)
    checkOpenCLError(ret)
    return hostPointer!
  }
}

class OpenCLLibrary: GPULibrary {
  typealias Backend = OpenCLBackend
  
  var program: cl_program
  
  init(program: cl_program) {
    self.program = program
  }
  
  deinit {
    clReleaseProgram(program)
  }
  
  func createKernel(name: String) -> OpenCLKernel {
    var ret: cl_int = 0
    let kernel = clCreateKernel(program, name, &ret)
    checkOpenCLError(ret)
    return OpenCLKernel(kernel: kernel!)
  }
}

class OpenCLKernel: GPUKernel {
  typealias Backend = OpenCLBackend
  
  var kernel: cl_kernel
  
  init(kernel: cl_kernel) {
    self.kernel = kernel
  }
  
  deinit {
    clReleaseKernel(kernel)
  }
}

class OpenCLBuffer: GPUBuffer {
  typealias Backend = OpenCLBackend
  
  var buffer: cl_mem
  var contents: UnsafeMutableRawPointer
  var length: Int
  
  init(buffer: cl_mem, contents: UnsafeMutableRawPointer, length: Int) {
    self.buffer = buffer
    self.contents = contents
    self.length = length
  }
  
  func setElements<T>(_ bytes: Array<T>) {
    bytes.withUnsafeBufferPointer { pointer in
      let numBytes = pointer.count * MemoryLayout<T>.stride
      memcpy(contents, pointer.baseAddress, numBytes)
    }
  }
  
  func getElements<T>() -> Array<T> {
    let arraySize = length / MemoryLayout<T>.stride
    precondition(
      length % MemoryLayout<T>.stride == 0,
      "Buffer not evenly divisible into elements of type '\(T.self)'.")
    
    // Using unsafe array initializer to avoid unnecessary copy.
    return Array(unsafeUninitializedCapacity: arraySize) { pointer, count in
      memcpy(pointer.baseAddress, contents, length)
      count = arraySize
    }
  }
  
  deinit {
    clReleaseMemObject(buffer)
  }
}

class OpenCLCommandQueue: GPUCommandQueue {
  typealias Backend = OpenCLBackend
  
  var commandQueue: cl_command_queue
  var currentKernel: OpenCLKernel!
  
  init(commandQueue: cl_command_queue) {
    self.commandQueue = commandQueue
  }
  
  func startCommandBuffer() {
    precondition(
      currentKernel == nil,
      "Called `startCommandBuffer` before submitting the current command buffer.")
  }
  
  func setKernel(_ kernel: OpenCLKernel) {
    currentKernel = kernel
  }
  
  func setBuffer(_ buffer: OpenCLBuffer, index: Int) {
    var argument = buffer.buffer
    checkOpenCLError(clSetKernelArg(
      currentKernel.kernel, cl_uint(index), 8, &argument))
  }
  
  func dispatchThreads(_ threads: Int, threadgroupSize: Int) {
    var workDimensions: Int = threads
    var localDimensions: Int = threadgroupSize
    checkOpenCLError(clEnqueueNDRangeKernel(
      commandQueue, currentKernel.kernel, 1, nil, &workDimensions,
      &localDimensions, 0, nil, nil))
  }
  
  func submitCommandBuffer() {
    currentKernel = nil
    clFlush(commandQueue)
  }
  
  func finishCommands() {
    precondition(
      currentKernel == nil,
      "Called `finishCommands` before submitting the current command buffer.")
    clFinish(commandQueue)
  }
  
  deinit {
    clFlush(commandQueue)
    clFinish(commandQueue)
    clReleaseCommandQueue(commandQueue)
  }
}

// MARK: - Tests

// Use `Any` type to make an array with different types.
struct KernelInvocation<T>: KernelInvocationProtocol {
  var typeName: String
  var kernel: any GPUKernel
  
  var bufferA: any GPUBuffer
  var bufferB: any GPUBuffer
  var bufferC: any GPUBuffer
  var expectedC: Array<T>
  
  init(
    device: any GPUDevice,
    library: any GPULibrary,
    inputA: Array<T>,
    inputB: Array<T>,
    expectedC: Array<T>
  ) {
    self.typeName = (T.self as! any ShaderRepresentable.Type).shaderType
    self.kernel = library.createKernel(name: "vectorOperation_\(typeName)")
    
    precondition(
      inputA.count == inputB.count && inputB.count == expectedC.count,
      "Inputs had different length.")
    self.bufferA = device.createBuffer(inputA)
    self.bufferB = device.createBuffer(inputB)
    self.bufferC = device.createBuffer(
      length: expectedC.count * MemoryLayout<T>.stride)
    self.expectedC = expectedC
  }
  
  // Returns a textual error if something didn't match expected.
  //
  // Finish the command queue before validating. It is a good idea to enqueue
  // future commands and validate asynchronously.
  func validate() -> String? {
    let actualC: Array<T> = self.bufferC.getElements()
    let length = expectedC.count * MemoryLayout<T>.stride
    if memcmp(actualC, expectedC, length) == 0 {
      return nil
    }
    
    for i in 0..<expectedC.count {
      // Workaround for Swift protocols preventing conformance to `Equatable`.
      var actual = actualC[i]
      var expected = expectedC[i]
      if _slowPath(memcmp(&actual, &expected, MemoryLayout<T>.stride) != 0) {
        // Sometimes floating-point numbers confuse -0.0 with +0.0.
        var shouldFail = true
        if T.self == Float.self {
          shouldFail = (actual as! Float) != (expected as! Float)
        }
        if T.self == SIMD2<Float>.self {
          shouldFail = any((actual as! SIMD2<Float>) .!=
                           (expected as! SIMD2<Float>))
        }
        if T.self == SIMD3<Float>.self {
          shouldFail = any((actual as! SIMD3<Float>) .!=
                           (expected as! SIMD3<Float>))
        }
        if T.self == SIMD4<Float>.self {
          shouldFail = any((actual as! SIMD4<Float>) .!=
                           (expected as! SIMD4<Float>))
        }
        if shouldFail {
          return "Element \(i): actual '\(actual)' != expected '\(expected)'"
        }
      }
    }
    
    // The mismatched element simply occurred because of floating-point signs.
    return nil
  }
}

// Bypass's Swift's inability to perform polymorphism over generic parameters.
protocol KernelInvocationProtocol {
  var typeName: String { get }
  var kernel: any GPUKernel { get }
  
  var bufferA: any GPUBuffer { get }
  var bufferB: any GPUBuffer { get }
  var bufferC: any GPUBuffer { get }
  
  func validate() -> String?
}

struct KernelInvocationGroup {
  var deviceType: Any.Type
  var body: String
  var invocations: [KernelInvocationProtocol] = []
  
  init(
    device: any GPUDevice,
    body: String,
    sequenceSize: Int,
    omitFloat: Bool = false,
    generate: (
      _ index: Int,
      _ A: inout Int,
      _ B: inout Int,
      _ C: inout Int
    ) -> Void
  ) {
    let types = omitFloat ? integerTypeStrings : defaultTypeStrings
    let source = generateSource(body: body, types: types)
    let library = device.createLibrary(source: source)
    self.deviceType = type(of: device)
    self.body = body
    
    precondition(
      sequenceSize <= 32 && sequenceSize.nonzeroBitCount == 1,
      "Sequence size must be power of 2, no more than 32.")
    var A: [Int] = .init(repeating: 0, count: sequenceSize)
    var B: [Int] = .init(repeating: 0, count: sequenceSize)
    var C: [Int] = .init(repeating: 0, count: sequenceSize)
    for i in 0..<sequenceSize {
      generate(i, &A[i], &B[i], &C[i])
    }
    while A.count < 32 {
      A.append(contentsOf: A)
      B.append(contentsOf: B)
      C.append(contentsOf: C)
    }
    
    let typeGroups = omitFloat ? integerTypeGroups : defaultTypeGroups
    for typeGroup in typeGroups {
      body(type: typeGroup)
      
      // Workaround to turn dynamic type into generic function type in Swift's
      // type system.
      func body<T: TypeGroupProtocol>(type: T.Type) {
        typealias T1 = T.T
        typealias T2 = SIMD2<T.T>
        typealias T3 = SIMD3<T.T>
        typealias T4 = SIMD4<T.T>
        
        let scalarsA: [T1] = A.map { T1(exactly: $0)! }
        let scalarsB: [T1] = B.map { T1(exactly: $0)! }
        let scalarsC: [T1] = C.map { T1(exactly: $0)! }
        let scalarInvocation = KernelInvocation(
          device: device, library: library, inputA: scalarsA,
          inputB: scalarsB, expectedC: scalarsC)
        invocations.append(scalarInvocation)
        
        func appendVectorInvocation<TN: SIMD>(type: TN.Type)
        where TN.Scalar: AutogeneratibleScalar, TN.Scalar == T1
        {
          let vectorsA: [TN] = scalarsA.map(TN.init(repeating:))
          let vectorsB: [TN] = scalarsB.map(TN.init(repeating:))
          let vectorsC: [TN] = scalarsC.map(TN.init(repeating:))
          let vectorInvocation = KernelInvocation(
            device: device, library: library, inputA: vectorsA,
            inputB: vectorsB, expectedC: vectorsC)
          invocations.append(vectorInvocation)
        }
        
        if usingSubgroupExtendedTypes {
          appendVectorInvocation(type: T2.self)
          appendVectorInvocation(type: T3.self)
          appendVectorInvocation(type: T4.self)
        }
      }
    }
  }
  
  func encode(queue: any GPUCommandQueue) {
    queue.startCommandBuffer()
    
    for invocation in invocations {
      queue.setKernel(invocation.kernel)
      queue.setBuffer(invocation.bufferA, index: 0)
      queue.setBuffer(invocation.bufferB, index: 1)
      queue.setBuffer(invocation.bufferC, index: 2)
      queue.dispatchThreads(32, threadgroupSize: 32)
    }
    
    queue.submitCommandBuffer()
  }
  
  // Must finish the command queue before validating.
  func validate() {
    for invocation in invocations {
      if let error = invocation.validate() {
        let errorMessage = """
          
          Kernel invocation failed.
          Backend: \(deviceType)
          Type: \(invocation.typeName)
          Source:\n\(body)
          \(error)
          """
        print(errorMessage)
        exit(0)
      }
    }
  }
}

// Run Metal and OpenCL commands in parallel, improving GPU utilization.
// It's okay to generate the source code twice, if that makes debugging easier.
let deviceTypes: [any GPUDevice.Type] = [
  MetalDevice.self,
  OpenCLDevice.self,
  OpenCLDevice.self,
  OpenCLDevice.self,
  OpenCLDevice.self,
]

// Thread 1: Metal Stdlib functions from Metal
// Thread 2: Metal Stdlib functions from OpenCL
// Thread 3: base OpenCL Stdlib extensions
// Thread 4: uniform, clustered OpenCL subgroup functions
DispatchQueue.concurrentPerform(iterations: 5) { deviceIndex in
  let device = deviceTypes[deviceIndex].init()
  let queue = device.createCommandQueue()
  
  let isExtensionThread = deviceIndex == 2
  let isOtherThread = deviceIndex == 3
  
  var allGroups: [KernelInvocationGroup] = []
  func appendGroup(_ group: KernelInvocationGroup) {
    group.encode(queue: queue)
    allGroups.append(group)
  }
  do {
    // Vector addition test.
    appendGroup(KernelInvocationGroup(
      device: device,
      body: """
        c[tid] = a[tid] + b[tid];
        """,
      sequenceSize: 32,
      generate: { index, A, B, C in
        A = index
        B = 2 * index
        C = A + B
      }
    ))
  }
  
  // Input Data
  
  func logical_and(_ x: Int, _ y: Int) -> Int {
    let x_boolean = x != 0 ? 1 : 0
    let y_boolean = y != 0 ? 1 : 0
    return x_boolean & y_boolean
  }
  
  func logical_or(_ x: Int, _ y: Int) -> Int {
    let x_boolean = x != 0 ? 1 : 0
    let y_boolean = y != 0 ? 1 : 0
    return x_boolean | y_boolean
  }
  
  func logical_xor(_ x: Int, _ y: Int) -> Int {
    let x_boolean = x != 0 ? 1 : 0
    let y_boolean = y != 0 ? 1 : 0
    return x_boolean ^ y_boolean
  }
  
  // Integer sequences cannot produce something over 127, because that would
  // overflow an 8-bit signed integer.
  let sumSequence4 = [
    12, 33, 22, 4,
  ]
  let productSequence4 = [
    7, 1, 5, 2,
  ]
  let bitwiseSequence4 = [
    0b0101_0110,
    0b0110_1110,
    0b0011_0100,
    0b0110_0110,
  ]
  let logicalSequence4 = [
    93, 0, 1, 9
  ]
  precondition(sumSequence4.reduce(0, +) == 71)
  precondition(sumSequence4.reduce(0, +) <= 127)
  precondition(productSequence4.reduce(1, *) == 70)
  precondition(productSequence4.reduce(1, *) <= 127)
  precondition(sumSequence4.reduce(127, min) == 4)
  precondition(sumSequence4.reduce(0, max) == 33)
  precondition(bitwiseSequence4.reduce(0x7F, &) == 0b0000_0100)
  precondition(bitwiseSequence4.reduce(0x00, |) == 0b0111_1110)
  precondition(bitwiseSequence4.reduce(0x00, ^) == 0b0110_1010)
  precondition(logicalSequence4.reduce(1, logical_and) == 0)
  precondition(logicalSequence4.reduce(0, logical_or) == 1)
  precondition(logicalSequence4.reduce(0, logical_xor) == 1)
  
  let sumSequence32 = [
    2, 1, 0, 3, 4, 5, 6, 7,
    7, 6, 5, 4, 3, 2, 1, 0,
    1, 2, 3, 4, 0, 7, 6, 5,
    5, 8, 2, 1, 4, 4, 4, 3,
  ]
  let productSequence32 = [
    1, 1, 1, 1, 1, 3, 1, 1,
    1, 2, 1, 1, 1, 1, 2, 1,
    1, 1, 2, 1, 2, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 2,
  ]
  let bitwiseSequence32 = [
    0b0101_0110, 0b0110_1110, 0b0011_0100, 0b0110_0110,
    0b0010_0100, 0b0011_0110, 0b0100_0110, 0b0110_1110,
    0b0011_0100, 0b0110_0110, 0b0101_0110, 0b0110_1110,
    0b0111_0110, 0b0011_0100, 0b0011_0100, 0b0101_0110,
    0b0100_0110, 0b0110_1110, 0b0011_0100, 0b0110_0110,
    0b0011_0100, 0b0110_0110, 0b0011_0100, 0b0011_0100,
    0b0010_0100, 0b0011_0110, 0b0101_0110, 0b0110_1110,
    0b0011_0100, 0b0110_0110, 0b0001_0110, 0b0110_1110,
  ]
  let logicalSequence32 = [
    36,  0, 52,  0, 49, 35, 22, 99,
    91, 12, 87, 51,  0,  6, 19,  0,
     0, 17, 84,  3, 63, 55, 57, 15,
     0, 22, 35, 64, 26, 99,  0, 61,
  ]
  precondition(sumSequence32.reduce(0, +) == 115)
  precondition(sumSequence32.reduce(0, +) <= 127)
  precondition(productSequence32.reduce(1, *) == 96)
  precondition(productSequence32.reduce(1, *) <= 127)
  precondition(sumSequence32.reduce(127, min) == 0)
  precondition(sumSequence32.reduce(0, max) == 8)
  precondition(bitwiseSequence32.reduce(0x7F, &) == 0b0000_0100)
  precondition(bitwiseSequence32.reduce(0x00, |) == 0b0111_1110)
  precondition(bitwiseSequence32.reduce(0x00, ^) == 0b0011_0010)
  precondition(logicalSequence32.reduce(1, logical_and) == 0)
  precondition(logicalSequence32.reduce(0, logical_or) == 1)
  precondition(logicalSequence32.reduce(0, logical_xor) == 1)
  
  enum Scope {
    case quad
    case simd
    
    var clusterSize: Int {
      switch self {
      case .quad: return 4
      case .simd: return 32
      }
    }
    
    var metalPrefix: String {
      switch self {
      case .quad: return "quad"
      case .simd: return "simd"
      }
    }
  }
  
  // Standard Reductions
  
  struct ReductionParams {
    var scope: Scope
    var metalName: String
    var openclName: String
    
    // Before cl_khr_subgroup_non_uniform_arithmetic, certain reduction
    // operations were not supported.
    var preNonUniform: Bool = false
    
    // Whether to only use integers.
    var omitFloat: Bool = false
    
    // Some reductions don't have prefix versions.
    var omitPrefixReduction: Bool = false
    
    // The number sequence to use.
    var sequence: [Int]
    
    // The number to combine the first operand with.
    var identity: Int
    
    // Combine two operands in sequence.
    var execute: (Int, Int) -> Int
  }
  
  let reductionLoopParams = [
    ReductionParams(
      scope: .quad, metalName: "sum", openclName: "add",
      preNonUniform: true,
      sequence: sumSequence4, identity: 0, execute: +),
    ReductionParams(
      scope: .quad, metalName: "product", openclName: "mul",
      sequence: productSequence4, identity: 1, execute: *),
    ReductionParams(
      scope: .quad, metalName: "min", openclName: "min",
      preNonUniform: true, omitPrefixReduction: true,
      sequence: sumSequence4, identity: 255, execute: min),
    ReductionParams(
      scope: .quad, metalName: "max", openclName: "max",
      preNonUniform: true, omitPrefixReduction: true,
      sequence: sumSequence4, identity: 0, execute: max),
    ReductionParams(
      scope: .quad, metalName: "min", openclName: "min",
      preNonUniform: true, omitPrefixReduction: true,
      sequence: sumSequence4, identity: 255, execute: min),
    ReductionParams(
      scope: .quad, metalName: "and", openclName: "and",
      omitFloat: true, omitPrefixReduction: true,
      sequence: bitwiseSequence4, identity: 0x7F, execute: &),
    ReductionParams(
      scope: .quad, metalName: "or", openclName: "or",
      omitFloat: true, omitPrefixReduction: true,
      sequence: bitwiseSequence4, identity: 0x00, execute: |),
    ReductionParams(
      scope: .quad, metalName: "xor", openclName: "xor",
      omitFloat: true, omitPrefixReduction: true,
      sequence: bitwiseSequence4, identity: 0x00, execute: ^),
    
    ReductionParams(
      scope: .simd, metalName: "sum", openclName: "add",
      preNonUniform: true,
      sequence: sumSequence32, identity: 0, execute: +),
    ReductionParams(
      scope: .simd, metalName: "product", openclName: "mul",
      sequence: productSequence32, identity: 1, execute: *),
    ReductionParams(
      scope: .simd, metalName: "min", openclName: "min",
      preNonUniform: true, omitPrefixReduction: true,
      sequence: sumSequence32, identity: 255, execute: min),
    ReductionParams(
      scope: .simd, metalName: "max", openclName: "max",
      preNonUniform: true, omitPrefixReduction: true,
      sequence: sumSequence32, identity: 0, execute: max),
    ReductionParams(
      scope: .simd, metalName: "min", openclName: "min",
      preNonUniform: true, omitPrefixReduction: true,
      sequence: sumSequence32, identity: 255, execute: min),
    ReductionParams(
      scope: .simd, metalName: "and", openclName: "and",
      omitFloat: true, omitPrefixReduction: true,
      sequence: bitwiseSequence32, identity: 0x7F, execute: &),
    ReductionParams(
      scope: .simd, metalName: "or", openclName: "or",
      omitFloat: true, omitPrefixReduction: true,
      sequence: bitwiseSequence32, identity: 0x00, execute: |),
    ReductionParams(
      scope: .simd, metalName: "xor", openclName: "xor",
      omitFloat: true, omitPrefixReduction: true,
      sequence: bitwiseSequence32, identity: 0x00, execute: ^),
  ]
  
  for params in reductionLoopParams {
    if isExtensionThread && params.scope == .quad {
      // This will not compile.
      continue
    }
    
    let totalResult = params.sequence.reduce(params.identity, params.execute)
    if isOtherThread {
      let clusteredFunction = "sub_group_clustered_reduce_\(params.openclName)"
      appendGroup(KernelInvocationGroup(
        device: device,
        body: """
        c[tid] = \(clusteredFunction)(a[tid], \(params.scope.clusterSize));
        """,
        sequenceSize: params.scope.clusterSize,
        omitFloat: params.omitFloat,
        generate: { index, A, B, C in
          A = params.sequence[index]
          B = 0
          C = totalResult
        }
      ))
      
      if (params.scope == .quad || !params.preNonUniform) {
        // This will not compile.
        continue
      }
    }
    
    var baseFunction: String
    if isOtherThread {
      baseFunction = "sub_group_reduce_\(params.openclName)"
    } else if isExtensionThread {
      baseFunction = "sub_group_non_uniform_reduce_\(params.openclName)"
    } else {
      baseFunction = "\(params.scope.metalPrefix)_\(params.metalName)"
    }
    appendGroup(KernelInvocationGroup(
      device: device,
      body: """
    c[tid] = \(baseFunction)(a[tid]);
    """,
      sequenceSize: params.scope.clusterSize,
      omitFloat: params.omitFloat,
      generate: { index, A, B, C in
        A = params.sequence[index]
        B = 0
        C = totalResult
      }
    ))
    
    if params.omitPrefixReduction {
      continue
    }
    
    var partialResult = params.identity
    var exclusiveSequence: [Int] = []
    for i in 0..<params.scope.clusterSize {
      exclusiveSequence.append(partialResult)
      exclusiveSequence[i] = partialResult
      partialResult = params.execute(partialResult, params.sequence[i])
    }
    
    var exclusiveFunction: String
    if isOtherThread {
      exclusiveFunction = "sub_group_scan_exclusive_\(params.openclName)"
    } else if isExtensionThread {
      exclusiveFunction = "sub_group_non_uniform_scan_exclusive_\(params.openclName)"
    } else {
      exclusiveFunction = "\(params.scope.metalPrefix)_prefix_exclusive_\(params.metalName)"
    }
    appendGroup(KernelInvocationGroup(
      device: device,
      body: """
      c[tid] = \(exclusiveFunction)(a[tid]);
      """,
      sequenceSize: params.scope.clusterSize,
      omitFloat: params.omitFloat,
      generate: { index, A, B, C in
        A = params.sequence[index]
        B = 0
        C = exclusiveSequence[index]
      }
    ))
    
    var inclusiveFunction: String
    if isOtherThread {
      inclusiveFunction = "sub_group_scan_inclusive_\(params.openclName)"
    } else if isExtensionThread {
      inclusiveFunction = "sub_group_non_uniform_scan_inclusive_\(params.openclName)"
    } else {
      inclusiveFunction = "\(params.scope.metalPrefix)_prefix_inclusive_\(params.metalName)"
    }
    appendGroup(KernelInvocationGroup(
      device: device,
      body: """
      c[tid] = \(inclusiveFunction)(a[tid]);
      """,
      sequenceSize: params.scope.clusterSize,
      omitFloat: params.omitFloat,
      generate: { index, A, B, C in
        A = params.sequence[index]
        B = 0
        C = params.execute(exclusiveSequence[index], params.sequence[index])
      }
    ))
  }
  
  // Logical Reductions
  
  var validationHandlers: [() -> Void] = []
  
  if isOtherThread {
    struct LogicalParams {
      var scope: Scope
      var name: String
      
      // Whether to treat as clustered if SIMD-scoped.
      var isClustered: Bool = true
      
      // The number sequence to use.
      var sequence: [Int]
      
      // The number to combine the first operand with.
      var identity: Int
      
      // Combine two operands in sequence.
      var execute: (Int, Int) -> Int
    }
    
    let logicalLoopParams = [
      LogicalParams(
        scope: .quad, name: "and",
        sequence: logicalSequence4, identity: 1, execute: logical_and),
      LogicalParams(
        scope: .quad, name: "or",
        sequence: logicalSequence4, identity: 0, execute: logical_or),
      LogicalParams(
        scope: .quad, name: "xor",
        sequence: logicalSequence4, identity: 0, execute: logical_xor),
      
      LogicalParams(
        scope: .simd, name: "and",
        sequence: logicalSequence32, identity: 1, execute: logical_and),
      LogicalParams(
        scope: .simd, name: "or",
        sequence: logicalSequence32, identity: 0, execute: logical_or),
      LogicalParams(
        scope: .simd, name: "xor",
        sequence: logicalSequence32, identity: 0, execute: logical_xor),
      
      LogicalParams(
        scope: .simd, name: "and", isClustered: false,
        sequence: logicalSequence32, identity: 1, execute: logical_and),
      LogicalParams(
        scope: .simd, name: "or", isClustered: false,
        sequence: logicalSequence32, identity: 0, execute: logical_or),
      LogicalParams(
        scope: .simd, name: "xor", isClustered: false,
        sequence: logicalSequence32, identity: 0, execute: logical_xor),
    ]
    
    // Create the shader source code.
    
    func generateUniqueName(_ params: LogicalParams) -> String {
      var prefix = params.scope.metalPrefix
      if params.isClustered {
        prefix += "_clustered"
      }
      return prefix + "_\(params.name)"
    }
    
    func generateBody(_ params: LogicalParams) -> String {
      var functionName: String
      if params.isClustered {
        functionName = "sub_group_clustered_reduce_logical_\(params.name)"
      } else {
        functionName = "sub_group_non_uniform_reduce_logical_\(params.name)"
      }
      
      var clusterArgument: String
      if !params.isClustered {
        clusterArgument = ""
      } else if params.scope == .quad {
        clusterArgument = ", 4"
      } else {
        clusterArgument = ", 32"
      }
      
      return "c[tid] = \(functionName)(a[tid]\(clusterArgument));"
    }
    
    func generateSource() -> String {
      var output: String = ""
      
      if usingSubgroupExtendedTypes {
        output += """
          #define OPENCL_USE_SUBGROUP_EXTENDED_TYPES 1
          
          """
      }
      
      output += """
        \(headerString!)
        
        #define KERNEL __kernel
        #define GLOBAL __global
        #define LOCAL __local
        #define BUFFER_BINDING(index)
        
        """
      
      for params in logicalLoopParams {
        let uniqueName = generateUniqueName(params)
        output.append("""
          KERNEL void vectorOperation_\(uniqueName)(
            GLOBAL int* a BUFFER_BINDING(0),
            GLOBAL int* b BUFFER_BINDING(1),
            GLOBAL int* c BUFFER_BINDING(2)
          #if __METAL__
            ,
            uint tid [[thread_position_in_grid]],
            ushort lane_id [[thread_index_in_simdgroup]]
          ) {
          #else
          ) {
            uint tid = uint(get_global_id(0));
          
            #pragma clang diagnostic push
            #pragma clang diagnostic ignored "-Wunused-variable"
            ushort lane_id = ushort(get_local_id(0)) % 32;
            #pragma clang diagnostic pop
          #endif
            \(generateBody(params))
          }
          
          """)
      }
      
      return output
    }
    
    let source = generateSource()
    let library = device.createLibrary(source: source)
    queue.startCommandBuffer()
    
    for params in logicalLoopParams {
      let uniqueName = generateUniqueName(params)
      let kernel = library.createKernel(name: "vectorOperation_\(uniqueName)")
      queue.setKernel(kernel)
      
      var finalResult = params.identity
      precondition(
        params.scope.clusterSize == params.sequence.count,
        "Cluster size did not match sequence size.")
      let sequenceSize = params.scope.clusterSize
      for i in 0..<sequenceSize {
        finalResult = params.execute(finalResult, params.sequence[i])
      }
      
      var A: [Int32] = .init(repeating: 0, count: sequenceSize)
      var B: [Int32] = .init(repeating: 0, count: sequenceSize)
      var C: [Int32] = .init(repeating: 0, count: sequenceSize)
      for i in 0..<sequenceSize {
        A[i] = Int32(params.sequence[i])
        C[i] = Int32(finalResult)
      }
      while A.count < 32 {
        A.append(contentsOf: A)
        B.append(contentsOf: B)
        C.append(contentsOf: C)
      }
      
      let bufferA = device.createBuffer(A)
      let bufferB = device.createBuffer(B)
      let bufferC = device.createBuffer(
        length: C.count * MemoryLayout<Int32>.stride)
      queue.setBuffer(bufferA, index: 0)
      queue.setBuffer(bufferB, index: 1)
      queue.setBuffer(bufferC, index: 2)
      queue.dispatchThreads(32, threadgroupSize: 32)
      
      validationHandlers.append {
        let expectedC = C
        let actualC: [Int32] = bufferC.getElements()
        let length = expectedC.count * MemoryLayout<Int32>.stride
        if memcmp(actualC, expectedC, length) == 0 {
          return
        }
        
        var error: String?
        for i in 0..<expectedC.count {
          var actual = actualC[i]
          var expected = expectedC[i]
          if _slowPath(memcmp(&actual, &expected, 4) != 0) {
            error = "Element \(i): actual '\(actual)' != expected '\(expected)'"
            break
          }
        }
        
        if let error = error {
          let errorMessage = """
            
            Kernel invocation failed.
            Backend: \(type(of: device))
            Type: int
            Source:\n\(generateBody(params))
            \(error)
            """
          print(errorMessage)
          exit(0)
        }
      }
    }
    
    queue.submitCommandBuffer()
  }
  
  // TODO: Shuffles
  
  // TODO: Ballot
  
  queue.finishCommands()
  for group in allGroups {
    group.validate()
  }
  for handler in validationHandlers {
    handler()
  }
}
