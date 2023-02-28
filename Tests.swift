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

// Returns a group of shader permutations for each type.
// Use `auto` keyword to make the body type-generic.
func generateKernel(body: String, types: [String] = defaultTypes) -> String {
  var output: String = """
    #if __METAL__
    #include <metal_stdlib>
    using namespace metal;
    
    #define KERNEL kernel
    #define GLOBAL device
    #define LOCAL threadgroup
    #define BUFFER_BINDING(index) [[buffer(index)]]
    #else
    \(headerString)
    
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
        ushort lane_id = ushort(get_local_id(0) % 32);
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
  
  func setElements<T>(_ bytes: Array<T>)
  
  func getElements<T>() -> Array<T>
}

protocol GPUCommandQueue {
  associatedtype Backend: GPUBackend
  
  // Call this explicitly before encoding new commands.
  func startCommandBuffer()
  
  // Call this before binding any arguments.
  func setKernel(_ kernel: Backend.Kernel)
  
  // Set raw data argument for the kernel.
  func setBytes<T>(_ bytes: Array<T>, index: Int)
  
  // Set API buffer argument for the kernel.
  func setBuffer(_ buffer: Backend.Buffer, index: Int)
  
  // Execute the kernel.
  func dispatchThreads(_ threads: Int, threadgroupSize: Int)
  
  // Call this explicitly before waiting for commands to finish.
  func submitCommandBuffer()
  
  // Halts until most recent command buffer has completed.
  func finishCommands()
}

// MARK: - Metal Types

class MetalBackend {
  
}

class MetalDevice {
  var device: MTLDevice
  
  init(device: MTLDevice) {
    self.device = device
  }
  
  convenience init() {
    self.init(device: MTLCopyAllDevices().first!)
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

class MetalLibrary {
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

class MetalKernel {
  var pipeline: MTLComputePipelineState
  
  init(pipeline: MTLComputePipelineState) {
    self.pipeline = pipeline
  }
}

class MetalBuffer {
  var buffer: MTLBuffer
  
  init(buffer: MTLBuffer) {
    self.buffer = buffer
  }
  
  func setElement<T>(_ bytes: Array<T>) {
    bytes.withUnsafeBufferPointer { pointer in
      let numBytes = pointer.count * MemoryLayout<T>.stride
      memcpy(buffer.contents(), pointer.baseAddress, numBytes)
    }
  }
  
  func getElement<T>() -> Array<T> {
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

class MetalCommandQueue {
  var commandQueue: MTLCommandQueue
  var commandBuffer: MTLCommandBuffer!
  var encoder: MTLComputeCommandEncoder!
  
  init(commandQueue: MTLCommandQueue) {
    self.commandQueue = commandQueue
  }
  
  func startCommandBuffer() {
    commandBuffer = commandQueue.makeCommandBuffer()!
    encoder = commandBuffer.makeComputeCommandEncoder()!
  }
  
  func setKernel(_ kernel: MetalKernel) {
    encoder.setComputePipelineState(kernel.pipeline)
  }
  
  func setBytes<T>(_ bytes: Array<T>, index: Int) {
    let numBytes = bytes.count * MemoryLayout<T>.stride
    encoder.setBytes(bytes, length: numBytes, index: index)
  }
  
  func setBuffer(_ buffer: MetalBuffer, index: Int) {
    encoder.setBuffer(buffer.buffer, offset: 0, index: index)
  }
  
  func dispatchThreads(_ threads: Int, threadgroupSize: Int) {
    
  }
  
  func submitCommandBuffer() {
    encoder.endEncoding()
    commandBuffer.commit()
  }
  
  func finishCommands() {
    commandBuffer.waitUntilCompleted()
  }
}

// MARK: - OpenCL Types

class OpenCLBackend {
  
}

class OpenCLDevice {
  
}

class OpenCLLibrary {
  
}

class OpenCLKernel {
  
}

class OpenCLBuffer {
  
}

class OpenCLCommandQueue {
  
}

// MARK: - Tests
