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
func generateKernels(
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
}

// TODO: Make common protocol to unify code for Metal and OpenCL
