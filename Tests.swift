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
    fatalError("Could not find locate header in Xcode app bundle.")
  }
  headerURL = _headerURL
}
