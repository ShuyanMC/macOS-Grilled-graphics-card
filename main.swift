import Foundation
import Metal

// 检查GPU是否可用
guard let device = MTLCreateSystemDefaultDevice() else {
    print("Metal is not supported on this device.")
    exit(1)
}

print("Using GPU: \(device.name)")

// 创建命令队列
guard let commandQueue = device.makeCommandQueue() else {
    fatalError("Failed to create command queue.")
}

// 加载Metal计算内核
guard let library = device.makeDefaultLibrary(),
      let function = library.makeFunction(name: "matrixMultiply") else {
    fatalError("Failed to load Metal library or function.")
}

// 创建计算管线描述符
let pipelineStateDescriptor = MTLComputePipelineDescriptor()
pipelineStateDescriptor.computeFunction = function

// 创建计算管线状态
do {
    let result = try device.makeComputePipelineState(descriptor: pipelineStateDescriptor, options: [])
    let pipelineState = result.0  // 解构元组，提取第一个元素

    // 创建一个持续运行的循环
    while true {
        autoreleasepool {
            // 随机生成矩阵大小和计算任务
            let matrixSize = Int.random(in: 15000...15000)  // 随机矩阵大小
            let matrixBytes = matrixSize * matrixSize * MemoryLayout<Float>.size

            // 创建输入矩阵
            let matrixA = [Float](repeating: 1.0, count: matrixSize * matrixSize)
            let matrixB = [Float](repeating: 2.0, count: matrixSize * matrixSize)

            // 创建输出矩阵
            var matrixC = [Float](repeating: 0.0, count: matrixSize * matrixSize)

            // 创建缓冲区
            guard let bufferA = device.makeBuffer(bytes: matrixA, length: matrixBytes, options: []),
                  let bufferB = device.makeBuffer(bytes: matrixB, length: matrixBytes, options: []),
                  let bufferC = device.makeBuffer(length: matrixBytes, options: []) else {
                fatalError("Failed to create buffers.")
            }

            // 创建命令缓冲区
            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let commandEncoder = commandBuffer.makeComputeCommandEncoder() else {
                fatalError("Failed to create command encoder.")
            }

            // 设置计算内核的参数
            commandEncoder.setComputePipelineState(pipelineState)
            commandEncoder.setBuffer(bufferA, offset: 0, index: 0)
            commandEncoder.setBuffer(bufferB, offset: 0, index: 1)
            commandEncoder.setBuffer(bufferC, offset: 0, index: 2)

            // 设置线程组大小
            let threadGroupSize = MTLSize(width: 16, height: 16, depth: 1)
            let threadGroups = MTLSize(width: (matrixSize + threadGroupSize.width - 1) / threadGroupSize.width,
                                       height: (matrixSize + threadGroupSize.height - 1) / threadGroupSize.height,
                                       depth: 1)

            commandEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
            commandEncoder.endEncoding()

            // 提交命令缓冲区
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            // 获取计算结果
            let resultPointer = bufferC.contents().bindMemory(to: Float.self, capacity: matrixSize * matrixSize)
            matrixC = Array(UnsafeBufferPointer(start: resultPointer, count: matrixSize * matrixSize))

            // 输出计算任务到控制台
            print("Matrix Size: \(matrixSize) x \(matrixSize)")
            print("Matrix A (first 10 elements):")
            print(matrixA.prefix(10)) // 打印部分矩阵A
            print("Matrix B (first 10 elements):")
            print(matrixB.prefix(10)) // 打印部分矩阵B
            print("Result Matrix C (first 10 elements):")
            print(matrixC.prefix(10)) // 打印部分结果矩阵C
        }

        // 添加一个简单的延迟，避免过快的循环
        Thread.sleep(forTimeInterval: 0.1)
    }
} catch {
    fatalError("Failed to create compute pipeline state: \(error)")
}
