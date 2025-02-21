#include <metal_stdlib>
using namespace metal;

kernel void matrixMultiply(const device float* A [[ buffer(0) ]],
                           const device float* B [[ buffer(1) ]],
                           device float* C [[ buffer(2) ]],
                           uint2 id [[ thread_position_in_grid ]],
                           uint2 group [[ threadgroup_position_in_grid ]]) {
    uint size = 256; // 矩阵大小
    float sum = 0.0;

    for (uint i = 0; i < size; i++) {
        sum += A[id.y * size + i] * B[i * size + id.x];
    }

    C[id.y * size + id.x] = sum;
}
#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float3 position [[attribute(0)]];
};

struct VertexOut {
    float4 position [[position]];
};

vertex VertexOut vertexShader(VertexIn in [[stage_in]]) {
    VertexOut out;
    out.position = float4(in.position, 1.0);
    return out;
}

fragment float4 fragmentShader(VertexOut in [[stage_in]]) {
    return float4(1.0, 0.0, 0.0, 1.0); // 红色
}
