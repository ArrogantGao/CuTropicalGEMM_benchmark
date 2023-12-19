using CUDA, GemmKernels, LinearAlgebra, CSV, DataFrames
using BenchmarkTools

function GemmKernels_TropicalGEMM(M, N, K)

    A = CUDA.rand(Float32, M, K)
    B = CUDA.rand(Float32, K, N)
    C = CUDA.rand(Float32, M, N)

    # result of tuning
    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 32
    OP_M = 16
    OP_N = 4
    OP_K = 4
    OP_MB = 8
    OP_NB = 4
    OP_KB = 1
    kernel = Kernel.matmul_singlestage

    # pow2-sized, 128-bit aligned inputs, so we can use aligned layouts.
    # we don't have transposed inputs, so everything is column major.
    # @assert stride(A, 2) % 16 == 0
    global_a_layout = Layout.ColMajor{eltype(A)}
    # @assert stride(B, 2) % 16 == 0
    global_b_layout = Layout.ColMajor{eltype(B)}
    # we want to do a simple C = A * B, so no need to load C first.
    global_c_layout = Layout.ColMajor{eltype(C)}
    # @assert stride(C, 2) % 16 == 0
    global_d_layout = Layout.ColMajor{eltype(C)}

    # shared layouts are similar.
    # the frequently-accessed a/b shmems are padded to avoid bank conflicts.
    shared_a_layout = Layout.Padded{Layout.UnsafeAlignedColMajor{eltype(A)}, 8}
    shared_b_layout = Layout.Padded{Layout.UnsafeAlignedColMajor{eltype(B)}, 8}
    shared_c_layout = shared_d_layout = Layout.UnsafeAlignedColMajor{eltype(C)}

    # we use the tropical FPU operator
    compute_type = promote_type(eltype(A), eltype(B))
    operator = Operator.TropicalFPUOp{OP_M, OP_N, OP_K, OP_MB, OP_NB, OP_KB,
                                        compute_type, eltype(C)}

    # the block shape is the result of tuning
    block_shape = (M = BLOCK_M, N = BLOCK_N, K = BLOCK_K)
    # @assert M % block_shape.M == 0
    # @assert N % block_shape.N == 0
    # @assert K % block_shape.K == 0

    conf = GemmKernels.get_config(;
        gemm_shape = (M = M, N = N, K = K),
        block_shape,
        operator,

        global_a_layout, global_b_layout, global_c_layout, global_d_layout,
        shared_a_layout, shared_b_layout, shared_c_layout, shared_d_layout,

        is_a_col_major = true,
        is_b_col_major = true
    )

    time_cost = @belapsed CUDA.@sync GemmKernels.matmul($conf, $A, $B, $C, $C; kernel=$kernel)

    CUDA.unsafe_free!(A)
    CUDA.unsafe_free!(B)
    CUDA.unsafe_free!(C)

    return M * N * K * 2 / time_cost / 1e12
end

begin
    output_file = "matmul_GemmKernels.csv"

    mat_size = [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096), (8192, 8192, 8192), (2048, 128, 128), (4 * 2048, 64, 64), (16 * 2048, 32, 32), (64 * 2048, 16, 16), (256 * 2048, 8, 8), (1024 * 2048, 4, 4)]
    for (M, N, K) in mat_size
        tflops = GemmKernels_TropicalGEMM(M, N, K)
        @show M, N, K, tflops

        df = DataFrame(
                type = Float32,
                MNK = (M, N, K),
                tflops = tflops,
            )
        CSV.write(output_file, df; append=true)
    end
end