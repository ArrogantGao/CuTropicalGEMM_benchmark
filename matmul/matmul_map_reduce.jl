using CUDA, LinearAlgebra, TropicalNumbers, BenchmarkTools, CSV, DataFrames

begin
    output_file = "matmul_map_reduce.csv"
    mat_size = [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096), (8192, 8192, 8192), (2048, 128, 128), (4 * 2048, 64, 64), (16 * 2048, 32, 32), (64 * 2048, 16, 16), (256 * 2048, 8, 8), (1024 * 2048, 4, 4)]
    Types = [TropicalMaxPlusF32, TropicalMinPlusF32, TropicalMaxMulF32]
    for T in Types
        for (M, N, K) in mat_size
            A = T.(CUDA.rand(Float32, M, K))
            B = T.(CUDA.rand(Float32, K, N))
            C = T.(CUDA.rand(Float32, M, N))
            time_cost = @belapsed CUDA.@sync $(C) + $(A) * $(B)
            tflops = 2 * M * N * K / time_cost / 1e12
            @show M, N, K, tflops

            df = DataFrame(
                type = T,
                MNK = (M, N, K),
                tflops = tflops,
            )
            CSV.write(output_file, df; append=true)
        end
    end
end