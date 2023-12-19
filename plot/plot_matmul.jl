using Plots, CSV, DataFrames

df_CuTropicalGEMM = CSV.read("matmul/matmul_CuTropicalGEMM.csv", DataFrame)
df_GemmKernels = CSV.read("matmul/matmul_GemmKernels.csv", DataFrame)
df_mapreduce = CSV.read("matmul/matmul_map_reduce.csv", DataFrame)

begin
    n = 4
    mat_size = df_CuTropicalGEMM."MNK"[1:n]
    tflops_CuTropicalGEMM = df_CuTropicalGEMM."tflops"[1:n]
    tflops_GemmKernels = df_GemmKernels."tflops"[1:n]
    tflops_mapreduce = df_mapreduce."tflops"[1:n]

    x_array = [1:n...]

    fig_matmul_square = plot(size = [1000, 800], legendfontsize = 12, tickfontsize = 12, guidefontsize = 12, legend = :topleft, xlabel = "Matrix Size", ylabel = "TFLOPS", xticks = (1:n, mat_size), xrotation = 45, title = "Tropical Matrix Multiplication Performance (FP32), Square", margin = 13Plots.mm, ylim = [-0.5, 20])

    scatter!(x_array, tflops_mapreduce, label="mapreduce", marker=:diamond, markersize=8, color=:green, legend=:topleft)
    scatter!(x_array, tflops_GemmKernels, label="GemmKernels FPUOp", marker=:square, markersize=8, color=:blue, legend=:topleft)
    scatter!(x_array, tflops_CuTropicalGEMM, label="CuTropicalGEMM", marker=:circle, markersize=8, color=:red, legend=:topleft)

    savefig(fig_matmul_square, "images/matmul_benchmark_square.png")
end

begin
    n = 6
    mat_size = df_CuTropicalGEMM."MNK"[5:10]
    tflops_CuTropicalGEMM = df_CuTropicalGEMM."tflops"[5:10]
    tflops_GemmKernels = df_GemmKernels."tflops"[5:10]
    tflops_mapreduce = df_mapreduce."tflops"[5:10]

    x_array = [1:n...]

    fig_matmul_square = plot(size = [1000, 800], legendfontsize = 12, tickfontsize = 12, guidefontsize = 12, legend = :topleft, xlabel = "Matrix Size", ylabel = "TFLOPS", xticks = (1:n, mat_size), xrotation = 45, title = "Tropical Matrix Multiplication Performance (FP32), Thin", margin = 13Plots.mm, ylim = [-0, 4])

    scatter!(x_array, tflops_mapreduce, label="mapreduce", marker=:diamond, markersize=8, color=:green, legend=:topleft)
    scatter!(x_array, tflops_GemmKernels, label="GemmKernels FPUOp", marker=:square, markersize=8, color=:blue, legend=:topleft)
    scatter!(x_array, tflops_CuTropicalGEMM, label="CuTropicalGEMM", marker=:circle, markersize=8, color=:red, legend=:topleft)

    savefig(fig_matmul_square, "images/matmul_benchmark_thin.png")
end