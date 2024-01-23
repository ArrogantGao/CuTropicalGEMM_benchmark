using CSV, DataFrames, GLMakie

df_gemm = CSV.read("cutropicalgemm_FP64.csv", DataFrame)
df_mapreduce = CSV.read("mapreduce_FP64.csv", DataFrame)

df_gemm.M = log2.(df_gemm.M)
df_gemm.N = log2.(df_gemm.N)
df_gemm.K = log2.(df_gemm.K)

df_mapreduce.M = log2.(df_mapreduce.M)
df_mapreduce.N = log2.(df_mapreduce.N)
df_mapreduce.K = log2.(df_mapreduce.K)

df = DataFrame(M = df_gemm.M, N = df_gemm.N, K = df_gemm.K, gemm = df_gemm.t, mapreduce = df_mapreduce.t, compare = df_mapreduce.t ./ df_gemm.t)

points = Vector{Point3f}()
turbo = Vector{Float64}()
for i in 1:length(df.M)
    if df.compare[i] > 1
        push!(points, Point3f(df.M[i], df.N[i], df.K[i]))
        push!(turbo, df.compare[i])
    end
end

f = Figure(size = (900, 650))
ax, hm = scatter(f[1,1], points, color = log10.(turbo), colormap = :Spectral)
Colorbar(f[1, 2], hm)