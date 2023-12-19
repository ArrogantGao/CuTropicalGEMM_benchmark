#!/bin/bash

# touch matmul_map_reduce.csv
# echo 'type,MNK,tflops' >> matmul_map_reduce.csv

# julia --project=@.. matmul_map_reduce.jl

# touch matmul_CuTropicalGEMM.csv
# echo 'type,MNK,tflops' >> matmul_CuTropicalGEMM.csv

# julia --project=@.. matmul_CuTropicalGEMM.jl

touch matmul_GemmKernels.csv
echo 'type,MNK,tflops' >> matmul_GemmKernels.csv

julia --project=/home/xuanzhaogao/code/CuTropicalGEMM_benchmark matmul_GemmKernels.jl