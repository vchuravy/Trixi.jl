using AMDGPU
using Trixi
using KernelAbstractions

A = AMDGPU.zeros(Float64, 10)
@show typeof(A)
@show Trixi.trixi_backend(A)

EXAMPLES_DIR = joinpath(examples_dir(), "p4est_2d_dgsem")

trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic_gpu.jl"),
              storage_type = ROCArray,
              real_type = Float32)
