module TestExamples1DBurgers

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "1d")

@testset "Inviscid Burgers" begin
  @testset "elixir_burgers_basic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_burgers_basic.jl"),
      l2   = [2.967470209082194e-5],
      linf = [0.00016152468882624227])
  end

  @testset "elixir_burgers_basic.jl with flux_godunov" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_burgers_basic.jl"),
      surface_flux = flux_godunov,
      l2   = [2.967494855204592e-5],
      linf = [0.0001615256989919711])
  end

  @testset "elixir_burgers_linear_stability.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_burgers_linear_stability.jl"),
      l2   = [0.5660569881106876],
      linf = [1.9352238038313998])
  end
end

end # module
