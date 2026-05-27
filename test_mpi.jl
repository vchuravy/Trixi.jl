module TestExamplesMPI

using Trixi

const EXAMPLES_DIR = pkgdir(Trixi, "examples")

begin
    #trixi_include(joinpath(EXAMPLES_DIR, "tree_2d_dgsem/elixir_advection_basic.jl"))
end # TreeMesh MPI

using Profile
using PProf

begin
    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR, "p4est_3d_dgsem/elixir_advection_amr.jl"))

    t = sol.t[end]
    u_ode = sol.u[end]
    du_ode = similar(u_ode)
    Profile.Allocs.@profile sample_rate=1.0 Trixi.rhs!(du_ode, u_ode, semi, t)
end

using MPI
rank = MPI.Comm_rank(MPI.COMM_WORLD)
PProf.Allocs.pprof(web = false, out = "alloc-mpi$(rank).pb.gz")

end # module
