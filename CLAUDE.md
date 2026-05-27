# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running Tests

Tests are partitioned by mesh/solver type and controlled via the `TRIXI_TEST` environment variable:

```bash
# Default: run threaded tests (fast, good coverage)
julia --threads=3 --check-bounds=yes test/runtests.jl

# Run a specific test partition
TRIXI_TEST=tree_part1 julia --threads=3 --check-bounds=yes test/runtests.jl

# Run MPI tests (spawns subprocess with mpiexec)
TRIXI_TEST=mpi julia --threads=1 --check-bounds=yes test/runtests.jl

# Run all tests
TRIXI_TEST=all julia --threads=3 --check-bounds=yes test/runtests.jl
```

Valid `TRIXI_TEST` values: `threaded`, `mpi`, `tree_part1`–`tree_part6`, `structured`, `p4est_part1`, `p4est_part2`, `t8code_part1`, `t8code_part2`, `unstructured_dgmulti`, `parabolic_part1`, `parabolic_part2`, `misc_part1`, `misc_part2`, `performance_specializations`, `kernelabstractions`, `all`.

To run a single elixir interactively (the pattern tests use):
```julia
julia> using Trixi
julia> trixi_include("examples/tree_1d_dgsem/elixir_advection_basic.jl")
```

Tests use `@test_trixi_include` which runs an elixir via `trixi_include` and checks solution norms against stored reference values.

### Code Formatting

```bash
julia -e 'using JuliaFormatter; format(".")'
```

Style: SciML (`style = "sciml"` in `.JuliaFormatter.toml`), with `yas_style_nesting = true` and `align_struct_field = true`. CI checks formatting via `FormatCheck.yml`.

## Architecture

### Core Pipeline

The simulation pipeline follows this composition:

```
Equations + Mesh + Solver → SemidiscretizationHyperbolic → ODEProblem → solve()
```

1. **Equations** (`src/equations/`) — define physics: fluxes, conversions, boundary conditions. Each equation type implements a dispatch-based interface (`flux`, `cons2prim`, `entropy`, etc.).

2. **Mesh** (`src/meshes/`) — `TreeMesh` (quadtree/octree, AMR-capable), `StructuredMesh`, `P4estMesh` (scalable parallel AMR via p4est), `T8codeMesh`, `UnstructuredMesh2D`, `DGMultiMesh`.

3. **Solver** (`src/solvers/`) — primarily DGSEM (Discontinuous Galerkin Spectral Element Method). The solver subdirectories are organized as `dgsem_{meshtype}/`. `DGMulti` handles unstructured/simplex meshes via StartUpDG.jl.

4. **Semidiscretization** (`src/semidiscretization/`) — combines mesh, equations, and solver into a spatial discretization. `semidiscretize(semi, tspan)` returns an `ODEProblem` for OrdinaryDiffEq.jl.

5. **Time integration** — primarily via OrdinaryDiffEq.jl. Custom low-storage RK schemes in `src/time_integration/`. MPI-distributed norm for error control is passed as `internalnorm = ode_norm` (see `src/auxiliary/mpi.jl`).

### State Vector Convention

Solution state is stored in two representations:
- **`u_ode`**: a flat 1D vector used by the ODE integrator (length = `nvariables × nnodes^ndims × nelements`)
- **`u`**: a reshaped multi-dimensional array obtained via `wrap_array(u_ode, mesh, equations, solver, cache)`, used inside `rhs!` and callbacks

`wrap_array` uses `unsafe_wrap` over the same memory (zero-copy). All interior Trixi code operates on the wrapped form. The pattern in every `rhs!` is:
```julia
u  = wrap_array(u_ode,  mesh, equations, solver, cache)
du = wrap_array(du_ode, mesh, equations, solver, cache)
```

### Performance Conventions

All compute-intensive files wrap code in `@muladd begin ... end` for fused multiply-add optimization. This is controlled by `MuladdMacro.jl` and is the standard pattern throughout the codebase.

Threading uses `@threaded` (defined in `src/auxiliary/auxiliary.jl`), which dispatches to either Polyester.jl's `@batch` or Julia's `Threads.@threads` based on the `_PREFERENCE_THREADING` preference (default: `:polyester`).

Loop vectorization uses `@turbo` from LoopVectorization.jl, conditional on `_PREFERENCE_LOOPVECTORIZATION` (default: `true`). When disabled, `mock_turbo.jl` provides a no-op fallback.

Preferences are set at load time from `Preferences.jl`:
- `backend`: threading backend (`"polyester"` or `"threads"`)
- `loop_vectorization`: whether to use `@turbo` (`true`/`false`)
- `sqrt`, `log`: NaN-safe math variants

### MPI Parallelism

MPI support is initialized via `init_mpi()` and global state tracked in `MPI_RANK[]`, `MPI_SIZE[]`, etc. The `ode_norm` function in `src/auxiliary/mpi.jl` computes a globally-reduced L2 norm across ranks, required for error-based time step control with MPI.

Ghost layer communication is handled per-mesh in `dgsem_p4est/` and `dgsem_t8code/`.

### Extensions

GPU support is loaded conditionally via Julia package extensions in `ext/`:
- `TrixiCUDAExt` — CUDA GPU support
- `TrixiAMDGPUExt` — AMD GPU support
- `TrixiMakieExt` — visualization

### Key Files

- `src/Trixi.jl` — module entry: imports, preferences, file includes, exports
- `src/basic_types.jl` — abstract type hierarchy (`AbstractSemidiscretization`, `AbstractEquations`, `AbstractMesh`, etc.)
- `src/auxiliary/mpi.jl` — MPI utilities, `ode_norm`, `ode_unstable_check`
- `src/solvers/dg.jl` — DG solver base: `wrap_array`, cache creation, solver interface
- `src/semidiscretization/semidiscretization.jl` — `semidiscretize`, `integrate`, `calc_error_norms`

### Elixir Pattern

Each example (`examples/`) is a self-contained "elixir" script that defines a complete simulation. Tests run elixirs via `trixi_include` and compare L2/Linf solution norms. When adding new functionality, add a corresponding elixir and test with `@test_trixi_include`.
