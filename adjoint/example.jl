using OrdinaryDiffEqLowStorageRK
using OrdinaryDiffEqTsit5
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = -1.0 # minimum coordinate
coordinates_max = 1.0 # maximum coordinate

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 30_000, periodicity = true) # set maximum capacity of tree data structure

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test,
                                    solver;
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
ode = semidiscretize(semi, (0.0, 1.0))

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval = 100)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval = 100,
                                     solution_variables = cons2prim)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl = 1.6)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, Tsit5(),
            ode_default_options()...);

using Enzyme
using SciMLSensitivity

import SciMLStructures

SciMLStructures.isscimlstructure(::SemidiscretizationHyperbolic) = true
SciMLStructures.ismutablescimlstructure(::SemidiscretizationHyperbolic) = true

function SciMLStructures.canonicalize(::SciMLStructures.Tunable,
                                      semi::SemidiscretizationHyperbolic)
    # concatenate all tunable values into a single vector
    buffer = vcat(semi.equations.advection_velocity)

    # repack takes a new vector of the same length as `buffer`, and constructs
    # a new `Parameters` object using the values from the new vector for tunables
    # and retaining old values for other parameters. This is exactly what replace does,
    # so we can use that instead.
    repack = let semi = semi
        function repack(newbuffer)
            return SemidiscretizationHyperbolic(semi.mesh,
                                                LinearScalarAdvectionEquation1D(newbuffer[1]),
                                                semi.initial_condition,
                                                semi.solver;
                                                boundary_conditions = semi.boundary_conditions)
        end
    end
    # the canonicalized vector, the repack function, and a boolean indicating
    # whether the buffer aliases values in the parameter object (here, it doesn't)
    return buffer, repack, false
end

# function SS.replace(::SS.Tunable, p::Parameters, newbuffer)
#     N = length(p.subparams) + length(p.coeffs)
#     @assert length(newbuffer) == N
#     subparams = [SubproblemParameters(newbuffer[i], subpar.q, subpar.r)
#                  for (i, subpar) in enumerate(p.subparams)]
#     coeffs = reshape(view(newbuffer, (length(p.subparams) + 1):length(newbuffer)), size(p.coeffs))
#     return Parameters(subparams, coeffs)
# end

# function SS.replace!(::SS.Tunable, p::Parameters, newbuffer)
#     N = length(p.subparams) + length(p.coeffs)
#     @assert length(newbuffer) == N
#     for (subpar, val) in zip(p.subparams, newbuffer)
#         subpar.p = val
#     end
#     copyto!(p.coeffs, view(newbuffer, (length(p.subparams) + 1):length(newbuffer)))
#     return p
# end

# https://docs.sciml.ai/SciMLStructures/stable/example/

SciMLSensitivity.STACKTRACE_WITH_VJPWARN[] = true

g(u, p, t) = sum(u .^ 2) ./ 2

adjoint_sol = adjoint_sensitivities(sol,
                                    CarpenterKennedy2N54(williamson_condition = false);
                                    g = g,
                                    dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                                    sensealg = InterpolatingAdjoint(; checkpointing = true,
                                                                    autojacvec = EnzymeVJP()))
