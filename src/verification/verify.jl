ENV["JULIA_DEBUG"] = Main # turns on logging (@debug, @info, @warn) for "included" files
ENV["GUROBI_HOME"] = "/home/smkatz/Documents/gurobi902/linux64"
ENV["GRB_LICENSE_FILE"] = "/home/smkatz/Documents/gurobi.lic"

base_path = "../../../../MIPVerifyWrapper/" #ARGS[1]

#=
Find the possible advisories for a given input region
=#

using Pkg
Pkg.activate("/home/smkatz/Documents/MIPVerifyWrapper")

using MIPVerify
using Gurobi
using Interpolations
using Parameters
using JuMP
using GLPKMathProgInterface
using MathProgBase
using CPUTime

#MIPVerify.setloglevel!("notice") # "info", "notice"

include(joinpath(base_path, "src/activation.jl"))
include(joinpath(base_path, "src/network.jl"))
include(joinpath(base_path, "src/util.jl"))
include(joinpath(base_path, "src/nnet_functions.jl"))

include(joinpath(base_path, "src/RunQueryUtils.jl"))

function get_control_optima(mipverify_network, num_inp, lbs, ubs, main_solver, tightening_solver)
    p1 = get_optimization_problem(
          (num_inp,),
          mipverify_network,
          main_solver,
          lower_bounds=lbs,
          upper_bounds=ubs,
          tightening_solver=tightening_solver
          )

    @objective(p1.model, Max, -0.74p1.output_variable[1] - 0.44p1.output_variable[2])
    solve(p1.model)
    max_control = getobjectivevalue(p1.model)

    @objective(p1.model, Min, -0.74p1.output_variable[1] - 0.44p1.output_variable[2])
    solve(p1.model)
    min_control = getobjectivevalue(p1.model)

    return min_control, max_control
end