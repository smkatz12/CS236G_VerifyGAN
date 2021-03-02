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

include("tree_utils.jl")

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

function verify_tree!(tree, mipverify_network, num_inp, main_solver, tightening_solver)
    # Stacks to go through tree
    lb_s = Stack{Vector{Float64}}()
    ub_s = Stack{Vector{Float64}}()
    s = Stack{Union{LEAFNODE, KDNODE}}()

    push!(lb_s, tree.lbs)
    push!(ub_s, tree.ubs)
    push!(s, tree.root_node)

    while !(isempty(s))
        println(length(s))
        curr = pop!(s)
        curr_lbs = pop!(lb_s)
        curr_ubs = pop!(ub_s)

        if typeof(curr) == LEAFNODE
            verify_lbs = [-1.0; -1.0; curr_lbs ./ [6.366468343804353, 17.248858791583547]]
            verify_ubs = [1.0; 1.0; curr_ubs ./ [6.366468343804353, 17.248858791583547]]
            
            min_control, max_control = get_control_optima(mipverify_network, num_inp, 
                                            verify_lbs, verify_ubs, main_solver, tightening_solver)
            
            curr.min_control = min_control
            curr.max_control = max_control
        else
            # Traverse tree and keep track of bounds
            dim = curr.dim
            split = curr.split
            # Go left, upper bounds will change
            left_ubs = copy(curr_ubs)
            left_ubs[dim] = split

            push!(lb_s, curr_lbs)
            push!(ub_s, left_ubs)
            push!(s, curr.left)

            # Go right, lower bounds will change
            right_lbs = copy(curr_lbs)
            right_lbs[dim] = split
            
            push!(lb_s, right_lbs)
            push!(ub_s, curr_ubs)
            push!(s, curr.right)
        end
    end
end