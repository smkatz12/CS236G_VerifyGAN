using NeuralVerification
using NeuralVerification: init_vars, get_bounds, add_set_constraint!, BoundedMixedIntegerLP, encode_network!, compute_output
using LazySets
using LazySets: translate
using DataStructures
using LinearAlgebra
using Gurobi
using Dates
const GRB_ENV = Gurobi.Env()

include("ai2zPQ.jl")
include("tree_utils.jl")

network = read_nnet("./models/full_mlp_best_conv.nnet")
gan_network = read_nnet("./models/mlp_gen_best_conv_rescaled.nnet")
control_network = read_nnet("models/KJ_TaxiNet.nnet")

const TOL = Ref(sqrt(eps()))

""" Discretized Ai2z"""

function discretized_ai2z_bounds(network, lbs, ubs, coeffs; n_per_latent = 10, n_per_state = 1)
    # Ai2z overapproximation through discretization 
    ai2z = Ai2z()
    overestimate = -Inf
    underestimate = Inf

    lbs_disc, ubs_disc = get_bounds(lbs, ubs, n_per_latent, n_per_state)

    for (curr_lbs, curr_ubs) in zip(lbs_disc, ubs_disc)
        # Construct the input set, then propagate forwards to a 
        # zonotope over-approximation of the output set
        input_set = Hyperrectangle(low=curr_lbs, high=curr_ubs)
        reach = forward_network(ai2z, network, input_set)

        # The support function ρ maximizes coeffs^T x for x in reach
        curr_overestimate = ρ(coeffs, reach)
        curr_overestimate ≥ overestimate ? overestimate = curr_overestimate : nothing
        # Maximize the negative and take negative to get minimum
        curr_underestimate = -ρ(-coeffs, reach)
        curr_underestimate ≤ underestimate ? underestimate = curr_underestimate : nothing
    end

    return underestimate, overestimate
end

function get_bounds(lbs, ubs, n_per_latent, n_per_state)
    lbs_disc = []
    ubs_disc = []

    for i = 1:n_per_latent
        for j = 1:n_per_latent
            for k = 1:n_per_state
                for l = 1:n_per_state
                    # Find the upper and lower bounds of your region 
                    lb1 = lbs[1] + (i-1)/n_per_latent * (ubs[1] - lbs[1])
                    lb2 = lbs[2] + (j-1)/n_per_latent * (ubs[2] - lbs[2])
                    lb3 = lbs[3] + (k-1)/n_per_state * (ubs[3] - lbs[3])
                    lb4 = lbs[4] + (l-1)/n_per_state * (ubs[4] - lbs[4])
                    ub1 = lbs[1] + (i)/n_per_latent * (ubs[1] - lbs[1])
                    ub2 = lbs[2] + (j)/n_per_latent * (ubs[2] - lbs[2])
                    ub3 = lbs[3] + (k)/n_per_state * (ubs[3] - lbs[3])
                    ub4 = lbs[4] + (l)/n_per_state * (ubs[4] - lbs[4])
                    cur_lbs = [lb1, lb2, lb3, lb4]
                    cur_ubs = [ub1, ub2, ub3, ub4]

                    push!(lbs_disc, cur_lbs)
                    push!(ubs_disc, cur_ubs)
                end
            end
        end
    end

    return lbs_disc, ubs_disc
end

""" ai2zPQ functions """

function ai2zPQ_bounds(network, lbs, ubs, coeffs)
    # Define functions
    evaluate_objective_max(network, x) = dot(coeffs, compute_output(network, x))
    evaluate_objective_min(network, x) = dot(-coeffs, compute_output(network, x))
    optimize_reach_max(reach) = ρ(coeffs, reach)
    optimize_reach_min(reach) = ρ(-coeffs, reach)

    # Get maximum control
    x, under, value = priority_optimization(network, lbs, ubs, optimize_reach_max, evaluate_objective_max)
    max_control = value
    println(value - under)
    println(x)
    # Get the minimum control
    _, _, value = priority_optimization(network, lbs, ubs, optimize_reach_min, evaluate_objective_min)
    min_control = -value

    return min_control, max_control
end


# Added by Chris for buffer optimization 
using Convex 
using JuMP 
using Mosek 
using MosekTools
import JuMP.MOI.OPTIMAL, JuMP.MOI.INFEASIBLE

compute_objective(network, x, coeffs) = dot(coeffs, compute_output(network, x))

# Approximate the optimum of an input set given by cell passed through 
# the network using the solver. 
function linear_approximate_optimize_cell(solver, network, cell, coeffs)
    reach = forward_network(solver, network, cell)
    return ρ(coeffs, reach)
end



# TODO: Sketchily stealing this method, probably I should push this to NeuralVerification.jl
NeuralVerification.add_set_constraint!(m::Model, set::Zonotope, z::Union{Vector{VariableRef}, Array{GenericAffExpr{Float64, VariableRef}, 1}}) = 
begin
    c = set.center
    G = set.generators
    x = @variable(m, [1:size(G, 2)])
    @constraint(m, x .<= 1.0)
    @constraint(m, x .>= -1.0)
    @constraint(m, z .== c .+ G * x)
end

function mip_linear_opt(network, input_set::Union{Hyperrectangle, Zonotope}, coeffs)
    # Get your bounds
    bounds = NeuralVerification.get_bounds(Ai2z(), network, input_set; before_act=true)

    # Create your model
    model = Model(with_optimizer(() -> Gurobi.Optimizer(GRB_ENV), OutputFlag=0, Threads=8, TimeLimit=10.0))
    z = init_vars(model, network, :z, with_input=true)
    δ = init_vars(model, network, :δ, binary=true)
    # get the pre-activation bounds:
    model[:bounds] = bounds
    model[:before_act] = true

    # Add the input constraint 
    NeuralVerification.add_set_constraint!(model, input_set, first(z))

    # Encode the network as an MIP
    encode_network!(model, network, BoundedMixedIntegerLP())
    @objective(model, Max, coeffs'*last(z))

    optimize!(model)
    if termination_status(model) == OPTIMAL
        return value.(first(z)), coeffs'*value.(last(z))
    else 
        @assert false "Non optimal result"
    end
end

# Test whether the general function works 
function linear_opt_wrapper(network, input_set, coeffs; n_steps=1000, solver=Ai2z(), early_stop=true, stop_freq=200, stop_gap=1e-4, initial_splits=0, verbosity=1)
    evaluate_objective = x -> compute_objective(network, x, coeffs)
    approximate_optimize_cell = cell -> ρ(coeffs, forward_network(solver, network, cell))
    return general_priority_optimization(input_set, approximate_optimize_cell, evaluate_objective;  n_steps = n_steps, solver=solver, early_stop=early_stop, stop_freq=stop_freq, stop_gap=stop_gap, initial_splits=initial_splits, verbosity=verbosity)
end

function ai2zPQ_bounds(network, lbs, ubs, coeffs; n_steps=1000, solver=Ai2z(), early_stop=true, stop_freq=200, stop_gap=1e-4, initial_splits=0, verbosity=1)
    x, lower, upper, steps = linear_opt_wrapper(network, Hyperrectangle(low=lbs, high=ubs), coeffs; n_steps=n_steps, solver=solver, early_stop=early_stop, stop_freq=stop_freq, stop_gap=stop_gap, initial_splits=initial_splits, verbosity=verbosity)
    max_val = upper 
    x, lower, upper, steps = linear_opt_wrapper(network, Hyperrectangle(low=lbs, high=ubs), -coeffs; n_steps=n_steps, solver=solver, early_stop=early_stop, stop_freq=stop_freq, stop_gap=stop_gap, initial_splits=initial_splits, verbosity=verbosity)
    min_val = -upper 
    return min_val, max_val
end

# Have with extra buffer. Assume the buffer can be Minkowski summed with the 
# zonotope and then concretized and passed into the reachability again. 
# this will be true if the buffer is a Hyperrectangle 
function linear_opt_with_buffer(network_one, network_two, lbs, ubs, coeffs, buffer; n_steps=1000, solver=Ai2z(), early_stop=true, stop_freq=200, stop_gap=1e-4, initial_splits=0, verbosity=1)
    evaluate_objective = x ->  compute_objective(network_two, compute_output(network_one, x), coeffs)
    approximate_optimize_cell = cell -> begin
                                                reach_one = forward_network(solver, network_one, cell)
                                                buffered_reach = concretize(reach_one ⊕ buffer)
                                                return ρ(coeffs, forward_network(solver, network_two, buffered_reach)) 
                                        end
    return general_priority_optimization(Hyperrectangle(low=lbs, high=ubs), approximate_optimize_cell, evaluate_objective;  n_steps = n_steps, solver=solver, early_stop=early_stop, stop_freq=stop_freq, stop_gap=stop_gap, initial_splits=initial_splits, verbosity=verbosity)
end

function linear_opt_with_buffer_breakdown(network_one, network_two, lbs, ubs, coeffs, buffer; n_steps=1000, solver=Ai2z(), early_stop=true, stop_freq=10, stop_gap=1e-2, initial_splits=0, verbosity=1)
    #evaluate_objective = x ->  compute_objective(network_two, NeuralVerification.compute_output(network_one, x), coeffs)
    evaluate_objective = x -> begin
                                    out_1 = compute_output(network_one, x)
                                    buffered_output = translate(buffer, out_1)
                                    opt_input, opt_val = mip_linear_opt(network_two, buffered_output, coeffs)
                                    return opt_val
                              end
    approximate_optimize_cell = cell -> begin
                                                reach_one = forward_network(solver, network_one, cell)
                                                buffered_reach = concretize(reach_one ⊕ buffer)
                                                # trying with the mip solver 
                                                opt_input, opt_val = mip_linear_opt(network_two, buffered_reach, coeffs)
                                                return opt_val
                                        end
    return general_priority_optimization(Hyperrectangle(low=lbs, high=ubs), approximate_optimize_cell, evaluate_objective;  n_steps = n_steps, solver=solver, early_stop=early_stop, stop_freq=stop_freq, stop_gap=stop_gap, initial_splits=initial_splits, verbosity=verbosity)
end

function ai2zPQ_bounds_buffered(network_one, network_two, lbs, ubs, coeffs, buffer; n_steps=1000, solver=Ai2z(), early_stop=true, stop_freq=200, stop_gap=1e-4, initial_splits=0, verbosity=1)
    x_max, lower, upper, steps = linear_opt_with_buffer(network_one, network_two, lbs, ubs, coeffs, buffer; n_steps=n_steps, solver=solver, early_stop=early_stop, stop_freq=stop_freq, stop_gap=stop_gap, initial_splits=initial_splits, verbosity=verbosity)
    max_val = upper
    x_min, lower, upper, steps = linear_opt_with_buffer(network_one, network_two, lbs, ubs, -coeffs, buffer; n_steps=n_steps, solver=solver, early_stop=early_stop, stop_freq=stop_freq, stop_gap=stop_gap, initial_splits=initial_splits, verbosity=verbosity)
    min_val = -upper
    return min_val, max_val
end

function ai2zPQ_bounds_buffered_breakdown(network_one, network_two, lbs, ubs, coeffs, buffer; n_steps=1000, solver=Ai2z(), early_stop=true, stop_freq=200, stop_gap=1e-4, initial_splits=200, verbosity=1)
    x_max, lower_max, upper_max, steps = linear_opt_with_buffer_breakdown(network_one, network_two, lbs, ubs, coeffs, buffer; n_steps=n_steps, solver=solver, early_stop=early_stop, stop_freq=stop_freq, stop_gap=stop_gap, initial_splits=initial_splits, verbosity=verbosity)
    max_val = upper_max
    x_min, lower_min, upper_min, steps = linear_opt_with_buffer_breakdown(network_one, network_two, lbs, ubs, -coeffs, buffer; n_steps=n_steps, solver=solver, early_stop=early_stop, stop_freq=stop_freq, stop_gap=stop_gap, initial_splits=initial_splits, verbosity=verbosity)
    min_val = -upper_min
    println("x max: ", x_max)
    println("Interval max: ", [lower_max, upper_max])
    println("x min: ", x_min)
    println("Interval min: ", [lower_min, upper_min])
    return min_val, max_val
end

""" Perform verification on entire tree """

function verify_tree!(tree, network; get_control_bounds = ai2zPQ_bounds, coeffs = [-0.74, -0.44], 
            n_per_latent = 30, n_per_state = 2, latent_bound = 1.0)
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
            verify_lbs = [-latent_bound; -latent_bound; curr_lbs ./ [6.366468343804353, 17.248858791583547]]
            verify_ubs = [latent_bound; latent_bound; curr_ubs ./ [6.366468343804353, 17.248858791583547]]
            
            min_control, max_control = get_control_bounds(network, verify_lbs, verify_ubs, coeffs)
            
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

function verify_tree_buffered!(tree, gan_network, control_network, full_network; coeffs = [-0.74, -0.44], 
     latent_bound = 0.8)
    # Stacks to go through tree
    lb_s = Stack{Vector{Float64}}()
    ub_s = Stack{Vector{Float64}}()
    s = Stack{Union{LEAFNODE, KDNODE}}()

    push!(lb_s, tree.lbs)
    push!(ub_s, tree.ubs)
    push!(s, tree.root_node)

    # Introduce some variables to track progress
    leafs_finished = 0
    total_leaves = length(get_leaves(tree))
    start_time = now()

    while !(isempty(s))
        println(length(s))
        curr = pop!(s)
        curr_lbs = pop!(lb_s)
        curr_ubs = pop!(ub_s)

        if typeof(curr) == LEAFNODE
            if curr.min_control != -Inf || curr.max_control != Inf
                @warn "Skipping leafs because they already have a min or max control"
            else
                # TODO: add if statement if no images don't use the harder verification 
                verify_lbs = [-latent_bound; -latent_bound; curr_lbs ./ [6.366468343804353, 17.248858791583547]]
                verify_ubs = [latent_bound; latent_bound; curr_ubs ./ [6.366468343804353, 17.248858791583547]]
                
                #min_control_linear, max_control_linear = ai2zPQ_bounds(network, verify_lbs, verify_ubs, coeffs)


                println("------Currently doing cell with ", length(curr.images), " images------")
                if length(curr.images) == 0
                    @time min_control, max_control = ai2zPQ_bounds(full_network, verify_lbs, verify_ubs, coeffs; n_steps=10000, stop_gap=1e-1, verbosity=0)
                else
                    @time min_control, max_control = ai2zPQ_bounds_buffered_breakdown(gan_network, control_network, verify_lbs, verify_ubs, coeffs, curr.buffer; n_steps=10000, stop_gap=1e-1, verbosity=0)
                end
                println("buffer: ", [min_control, max_control])

                curr.min_control = min_control
                curr.max_control = max_control
            end 
            leafs_finished = leafs_finished + 1
            percent_done = leafs_finished/total_leaves
            println("Leafs finished: ", leafs_finished, "   ", round(100*percent_done, digits=2), "% done")
            println("ETA: ", (1.0/percent_done - 1.0)*(Dates.value(now() - start_time)) / 1000 / 3600, " hours")
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