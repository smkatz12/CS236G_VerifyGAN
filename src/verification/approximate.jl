@everywhere using NeuralVerification
@everywhere using NeuralVerification: init_vars, get_bounds, add_set_constraint!, BoundedMixedIntegerLP, encode_network!, compute_output, _ẑᵢ₊₁, TOL
@everywhere using LazySets
@everywhere using LazySets: translate
@everywhere using DataStructures
@everywhere using LinearAlgebra
@everywhere using Gurobi
@everywhere using Dates
@everywhere using Distributed
const GRB_ENV = Gurobi.Env()

@everywhere include(string(@__DIR__, "/ai2zPQ.jl"))
@everywhere include(string(@__DIR__, "/tree_utils.jl"))

network = read_nnet("./models/full_mlp_best_conv.nnet")
gan_network = read_nnet("./models/mlp_gen_best_conv_rescaled.nnet")
control_network = read_nnet("models/KJ_TaxiNet.nnet")

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


function mip_linear_opt_value_only_old(network, input_set::Union{Hyperrectangle, Zonotope}, coeffs)
    # Get your bounds
    bounds = NeuralVerification.get_bounds(Ai2z(), network, input_set; before_act=true)

    # Create your model
    model = Model(with_optimizer(() -> Gurobi.Optimizer(GRB_ENV), OutputFlag=0, Threads=8, TimeLimit=300.0))
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
        return objective_value(model)
    else 
        @assert false "Non optimal result"
    end
end


"""

objective_threshold gives a value above which 
"""
function mip_linear_opt_value_only(network, input_set::Union{Hyperrectangle, Zonotope}, coeffs)
    # Get your bounds
    bounds = NeuralVerification.get_bounds(Ai2z(), network, input_set; before_act=true)

    # Create your model
    model = Model(with_optimizer(() -> Gurobi.Optimizer(GRB_ENV), OutputFlag=0, Threads=1, TimeLimit=600.0))
    z = init_vars(model, network, :z, with_input=true)
    δ = init_vars(model, network, :δ, binary=true)
    # get the pre-activation bounds:
    model[:bounds] = bounds
    model[:before_act] = true

    # Add the input constraint 
    NeuralVerification.add_set_constraint!(model, input_set, first(z))

    # Encode the network as an MIP
    encode_network!(model, network, BoundedMixedIntegerLP())
    
    # Set lower and upper bounds 
    #for the first layer it's special because it has no ẑ
    set_lower_bound.(z[1], low(bounds[1]))
    set_upper_bound.(z[1], high(bounds[1]))
    for i = 2:length(z)-1
        # Set lower and upper bounds for the intermediate layers
        ẑ_i =  _ẑᵢ₊₁(model, i-1)
        z_i = z[i]
        # @constraint(model, ẑ_i .>= low(bounds[i])) These empirically seem to slow it down?
        # @constraint(model, ẑ_i .<= high(bounds[i]))
        z_low = max.(low(bounds[i]), 0.0)
        z_high = max.(high(bounds[i]), 0.0)
        set_lower_bound.(z_i, z_low)
        set_upper_bound.(model[:z][i], z_high)
    end
    # Set lower and upper bounds for the last layer special because 
    # it has no ReLU
    set_lower_bound.(z[end], low(bounds[end]))
    set_upper_bound.(z[end], high(bounds[end]))
    
    # Set the objective 
    @objective(model, Max, coeffs'*last(z))

    optimize!(model)
    if termination_status(model) == OPTIMAL
        return objective_value(model)
    elseif termination_status(model) == INFEASIBLE
        @warn "Infeasible result, did you have an output threshold? If not, then it should never return infeasible"
        return maximize ? -Inf : Inf  
    else
        @assert false "Non optimal result"
    end
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
                                    opt_val = mip_linear_opt_value_only(network_two, buffered_output, coeffs)
                                    return opt_val
                              end
    approximate_optimize_cell = cell -> begin
                                                reach_one = forward_network(solver, network_one, cell)
                                                buffered_reach = concretize(reach_one ⊕ buffer)
                                                # trying with the mip solver 
                                                opt_val = mip_linear_opt_value_only(network_two, buffered_reach, coeffs)
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

function verify_tree_buffered_parallel!(tree, gan_network, control_network, full_network; coeffs = [-0.74, -0.44], latent_bound = 0.8)
    leaves, lbs, ubs = get_leaves_and_bounds(tree, latent_bound)
    # TODO: Remove this, temporary for testing with just a few queries
    n = 20
    leaves = leaves[1:n]
    lbs = lbs[1:n]
    ubs = ubs[1:n]
    println("leaf 1 min, max before: ", [leaves[1].min_control, leaves[1].max_control])

    # Compute the appropriate controls 
    # Introduce some variables to track progress
    leafs_finished = 0
    total_leaves = length(get_leaves(tree))
    start_time = now()    
    
    tree_controls_rc = Dict()
    tree_controls = Dict()

    for (i, (leaf, lb, ub)) in enumerate(zip(leaves, lbs, ubs))
        verify_lbs = [lb[1], lb[2], (lb[3:4] ./ [6.366468343804353, 17.248858791583547])...]
        verify_ubs = [ub[1], ub[2], (ub[3:4] ./ [6.366468343804353, 17.248858791583547])...]
        println("lbs, ubs normalized: ", verify_lbs, verify_ubs)
        if length(leaf.images) == 0
            tree_controls_rc[i] = remotecall(ai2zPQ_bounds, mod(i-1, nprocs()-1) + 2, full_network, verify_lbs, verify_ubs, coeffs; n_steps=10000, stop_gap=1e-1, verbosity=0)
        else
            tree_controls_rc[i] = remotecall(ai2zPQ_bounds_buffered_breakdown, mod(i-1, nprocs()-1) + 2, gan_network, control_network, verify_lbs, verify_ubs, coeffs, leaf.buffer; n_steps=10000, stop_gap=1e-1, verbosity=0)
        end
    end

    for (i, (leaf, lb, ub)) in enumerate(zip(leaves, lbs, ubs))
        println("Fetch loop ", i)
        tree_controls[i] = fetch(tree_controls_rc[i])
    end

    for (i, (leaf, lb, ub)) in enumerate(zip(leaves, lbs, ubs))
        leaf.min_control = tree_controls[i][1]
        leaf.max_control = tree_controls[i][2]
    end
    # lk = ReentrantLock()
    # println("Start of mapping function")

    # Threads.@threads for (leaf, lb, ub) in collect(zip(leaves, lbs, ubs))
    #     println("thread id: ", Threads.threadid())
    
    # verify_lbs = [lb[1], lb[2], (lb[3:4] ./ [6.366468343804353, 17.248858791583547])...]
    # verify_ubs = [ub[1], ub[2], (ub[3:4] ./ [6.366468343804353, 17.248858791583547])...]

    #     println("----Starting query with ", length(leaf.images), " images----")
    #     if length(leaf.images) == 0
    #         @time min_control, max_control = ai2zPQ_bounds(full_network, verify_lbs, verify_ubs, coeffs; n_steps=10000, stop_gap=1e-1, verbosity=0)
    #     else
    #         @time min_control, max_control = ai2zPQ_bounds_buffered_breakdown(gan_network, control_network, verify_lbs, verify_ubs, coeffs, leaf.buffer; n_steps=10000, stop_gap=1e-1, verbosity=0)
    #     end

    #     min_control, max_control = -10000.0, 10000.0    
    #     # Track progress
    #     lock(lk)
    #     try
    #         leafs_finished = leafs_finished + 1
    #         percent_done = leafs_finished/total_leaves
    #         println("Leafs finished: ", leafs_finished, "   ", round(100*percent_done, digits=2), "% done")
    #         println("ETA: ", (1.0/percent_done - 1.0)*(Dates.value(now() - start_time)) / 1000 / 3600, " hours")
            
    #         leaf.min_control = min_control 
    #         leaf.max_control = max_control 
    #     finally 
    #         unlock(lk)
    #     end 
    # end

    # Threads.@threads for (leaf, lb, ub) in collect(zip(leaves, lbs, ubs))
    #     println("thread id: ", Threads.threadid())
    
    #     verify_lbs = [lb[1], lb[2], (lb[3:4] ./ [6.366468343804353, 17.248858791583547])...]
    #     verify_ubs = [ub[1], ub[2], (ub[3:4] ./ [6.366468343804353, 17.248858791583547])...]
    #     println("----Starting query with ", length(leaf.images), " images----")
    #     if length(leaf.images) == 0
    #         @time min_control, max_control = ai2zPQ_bounds(full_network, verify_lbs, verify_ubs, coeffs; n_steps=10000, stop_gap=1e-1, verbosity=0)
    #     else
    #         @time min_control, max_control = ai2zPQ_bounds_buffered_breakdown(gan_network, control_network, verify_lbs, verify_ubs, coeffs, leaf.buffer; n_steps=10000, stop_gap=1e-1, verbosity=0)
    #     end

    #     # Track progress
    #     lock(lk) do
    #         leafs_finished = leafs_finished + 1
    #         percent_done = leafs_finished/total_leaves
    #         println("Leafs finished: ", leafs_finished, "   ", round(100*percent_done, digits=2), "% done")
    #         println("ETA: ", (1.0/percent_done - 1.0)*(Dates.value(now() - start_time)) / 1000 / 3600, " hours")
            
    #         leaf.min_control = min_control 
    #         leaf.max_control = max_control 
    #     end 
    # end

    # println("leaf 1 min, max after: ", [leaves[1].min_control, leaves[1].max_control])

    # leaf_to_controls = 
    # (leaf::LEAFNODE, lb::Array{Float64, 1}, ub::Array{Float64, 1}) ->
    # begin
    #     println("Start of mapping function")
    #     verify_lbs = [lb[1], lb[2], (lb[3:4] ./ [6.366468343804353, 17.248858791583547])...]
    #     verify_ubs = [ub[1], ub[2], (ub[3:4] ./ [6.366468343804353, 17.248858791583547])...]
    #     if length(leaf.images) == 0
    #         @time min_control, max_control = ai2zPQ_bounds(full_network, verify_lbs, verify_ubs, coeffs; n_steps=10000, stop_gap=1e-1, verbosity=0)
    #     else
    #         @time min_control, max_control = ai2zPQ_bounds_buffered_breakdown(gan_network, control_network, verify_lbs, verify_ubs, coeffs, leaf.buffer; n_steps=10000, stop_gap=1e-1, verbosity=0)
    #     end

    #     # Track progress
    #     leafs_finished = leafs_finished + 1
    #     percent_done = leafs_finished/total_leaves
    #     println("Leafs finished: ", leafs_finished, "   ", round(100*percent_done, digits=2), "% done")
    #     println("ETA: ", (1.0/percent_done - 1.0)*(Dates.value(now() - start_time)) / 1000 / 3600, " hours")

    #     return min_control, max_control
    # end

    # # Map each leaf to its lower and upper bound on its controls 
    # println("Starting parallel map")
    # controls = pmap(leaf_to_controls, leaves, lbs, ubs)

    # # Update the leaves with their appropriate controls 
    # for (i, leaf) in enumerate(leaves)
    #     leaf.min_control = controls[i][1]
    #     leaf.max_control = controls[i][2]
    # end
end

function verify_tree_buffered!(tree, gan_network, control_network, full_network; coeffs = [-0.74, -0.44], latent_bound = 0.8)
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
            if false && (curr.min_control != -Inf || curr.max_control != Inf)
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