using NeuralVerification
using LazySets
using DataStructures
using LinearAlgebra

include("tree_utils.jl")

network = read_nnet("../../models/full_msle_uniform.nnet")

const TOL = Ref(sqrt(eps()))

# Approximate the optimum of an input set given by cell passed through 
# the network using the solver. 
function approximate_optimize_cell(solver, network, cell, coeffs)
    reach = forward_network(solver, network, cell)
    return ρ(coeffs, reach)
end

# Split a hyperrectangle into multiple hyperrectangles
function split_cell(cell::Hyperrectangle)
    lbs, ubs = low(cell), high(cell)
    largest_dimension = argmax(ubs .- lbs)
    # have a vector [0, 0, ..., 1/2 largest gap at largest dimension, 0, 0, ..., 0]
    delta = zeros(length(lbs))
    delta[largest_dimension] = 0.5 * (ubs[largest_dimension] - lbs[largest_dimension])
    #delta = elem_basis(largest_dimension, length(lbs)) * 0.5 * (ubs[largest_dimension] - lbs[largest_dimension])
    cell_one = Hyperrectangle(low=lbs, high=(ubs .- delta))
    cell_two = Hyperrectangle(low=(lbs .+ delta), high=ubs)
    return [cell_one, cell_two]
end

function approximate_priority_optimization(network, lbs, ubs, coeffs; n_steps = 1000, solver=Ai2z(), 
                early_stop=false, stop_freq=200, stop_gap=1e-4)
    # Create your queue, then add your original cell 
    cells = PriorityQueue(Base.Order.Reverse) # pop off largest first 
    start_cell = Hyperrectangle(low=lbs, high=ubs)
    start_value = approximate_optimize_cell(solver, network, start_cell, coeffs)
    enqueue!(cells, start_cell, start_value)
    # For n_steps dequeue a cell, split it, and then 
    for i = 1:n_steps
        cell, value = peek(cells) # peek instead of dequeue to get value, is there a better way?
        dequeue!(cells)
        # Early stopping
        if early_stop
            if i % stop_freq == 0
                lower_bound = compute_objective(network, cell.center, coeffs)
                if (value .- lower_bound) <= stop_gap
                    return lower_bound, value, i
                end
            end
        end
        new_cells = split_cell(cell)
        # Enqueue each of the new cells
        for new_cell in new_cells
            # If you've made the max objective cell tiny
            # break (otherwise we end up with zero radius cells)
            if max(radius(new_cell) < TOL[])
                # Return a concrete value and the upper bound from the parent cell
                # that was just dequeued, as it must have higher value than all other cells
                # that were on the queue, and they constitute a tiling of the space
                return compute_objective(network, cell.center, coeffs), value, i 
            end
            enqueue!(cells, new_cell, approximate_optimize_cell(solver, network, new_cell, coeffs))
        end
    end
    # The largest value in our queue is the approximate optimum 
    cell, value = peek(cells)
    return compute_objective(network, cell.center, coeffs), value, n_steps
end

function approximate_optimization(lbs, ubs, coeffs; n_per_latent = 10, n_per_state = 1)
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

function verify_tree!(tree; coeffs = [-0.74, -0.44], n_per_latent = 30, n_per_state = 2)
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
            
            min_control, max_control = approximate_optimization(verify_lbs, verify_ubs, coeffs, 
                                            n_per_latent = n_per_latent, n_per_state = n_per_state)
            
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

function verify_tree_buffer!(tree; coeffs = [-0.74, -0.44], n_per_latent = 30, n_per_state = 2)
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
            
            min_control, max_control = approximate_optimization(verify_lbs, verify_ubs, coeffs, 
                                           n_per_latent = n_per_latent, n_per_state = n_per_state)
            
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