using Convex 
using JuMP 
using Mosek 
using MosekTools
import JuMP.MOI.OPTIMAL, JuMP.MOI.INFEASIBLE
const TOL = Ref(sqrt(eps()))

compute_objective(network, x, coeffs) = dot(coeffs, NeuralVerification.compute_output(network, x))

function fgsm(network, x, lbs, ubs, coeffs; step_size=0.1)
    grad_full = NeuralVerification.get_gradient(network, x)
    grad = grad_full' * coeffs
    return clamp.(x + grad * step_size, lbs, ubs)
end

function pgd(network, x, lbs, ubs, coeffs; step_size=0.01, iterations=600)
    cur_x = x
    for i = 1:iterations
        cur_x = fgsm(network, cur_x, lbs, ubs, coeffs; step_size=step_size)
    end
    return cur_x
end

function repeated_pgd(network, x, lbs, ubs, coeffs; step_size=0.01, pgd_iterations=25, samples=50)
    best_x = x
    best_y = compute_objective(network, x, coeffs)
    for i = 1:samples
        rand_x = rand(length(x)) .* (ubs - lbs) .+ lbs
        cur_x = pgd(network, rand_x, lbs, ubs, coeffs; step_size=step_size, iterations=pgd_iterations)
        y = compute_objective(network, cur_x, coeffs)
        if y > best_y
            best_x, best_y = cur_x, y
        end
    end
    return best_x
end

elem_basis(i, n) = [k == i ? 1.0 : 0.0 for k in 1:n]

function hookes_jeeves(network, x, lbs, ubs, coeffs, α, ϵ, γ=0.5)
    f(in) = compute_objective(network, in, coeffs)
    y, n = f(x), length(x)
    while α > ϵ
        improved = false 
        x_best, y_best = x, y
        for i in 1:n 
            for sgn in (-1, 1)
                x_prime = clamp.(x + sgn*α*elem_basis(i, n), lbs, ubs)
                y_prime = f(x_prime)
                if y_prime > y_best
                    x_best, y_best, improved = x_prime, y_prime, true
                end
            end
        end
        x, y = x_best, y_best
        
        if !improved 
            α *= γ
        end
    end
    return x
end

# Approximate the optimum of an input set given by cell passed through 
# the network using the solver. 
function linear_approximate_optimize_cell(solver, network, cell, coeffs)
    reach = forward_network(solver, network, cell)
    return ρ(coeffs, reach)
end

# Split a hyperrectangle into multiple hyperrectangles
function split_cell(cell::Hyperrectangle)
    lbs, ubs = low(cell), high(cell)
    largest_dimension = argmax(ubs .- lbs)
    # have a vector [0, 0, ..., 1/2 largest gap at largest dimension, 0, 0, ..., 0]
    delta = elem_basis(largest_dimension, length(lbs)) * 0.5 * (ubs[largest_dimension] - lbs[largest_dimension])
    cell_one = Hyperrectangle(low=lbs, high=(ubs .- delta))
    cell_two = Hyperrectangle(low=(lbs .+ delta), high=ubs)
    return [cell_one, cell_two]
end


# Test whether the general function works 
function linear_opt_wrapper(network, lbs, ubs, coeffs; n_steps=1000, solver=Ai2z(), early_stop=true, stop_freq=200, stop_gap=1e-4, initial_splits=0)
    evaluate_objective = x -> compute_objective(network, x, coeffs)
    approximate_optimize_cell = cell -> ρ(coeffs, forward_network(solver, network, cell))
    return general_priority_optimization(lbs, ubs, approximate_optimize_cell, evaluate_objective;  n_steps = n_steps, solver=solver, early_stop=early_stop, stop_freq=stop_freq, stop_gap=stop_gap, initial_splits=initial_splits)
end

function max_min_linear(network, lbs, ubs, coeffs; n_steps=1000, solver=Ai2z(), early_stop=true, stop_freq=200, stop_gap=1e-4, initial_splits=0)
    x, lower, upper, steps = linear_opt_wrapper(network, lbs, ubs, coeffs; n_steps=n_steps, solver=solver, early_stop=early_stop, stop_freq=stop_freq, stop_gap=stop_gap, initial_splits=initial_splits)
    max_val = upper 
    x, lower, upper, steps = linear_opt_wrapper(network, lbs, ubs, -coeffs; n_steps=n_steps, solver=solver, early_stop=early_stop, stop_freq=stop_freq, stop_gap=stop_gap, initial_splits=initial_splits)
    min_val = -upper 
    return min_val, max_val
end
# Have with extra buffer. Assume the buffer can be Minkowski summed with the 
# zonotope and then concretized and passed into the reachability again. 
# this will be true if the buffer is a Hyperrectangle 
function linear_opt_with_buffer(network_one, network_two, lbs, ubs, coeffs, buffer; n_steps=1000, solver=Ai2z(), early_stop=true, stop_freq=200, stop_gap=1e-4, initial_splits=0)
    evaluate_objective = x ->  compute_objective(network_two, NeuralVerification.compute_output(network_one, x), coeffs)
    approximate_optimize_cell = cell -> begin
                                                reach_one = forward_network(solver, network_one, cell)
                                                buffered_reach = concretize(reach_one ⊕ buffer)
                                                return ρ(coeffs, forward_network(solver, network_two, buffered_reach)) 
                                        end
    return general_priority_optimization(lbs, ubs, approximate_optimize_cell, evaluate_objective;  n_steps = n_steps, solver=solver, early_stop=early_stop, stop_freq=stop_freq, stop_gap=stop_gap, initial_splits=initial_splits)
end

function max_min_with_buffer(network_one, network_two, lbs, ubs, coeffs, buffer; n_steps=1000, solver=Ai2z(), early_stop=true, stop_freq=200, stop_gap=1e-4, initial_splits=0)
    x, lower, upper, steps = linear_opt_with_buffer(network_one, network_two, lbs, ubs, coeffs, buffer; n_steps=n_steps, solver=solver, early_stop=early_stop, stop_freq=stop_freq, stop_gap=stop_gap, initial_splits=initial_splits)
    max_val = upper
    x, lower, upper, steps = linear_opt_with_buffer(network_one, network_two, lbs, ubs, -coeffs, buffer; n_steps=n_steps, solver=solver, early_stop=early_stop, stop_freq=stop_freq, stop_gap=stop_gap, initial_splits=initial_splits)
    min_val = -upper
    return min_val, max_val
end

# Solve the distance for an arbitrary p-norm 
function dist_to_zonotope_p(reach, point; p = 2)
    G = reach.generators
    c = reach.center
    n, m = size(G)
    x = Variable(m)
    obj = norm(G * x + c - point, p)
    prob = minimize(obj, [x <= 1.0, x >= -1.0])
    solve!(prob, Mosek.Optimizer(LOG=0))
    @assert prob.status == OPTIMAL "Solve must result in optimal status"
    return prob.optval
end

function dist_to_zonotope_lp(reach, point)
    model = Model(GLPK.Optimizer)
    G = reach.generators
    c = reach.center
    n, m = size(G)
    x = @variable(model, [1:m])
    t = @variable(model)
    z = G * x + c
    @constraint(model, x .>= -1.0)
    @constraint(model, x .<= 1.0)
    # Encode the L_infty norm
    @constraint(model, z - point .<= t)
    @constraint(model, point - z .<= t)
    @objective(model, Min, t)

    optimize!(model)
    @assert termination_status(model) == OPTIMAL "Solve must result in optimal status"
    return value(t) # should this add a TOL[] be here?
end

# Check whether a point is included in the output of a network
# Right now this finds the negative distance of the point to the network:
#  inf_x,y ||y_0 - y||_p s.t. y = network(x), lb <= x <= ub. It will actually solve 
# max_x,y -||y_0 - y||_p s.t. y = network(x), lb <= x <= ub, so the return value is negative of what you ight expect   
function inclusion_wrapper(network, lbs, ubs, y₀, p; n_steps = 1000, solver=Ai2z(), early_stop=true, stop_freq=200, stop_gap=1e-4, initial_splits=0)
    evaluate_objective = x -> -norm(y₀ - NeuralVerification.compute_output(network, x), p)
    # begin
    #     if y₀ ∈ reach
    #         direction = y₀ - reach.center
    #         max_in_dir = ρ(direction, reach)
    #         return (max_in_dir - direction'*y₀)/max_in_dir
    #     else
    #         return val = -1.0
    #     end
    # end
    #approximate_optimize_cell = reach -> y₀ ∈ reach ? 0 : -1.0
    
    approximate_optimize_cell = cell -> -dist_to_zonotope_p(forward_network(solver, network, cell), y₀; p=p)
    return general_priority_optimization(lbs, ubs, approximate_optimize_cell, evaluate_objective;  n_steps = n_steps, solver=solver, early_stop=early_stop, stop_freq=stop_freq, stop_gap=stop_gap, initial_splits=initial_splits)
end

# Weird shouldnt work because requires each zonotope to have all of them? rethink through this
function inclusion_batch_wrapper(network, lbs, ubs, Y₀; n_steps = 1000, solver=Ai2z(), early_stop=true, stop_freq=200, stop_gap=1e-4, initial_splits=0)
    # #evaluate_objective = x -> max(-norm(y₀ - NeuralVerification.compute_output(network, x))
    # evaluate_objective = x -> -1.0
    # approximate_optimize_cell = reach -> all([y ∈ reach for y in eachcol(Y₀)]) ? 0 : -1.0
    # return general_priority_optimization(network, lbs, ubs, approximate_optimize_cell, evaluate_objective;  n_steps = n_steps, solver=solver, early_stop=early_stop, stop_freq=stop_freq, stop_gap=stop_gap, initial_splits=initial_splits)
end

function split_multiple_times(cell, n)
    q = Queue{Hyperrectangle}()
    enqueue!(q, cell)
    for i = 1:n
        new_cells = split_cell(dequeue!(q))
        enqueue!(q, new_cells[1])
        enqueue!(q, new_cells[2])
    end
    return q
end

# Use a priority based approach to split your space 
# General to any objective function passed in as well as an evaluate objective 
# The function approximate_optimize_cell takes in a cell and then does an approximate optimization over it 
# The function evaluate_objective takes in a point in the input space and evaluates the objective 
# This optimization strategy then uses these functions to provide bounds on the maximum objective
function general_priority_optimization(lbs, ubs, approximate_optimize_cell, evaluate_objective;  n_steps = 1000, solver=Ai2z(), early_stop=true, stop_freq=200, stop_gap=1e-4, initial_splits = 0)
        start_cell = Hyperrectangle(low=lbs, high=ubs)
        initial_cells = split_multiple_times(start_cell, initial_splits)
        println("Done with initial splits")
        # Create your queue, then add your original new_cells 
        cells = PriorityQueue(Base.Order.Reverse) # pop off largest first 
        [enqueue!(cells, cell, approximate_optimize_cell(cell)) for cell in initial_cells] # add with priority
        best_lower_bound = -Inf
        best_x = nothing
    
        # For n_steps dequeue a cell, split it, and then 
        for i = 1:n_steps
            cell, value = peek(cells) # peek instead of dequeue to get value, is there a better way?
            dequeue!(cells)
            
            # Early stopping
            if early_stop
                if i % stop_freq == 0
                    lower_bound = evaluate_objective(cell.center)
                    if lower_bound > best_lower_bound
                        best_lower_bound = lower_bound
                        best_x = cell.center
                    end
                    println("i: ", i)
                    println("lower bound: ", lower_bound)
                    println("best lower bound: ", best_lower_bound)
                    println("value: ", value)
                    if (value .- lower_bound) <= stop_gap
                        return best_x, best_lower_bound, value, i
                    end
                    println("max radius: ", max(radius(cell)))
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
                    lower_bound = evaluate_objective(cell.center)
                    if lower_bound > best_lower_bound
                        best_lower_bound = lower_bound
                        best_x = cell.center
                    end
                    return best_x, best_lower_bound, value, i 
                end
                new_value = approximate_optimize_cell(new_cell)
                enqueue!(cells, new_cell, new_value)
            end
        end
        # The largest value in our queue is the approximate optimum 
        cell, value = peek(cells)
        lower_bound = evaluate_objective(cell.center)
        if lower_bound > best_lower_bound
            best_lower_bound = lower_bound
            best_x = cell.center
        end
        return best_x, best_lower_bound, value, n_steps
end


# Use a priority based approach to split your space 
function approximate_priority_optimization(network, lbs, ubs, coeffs; n_steps = 1000, solver=Ai2z(), early_stop=true, stop_freq=200, stop_gap=1e-4)
    # Create your queue, then add your original cell 
    cells = PriorityQueue(Base.Order.Reverse) # pop off largest first 
    start_cell = Hyperrectangle(low=lbs, high=ubs)
    start_value = linear_approximate_optimize_cell(solver, network, start_cell, coeffs)
    enqueue!(cells, start_cell, start_value)

    best_lower_bound = -Inf

    # For n_steps dequeue a cell, split it, and then 
    for i = 1:n_steps
        cell, value = peek(cells) # peek instead of dequeue to get value, is there a better way?
        dequeue!(cells)
        
        # Early stopping
        if early_stop
            if i % stop_freq == 0
                lower_bound = compute_objective(network, cell.center, coeffs)
                best_lower_bound = max(best_lower_bound, lower_bound)
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
                return max(compute_objective(network, cell.center, coeffs), best_lower_bound), value, i 
            end
            enqueue!(cells, new_cell, linear_approximate_optimize_cell(solver, network, new_cell, coeffs))
        end
    end
    
    # The largest value in our queue is the approximate optimum 
    cell, value = peek(cells)
    return max(compute_objective(network, cell.center, coeffs), best_lower_bound), value, n_steps
end

function approximate_optimization(network, lbs, ubs, coeffs; solver=Ai2z(), n_per_latent=10, n_per_state=1, find_gap=true)
    # Ai2z overapproximation through discretization 
    overestimate = -99999.0
    underestimate = -99999.0
    
    n = 0
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

                    # Construct the input set, then propagate forwards to a 
                    # zonotope over-approximation of the output set
                    input_set = Hyperrectangle(low=cur_lbs, high=cur_ubs)
                    reach = forward_network(solver, network, input_set)
                    # The support function ρ maximizes coeffs^T x for x in reach
                    cur_overestimate = ρ(coeffs, reach)
                    cur_overestimate >= overestimate ? overestimate = cur_overestimate : nothing
                    n = n + 1

                    # Sample a point to estimate the gap
                    if find_gap
                        x_adv = fgsm(network, input_set.center, cur_lbs, cur_ubs, coeffs, median(cur_ubs - cur_lbs))
                        out_fgsm = dot(coeffs, NeuralVerification.compute_output(network, x_adv))
                        #out = dot(coeffs, NeuralVerification.compute_output(network, input_set.center))
                        out_fgsm >= underestimate ? underestimate = out_fgsm : nothing
                    end
                end
            end
        end
        println("Percent done: ", round(n/n_per_latent^2/n_per_state^2*100, digits=2))
    end
    return underestimate, overestimate
end