using DataStructures

include("tree_utils.jl")
include("../dynamics_model/dubins_model.jl")

function model_check!(tree::KDTREE; max_iterations = 200, belres = 1e-6, γ = 1.0)
    for i = 1:max_iterations
        residual = 0.0
        
        # Stacks to go through tree
        lb_s = Stack{Vector{Float64}}()
        ub_s = Stack{Vector{Float64}}()
        s = Stack{Union{LEAFNODE, KDNODE}}()

        push!(lb_s, tree.lbs)
        push!(ub_s, tree.ubs)
        push!(s, tree.root_node)

        while !isempty(s)
            curr = pop!(s)
            curr_lbs = pop!(lb_s)
            curr_ubs = pop!(ub_s)

            if typeof(curr) == LEAFNODE
                min_cte = curr_lbs[1]
                max_cte = curr_ubs[1]

                if (min_cte < -10.0) || (max_cte > 10.0)
                    curr.prob = 1.0
                else
                    old_prob = curr.prob

                    next_lbs, next_ubs = reachable_cell(curr_lbs, curr_ubs, curr.min_control, curr.max_control)
                    next_nodes = get_overlapping_nodes(tree.root_node, next_lbs, next_ubs)
                    new_prob = γ * maximum([node.prob for node in next_nodes])
                    curr.prob = new_prob

                    change = abs(old_prob - new_prob)
                    change > residual && (residual = change)
                end
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
        println("[Iteration $i] residual: $residual")
        residual < belres && break
    end
end

function forward_reach(init_tree::KDTREE; max_iter = 50, verbose = false)
    # This function assumes that the first tree is labeled already with the states you
    # want to allow the system to start in

    trees = [init_tree]

    for i = 1:max_iter
        tree = trees[end]
        next_tree = copy(init_tree)
        zero_out!(next_tree)
        
        # Stacks to go through tree
        lb_s = Stack{Vector{Float64}}()
        ub_s = Stack{Vector{Float64}}()
        s = Stack{Union{LEAFNODE, KDNODE}}()

        push!(lb_s, tree.lbs)
        push!(ub_s, tree.ubs)
        push!(s, tree.root_node)

        while !isempty(s)
            curr = pop!(s)
            curr_lbs = pop!(lb_s)
            curr_ubs = pop!(ub_s)

            if typeof(curr) == LEAFNODE
                next_lbs, next_ubs = reachable_cell(curr_lbs, curr_ubs, curr.min_control, curr.max_control)
                curr_nodes = get_overlapping_nodes(tree.root_node, next_lbs, next_ubs)
                next_nodes = get_overlapping_nodes(next_tree.root_node, next_lbs, next_ubs)
                for node in next_nodes
                    node.prob = curr.prob
                end
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
        push!(trees, next_tree)
        # Check convergence
        converged = tree == next_tree
        verbose && println("[Iteration $i] converged: $converged")
        converged && break
    end
    return trees
end

function forward_reach!(trees::Vector{KDTREE}; verbose = false)
    # This function assumes that the first tree is labeled already with the states you
    # want to allow the system to start in

    for i = 1:length(trees) - 1
        tree = trees[i]
        next_tree = trees[i + 1]
        
        # Stacks to go through tree
        lb_s = Stack{Vector{Float64}}()
        ub_s = Stack{Vector{Float64}}()
        s = Stack{Union{LEAFNODE, KDNODE}}()

        push!(lb_s, tree.lbs)
        push!(ub_s, tree.ubs)
        push!(s, tree.root_node)

        while !isempty(s)
            curr = pop!(s)
            curr_lbs = pop!(lb_s)
            curr_ubs = pop!(ub_s)

            if typeof(curr) == LEAFNODE
                next_lbs, next_ubs = reachable_cell(curr_lbs, curr_ubs, curr.min_control, curr.max_control)
                curr_nodes = get_overlapping_nodes(tree.root_node, next_lbs, next_ubs)
                next_nodes = get_overlapping_nodes(next_tree.root_node, next_lbs, next_ubs)
                for node in next_nodes
                    node.prob = curr.prob
                end
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
        # Check convergence
        converged = tree == next_tree
        verbose && println("[Iteration $i] converged: $converged")
    end
end

""" Labeling functions """

function label_tree_failures!(tree)
    # Stacks to go through tree
    lb_s = Stack{Vector{Float64}}()
    ub_s = Stack{Vector{Float64}}()
    s = Stack{Union{LEAFNODE, KDNODE}}()

    push!(lb_s, tree.lbs)
    push!(ub_s, tree.ubs)
    push!(s, tree.root_node)

    while !(isempty(s))
        curr = pop!(s)
        curr_lbs = pop!(lb_s)
        curr_ubs = pop!(ub_s)

        if typeof(curr) == LEAFNODE
            # Check if off runway
            min_cte = curr_lbs[1]
            max_cte = curr_ubs[1]

            curr.prob = (min_cte < -10.0) || (max_cte > 10.0) ? 1.0 : 0.0
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

function label_tree_observable!(tree; control_gains = [-0.74, -0.44])
    # Stacks to go through tree
    lb_s = Stack{Vector{Float64}}()
    ub_s = Stack{Vector{Float64}}()
    s = Stack{Union{LEAFNODE, KDNODE}}()

    push!(lb_s, tree.lbs)
    push!(ub_s, tree.ubs)
    push!(s, tree.root_node)

    while !(isempty(s))
        curr = pop!(s)
        curr_lbs = pop!(lb_s)
        curr_ubs = pop!(ub_s)

        if typeof(curr) == LEAFNODE
            curr.min_control = control_gains' * curr_ubs
            curr.max_control = control_gains' * curr_lbs

            # if curr.min_control > curr.max_control
            #     println(curr_ubs)
            #     println(curr_lbs)
            # end
            
            # Check if off runway
            min_cte = curr_lbs[1]
            max_cte = curr_ubs[1]

            curr.prob = (min_cte < -10.0) || (max_cte > 10.0) ? 1.0 : 0.0
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

function label_tree_with_prob!(tree, prob; control_gains = [-0.74, -0.44])
    # Stacks to go through tree
    lb_s = Stack{Vector{Float64}}()
    ub_s = Stack{Vector{Float64}}()
    s = Stack{Union{LEAFNODE, KDNODE}}()

    push!(lb_s, tree.lbs)
    push!(ub_s, tree.ubs)
    push!(s, tree.root_node)

    while !(isempty(s))
        curr = pop!(s)
        curr_lbs = pop!(lb_s)
        curr_ubs = pop!(ub_s)

        if typeof(curr) == LEAFNODE
            curr.min_control = control_gains' * curr_ubs
            curr.max_control = control_gains' * curr_lbs
            
            curr.prob = prob
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

function one_out!(tree)
    # Stacks to go through tree
    lb_s = Stack{Vector{Float64}}()
    ub_s = Stack{Vector{Float64}}()
    s = Stack{Union{LEAFNODE, KDNODE}}()

    push!(lb_s, tree.lbs)
    push!(ub_s, tree.ubs)
    push!(s, tree.root_node)

    while !(isempty(s))
        curr = pop!(s)
        curr_lbs = pop!(lb_s)
        curr_ubs = pop!(ub_s)

        if typeof(curr) == LEAFNODE
            curr.prob = 1.0
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

function zero_out!(tree)
    # Stacks to go through tree
    lb_s = Stack{Vector{Float64}}()
    ub_s = Stack{Vector{Float64}}()
    s = Stack{Union{LEAFNODE, KDNODE}}()

    push!(lb_s, tree.lbs)
    push!(ub_s, tree.ubs)
    push!(s, tree.root_node)

    while !(isempty(s))
        curr = pop!(s)
        curr_lbs = pop!(lb_s)
        curr_ubs = pop!(ub_s)

        if typeof(curr) == LEAFNODE
            curr.prob = 0.0
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