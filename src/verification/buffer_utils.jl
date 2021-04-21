using Colors
using Plots

"""
    add_images_to_tree(tree, images, states)

This function adds a list of images to the corresponding leaves of the tree based 
on each images' position. Assume images are in a dim_1 x dim_2 x num_images Array
"""
function add_images_to_tree(tree, images, states)
    for i = 1:size(images, 3)
        image = images[:, :, i]
        state = states[:, i]
        add_image_to_tree(tree, image, state)
    end
end

"""
    add_image_to_tree(tree, image, state)

Add a single image to the tree
"""
function add_image_to_tree(tree, image, state)
    leaf = get_leaf(tree, state)
    push!(leaf.images, image)
    push!(leaf.states, state)
end


function get_closest_generated_image(gan, image, state)
    state_normalized = state ./ [6.366468343804353, 17.248858791583547] 
    lbs = [-1.0, -1.0, state_normalized...]
    ubs = [1.0, 1.0, state_normalized...]
    vec_image = vec(image)
    x, lower, upper, steps = inclusion_wrapper(gan, lbs, ubs, vec_image, 2; n_steps = 2, stop_gap=0.05)
    println("Interval: ", (lower, upper))
    println(x)
    return NeuralVerification.compute_output(gan, x)
end

function sample_closest_generated_image(gan, image, state; n_samples=100)
    state_normalized = state ./ [6.366468343804353, 17.248858791583547] 
    lbs = [-1.0, -1.0, state_normalized...]
    ubs = [1.0, 1.0, state_normalized...]
    vec_image = vec(image)
    hyperrectangle = Hyperrectangle(low=lbs, high=ubs)
    samples = sample(hyperrectangle, n_samples)

    best_so_far = Inf
    best_output_so_far = []
    for sample in samples
        output = NeuralVerification.compute_output(gan, sample)
        curr_dist = norm(output .- vec(image), 2)
        if curr_dist < best_so_far
            best_so_far = curr_dist 
            best_output_so_far = output
        end
    end
    return best_output_so_far
end

function get_image_gaps_for_leaf(gan, leaf)
    states = leaf.states
    images = leaf.images
    if length(images) > 0
        gaps = []
        for (state, image) in zip(states, images)
            vec_image = reshape(image, prod(size(image)))
            gap = image .- reshape(sample_closest_generated_image(gan, vec_image, state), size(image))
            push!(gaps, gap)
        end
        return gaps
    # Otherwise return a gap of 0 for this leaf
    else 
        return [zeros(16, 8)] # TODO: how to access dimension here
    end
end

function image_gaps_to_buffer(image_gaps)
    lbs = Inf*ones(size(image_gaps[1]))
    ubs = -Inf*ones(size(image_gaps[1]))
    for gap in image_gaps 
        lbs = min.(lbs, gap)
        ubs = max.(ubs, gap)
    end
    return Hyperrectangle(low=vec(lbs), high=vec(ubs))
end

function add_buffer_to_leaf(gan, leaf)
    leaf.buffer = image_gaps_to_buffer(get_image_gaps_for_leaf(gan, leaf))
end

function add_buffers_to_tree(gan, tree)
    for (i, leaf) in enumerate(get_leaves(tree))
        add_buffer_to_leaf(gan, leaf)
        println("Added leaf ", i)
    end
end

function visualize_buffer(output_file_base, tree, state)
    leaf = get_leaf(tree, state)
    buffer_radius = radius_hyperrectangle(leaf.buffer)
    plot(Gray.(reshape(buffer_radius, 16, 8)'), title="Buffer radius")
    savefig(output_file_base*"radius.png")
    plot(Gray.(reshape(leaf.buffer.center, 16, 8)'), title="Buffer Center")
    savefig(output_file_base*"center.png")
end

function plot_images_from_tree(output_file_base, tree, state)
    leaf = get_leaf(tree, state)
    images = leaf.images 
    for (i, image) in enumerate(images)
        plot(Gray.(image'))
        savefig(output_file_base*string(i)*".png")
    end
end