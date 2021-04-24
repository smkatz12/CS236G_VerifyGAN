using Colors
using Plots
using NeuralVerification
using NeuralVerification: Layer, Id, write_nnet, compute_output

(network::Network)(input::Array{Float32, 1}) = compute_output(network, input)
(network::Network)(input::Array{Float32, 2}) = mapslices(network, input; dims=1)

include(string(@__DIR__, "/../gan_evaluation/approx_radius.jl"))

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


# function get_closest_generated_image(gan, image, state; latent_bound=0.8)
#     state_normalized = state ./ [6.366468343804353, 17.248858791583547] 
#     lbs = [-latent_bound, -latent_bound, state_normalized...]
#     ubs = [latent_bound, latent_bound, state_normalized...]
#     vec_image = vec(image)
#     x, lower, upper, steps = inclusion_wrapper(gan, lbs, ubs, vec_image, 2; n_steps = 2, stop_gap=0.05)
#     println("Interval: ", (lower, upper))
#     println(x)
#     return NeuralVerification.compute_output(gan, x)
# end

function sample_closest_generated_image(gan, image, state; num_samples=1000, latent_bound=0.8)
    state_normalized = state ./ [6.366468343804353, 17.248858791583547] 
    lbs = [-latent_bound, -latent_bound, (state_normalized .- 1e-5)...]
    ubs = [latent_bound, latent_bound, (state_normalized .+ 1e-5)...]
    vec_image = vec(image)
    x, best_dist = get_approximate_minimum_radius(gan, lbs, ubs, vec_image; num_samples = num_samples, p = 2)

    return compute_output(gan, x)
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
    # Grow hyperrectangle that will include the original point 
    # until it touches all points 
    highest = zeros(size(image_gaps[1]))
    lowest = zeros(size(image_gaps[1]))
    for gap in image_gaps 
        highest = max.(highest, gap)
        lowest = min.(lowest, gap)
    end
    center = (highest .+ lowest) ./ 2.0
    radius = (highest .- lowest) ./ 2.0
    return Hyperrectangle(vec(center), vec(radius))

    # Expand hyperrectangle centered on generated point until contains all 
    # radii = zeros(size(image_gaps[1]))
    # for gap in image_gaps 
    #     radii = max.(radii, abs.(gap))
    # end
    # return Hyperrectangle(zeros(length(image_gaps[1])), vec(radii))
end

function add_buffer_to_leaf(gan, leaf)
    leaf.buffer = image_gaps_to_buffer(get_image_gaps_for_leaf(gan, leaf))
end

function add_buffers_to_tree(gan, tree)
    Threads.@threads for (i, leaf) in collect(enumerate(get_leaves(tree)))
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
        println(Threads.threadid())
        plot(Gray.(image'))
        savefig(output_file_base*string(i)*".png")
    end
end

function plot_image_and_closest_gen(gan, image, state)
    close_image = reshape(sample_closest_generated_image(gan, image, state), 16, 8)
    plot_original = plot(Gray.(image'), title="Original")
    plot_close = plot(Gray.(close_image'), title="Generated")
    plot(plot_original, plot_close, layout=(1, 2))
end

# Rescale from -1 --> 1 to 0 --> 1
# So we want Ax + b for that last layer instead of mapping -1 --> 1 to be 0 --> 1
# We want (Ax + b + 1)/2. So we should update A to A/2 and b to (b+1)/2
function rescale_gan_output(output_file, gan)
    layers = gan.layers
    new_layers = Layer[layers[i] for i = 1:length(layers)-1]
    push!(new_layers, Layer(layers[end].weights ./ 2.0, (layers[end].bias .+ 1) ./ 2.0, Id()))
    new_network = Network(new_layers)
    write_nnet(output_file, new_network)
end


# Code to test gap calculation 
# points = [randn(2) .+ 2.0 for i = 1:10]
# Plots.scatter([point[1] for point in points], [point[2] for point in points], label="gaps")
# rect = image_gaps_to_buffer(points)
# plot!(rect, label="buffer")

# Code to check close images
# images_for_leafs = [get_leaf(tree, state).images for state in eachcol(states)]
# index = 1
# plot_image_and_closest_gen(gan_network, images_for_leafs[index][1], states[1:2, index])