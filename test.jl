using HDF5
using NeuralVerification
using NeuralVerification: compute_output
using BSON: @save, @load

include("./src/verification/tree_utils.jl")
include("./src/verification/approximate.jl")
include("./src/verification/buffer_utils.jl")

# Load your networks
full_network_file = "./models/full_mlp_best_conv.nnet"
gan_network_file = "./models/mlp_gen_best_conv_rescaled.nnet"
control_network_file = "models/KJ_TaxiNet.nnet"

full_network = read_nnet(full_network_file)
gan_network = read_nnet(gan_network_file)
control_network = read_nnet(control_network_file)

coeffs = [-0.74, -0.44]

# Load your image data
states = h5read("./data/SK_DownsampledGANFocusAreaData.h5", "X_train") # 3 x 10000
images = h5read("./data/SK_DownsampledGANFocusAreaData.h5", "y_train") # 16 x 8 x 10000
#images = (images .* 2) .- 1 # don't rescale to match the output of the network! because the network gan_network has already been rescaled to the 0-->1 range?

# Create your tree
max_widths = [0.2, 0.5] # Maximum cell widths
lbs = [-11.0, -30.0]
ubs = [11.0, 30.0]
#tree = create_tree(max_widths, lbs, ubs);

# Add your images to the tree. Note for states we drop the downtrack position so only include 1:2
#add_images_to_tree(tree, images, states[1:2, :])
#add_buffers_to_tree(gan_network, tree)

#@save "./tree_1000samples_preverification.bson" tree
@load "./src/verification/verified_trees/tree_1000samples_preverification.bson" tree
# For the given state visualize the buffer and images, and then find the max / min control 
# with and without the buffer
state = [0.0, 0.0]
# visualize_buffer("./plots/buffer", tree, state)
# plot_images_from_tree("./plots/images", tree, state)

# buffer = get_buffer(tree, state)

# lb_verify = [-0.8, -0.8, (state ./ [6.366468343804353, 17.248858791583547])...]
# ub_verify = [0.8, 0.8, (state ./ [6.366468343804353, 17.248858791583547])...]
# @time min_val_buffer, max_val_buffer = ai2zPQ_bounds_buffered(gan_network, control_network, lb_verify, ub_verify, coeffs, buffer; n_steps=1000)
# @time min_val_linear, max_val_linear = max_min_linear(full_network, lb_verify, ub_verify, coeffs; n_steps=1000)
# @time min_val_buffer_breakdown, max_val_buffer_breakdown = ai2zPQ_bounds_buffered_breakdown(gan_network, control_network, lb_verify, ub_verify, coeffs, buffer; n_steps=1000, stop_freq = 50, stop_gap=1e-1, initial_splits=0)


# println("Min, max with linear: ", [min_val_linear, max_val_linear])
# println("Min, max with buffer: ", [min_val_buffer, max_val_buffer])
# println("Min, max with buffer breakdown: ", [min_val_buffer_breakdown, max_val_buffer_breakdown])

#verify_tree_buffered!(tree, gan_network, control_network, full_network)

verify_tree_buffered_parallel!(tree, gan_network, control_network, full_network)

@save "./src/verification/verified_trees/buffer_breakdown_1000samples.bson" tree

### Test running a linear optimization with a zonotope input
# input_set = rand(Zonotope, dim=4)
# scale = 0.01
# input_set = affine_map(scale * Matrix(I, 4, 4), rand(Zonotope, dim=4), [0; 0; 0; 0;])
# x, lower, upper, steps = linear_opt_wrapper(full_network, input_set, coeffs; n_steps=100000)
# println("interval: ", [lower, upper])
# Now, run the verification 

### Test after finding the optimal input from buffer breakdown approach, see what the image looks like 
# and if it really leads to that control. 
# x_max = [-0.7999999880790725, 0.7999999880790711, 0.0, 0.0]
# x_min = [-0.08108596801757816, 0.799996566772461, 0.0, 0.0]

# # deal with max 
# gan_output_max = compute_output(gan_network, x_max)
# buffer_region_max = LazySets.translate(buffer, gan_output_max) # translate the buffer over to the gan's output
# # now optimize over the buffer region to find the worst image in that region 
# opt_image_max, opt_val_max = mip_linear_opt(control_network, buffer_region_max, coeffs)

# println("opt val max mip: ", opt_val_max)
# println("control from image max: ", coeffs' * compute_output(control_network, opt_image_max))
# println("control from unbuffered image max: ", coeffs' * compute_output(control_network, gan_output_max))
# plot(Gray.(reshape(opt_image_max, 16, 8)'))
# println("Optimal image gap with the gan output is in the buffer: ", (opt_image_max .- gan_output_max) ∈ buffer)

# # deal with min 
# gan_output_min = compute_output(gan_network, x_min)
# buffer_region_min = LazySets.translate(buffer, gan_output_min) # translate the buffer over to the gan's output
# # now optimize over the buffer region to find the worst image in that region 
# opt_image_min, opt_val_min = mip_linear_opt(control_network, buffer_region_min, -coeffs)
# opt_val_min = -opt_val_min # because of -coeffs for minimization

# println()
# println("opt val min mip: ", opt_val_min)
# println("control from image min: ", coeffs' * compute_output(control_network, opt_image_min))
# println("control from unbuffered image min: ", coeffs' * compute_output(control_network, gan_output_min))
# plot(Gray.(reshape(opt_image_min, 16, 8)'))
# println("Optimal image gap with the gan output is in the buffer: ", (opt_image_min .- gan_output_min) ∈ buffer)


### get a histogram of the norms of the buffers 
# p = Inf
# buffer_norms = [norm(vec(radius_hyperrectangle(leaf.buffer)), p) for leaf in get_leaves(tree)]
# histogram(buffer_norms, title="Histogram of buffer radius norms with p = "*string(p))


#image_lengths = [length(leaf.images) for leaf in get_leaves(tree)]

# Parallelization play 
# ntails = @parallel (+) for i = 1:1000
#     rand(Bool)
# end
