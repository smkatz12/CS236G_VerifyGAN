include("ai2zPQ.jl")
include("approximate.jl")

network = read_nnet("../../models/full_mlp_best_conv.nnet")

min_cte = -0.2
max_cte = 0.0

min_he = -0.5
max_he = 0.5

lbs = [-1.0, -1.0, min_cte / 6.366468343804353, min_he / 17.248858791583547]
ubs = [1.0, 1.0, max_cte / 6.366468343804353, max_he / 17.248858791583547]

coeffs = [-0.74, -0.44]

# @time min_control, max_control = approximate_optimization(lbs, ubs, coeffs, 
#                                             n_per_latent = 30, n_per_state = 2)

# # @time lower_bound, max_control, _ = approximate_priority_optimization(network, lbs, ubs, coeffs)

#@time best_x, best_lower_bound, value = priority_optimization(network, lbs, ubs, optimize_reach, evaluate_objective)

@time min_control, max_control = ai2zPQ_bounds(network, lbs, ubs, coeffs)