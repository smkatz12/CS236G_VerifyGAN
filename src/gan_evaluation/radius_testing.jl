### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 6040b06a-88e2-11eb-065a-91a57744ddee
using PlutoUI

# ╔═╡ 8436dc1a-88e2-11eb-0af8-e720babe14a4
using Plots

# ╔═╡ 6c29ca2e-88ec-11eb-26f5-77edf9cc9957
using NeuralVerification, LazySets, DataStructures, LinearAlgebra

# ╔═╡ c04277a0-88ec-11eb-3134-95c071907b91
using HDF5

# ╔═╡ f1feec76-88f9-11eb-0815-df81c3745d23
using Convex

# ╔═╡ 16d26c36-88f9-11eb-29b5-095267c101a2
using Mosek, MosekTools

# ╔═╡ 88308c6c-88e2-11eb-2592-291d612f7036
include("../verification/ai2zPQ.jl");

# ╔═╡ cf39445a-88fb-11eb-12a4-b37fb4d3386c
Core.eval(Main, :(const TOL = Ref(sqrt(eps()))))

# ╔═╡ 7a364e4e-88ec-11eb-10f0-e54195a19449
network = read_nnet("../../models/mlp256x4_msle.nnet");

# ╔═╡ c33129fc-88ee-11eb-3567-470b5d749438
function dist_to_zonotope_p(reach, point; p = 2)
    G = reach.generators
    c = reach.center
    n, m = size(G)
    x = Variable(m)
    obj = norm(G * x + c - point, p)
    prob = minimize(obj, [x <= 1.0, x >= -1.0])
    solve!(prob, Mosek.Optimizer(LOG=0))
    #@assert prob.status == OPTIMAL "Solve must result in optimal status"
    return prob.optval
end

# ╔═╡ 83e84d52-88ec-11eb-0862-87c21705a459
md"""
### Real data
"""

# ╔═╡ 8e65d074-88ec-11eb-22b5-bfcba861ec15
begin
	fn = "../../data/SK_DownsampledGANFocusAreaData.h5"
	images = h5read(fn, "y_train")
	images = reshape(images, 16*8, :)
	y = h5read(fn, "X_train")[1:2, :]
	nothing
end

# ╔═╡ c6f314ba-88ec-11eb-09f9-97ea3356de94
md"""
### Test nearest point
"""

# ╔═╡ dd8e5cc0-88ec-11eb-14f2-cf1cec0d1fcc
state_eps = 1e-5;

# ╔═╡ f285c6fe-88ec-11eb-2d81-3f7287a2ffde
md"""
image index: $(@bind image_ind NumberField(1:10000, default = 1))
p: $(@bind p NumberField(1:2, default = 2))
"""

# ╔═╡ 2cbbc3b4-88ed-11eb-1db2-ed4e3a7029db
begin
	lbs = [-1.0, -1.0, y[1, image_ind] / 6.366468343804353 - state_eps, y[2, image_ind] / 17.248858791583547 - state_eps]
	ubs = [1.0, 1.0, y[1, image_ind] / 6.366468343804353 + state_eps, y[2, image_ind] / 17.248858791583547 + state_eps]
end

# ╔═╡ 6d185bfa-88ed-11eb-18e3-4faac449d9df
y₀ = (images[:, image_ind] .* 2) .- 1;

# ╔═╡ 519f8390-88ee-11eb-26f4-05e4c89bf720
evaluate_objective(network, x) = -norm(y₀ - NeuralVerification.compute_output(network, x), p);

# ╔═╡ 93b64ebe-88ee-11eb-2b3b-db56ec68e3c1
optimize_reach(reach) = -dist_to_zonotope_p(reach, y₀; p=p);

# ╔═╡ 0f3a58d0-88ef-11eb-3dc8-9593b5567941
@time x, under, over = priority_optimization(network, lbs, ubs, optimize_reach, evaluate_objective);

# ╔═╡ 097e9d64-8900-11eb-26ec-41f3398f604f
-under, -over

# ╔═╡ 6e417cb0-88fd-11eb-2456-913367310813
begin
	gen_im = (NeuralVerification.compute_output(network, x) .+ 1) ./ 2
	gen_im = reshape(gen_im, 16, 8)'
	
	real_im = images[:, image_ind]
	real_im = reshape(real_im, 16, 8)'
	
	plot_im = cat(real_im, gen_im, dims = 2)
	plot(Gray.(plot_im), axis = [])
end

# ╔═╡ Cell order:
# ╠═6040b06a-88e2-11eb-065a-91a57744ddee
# ╠═8436dc1a-88e2-11eb-0af8-e720babe14a4
# ╠═6c29ca2e-88ec-11eb-26f5-77edf9cc9957
# ╠═c04277a0-88ec-11eb-3134-95c071907b91
# ╠═f1feec76-88f9-11eb-0815-df81c3745d23
# ╠═16d26c36-88f9-11eb-29b5-095267c101a2
# ╠═cf39445a-88fb-11eb-12a4-b37fb4d3386c
# ╠═88308c6c-88e2-11eb-2592-291d612f7036
# ╠═7a364e4e-88ec-11eb-10f0-e54195a19449
# ╠═c33129fc-88ee-11eb-3567-470b5d749438
# ╟─83e84d52-88ec-11eb-0862-87c21705a459
# ╠═8e65d074-88ec-11eb-22b5-bfcba861ec15
# ╟─c6f314ba-88ec-11eb-09f9-97ea3356de94
# ╠═dd8e5cc0-88ec-11eb-14f2-cf1cec0d1fcc
# ╠═2cbbc3b4-88ed-11eb-1db2-ed4e3a7029db
# ╠═6d185bfa-88ed-11eb-18e3-4faac449d9df
# ╠═519f8390-88ee-11eb-26f4-05e4c89bf720
# ╠═93b64ebe-88ee-11eb-2b3b-db56ec68e3c1
# ╟─f285c6fe-88ec-11eb-2d81-3f7287a2ffde
# ╠═0f3a58d0-88ef-11eb-3dc8-9593b5567941
# ╠═097e9d64-8900-11eb-26ec-41f3398f604f
# ╠═6e417cb0-88fd-11eb-2456-913367310813
