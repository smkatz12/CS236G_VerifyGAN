### A Pluto.jl notebook ###
# v0.12.20

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

# ╔═╡ 070361c0-76fc-11eb-357e-cf19d1d3b5b4
using PlutoUI

# ╔═╡ 3951770e-76fc-11eb-1c87-b78dd730f76a
using Distributions

# ╔═╡ ca1681d2-76fd-11eb-2909-9dd6ab8229b5
using Plots

# ╔═╡ 71698ab4-76fa-11eb-38c9-7fa42f02f56d
md"""
# Dynamics
"""

# ╔═╡ bf956b40-76fa-11eb-212f-3f3112b8bc6d
begin
	L = 5 # m
	v = 5 # m/s
end

# ╔═╡ d07ee170-76fa-11eb-162e-f72ea40b9986
function next_state(x, θ, ϕ; dt = 0.05)
    # x is in meters and θ is in degrees
	# ϕ is steering angle input (deg)
	
	# Dynamics model
	ẋ = v * sind(θ)
	θ̇ = (v / L) * tand(ϕ)
	
	x′ = x + ẋ * dt
	θ′ = θ + rad2deg(θ̇) * dt
	
	return [x′, θ′]
end

# ╔═╡ d82afe4a-76fa-11eb-28c8-97eb770bc4db
function next_step(x, θ, ϕ; num_steps = 20)
	for i = 1:num_steps
		x, θ = next_state(x, θ, ϕ)
	end
	return x, θ
end

# ╔═╡ b2d41f5c-76fd-11eb-2dbd-9368d2701655
function next_region(lbs, ubs, ϕ_min, ϕ_max)
	next_lb_x, next_lb_θ = next_step(lbs[1], lbs[2], ϕ_min)
	next_ub_x, next_ub_θ = next_step(ubs[1], ubs[2], ϕ_max)
	return [next_lb_x, next_lb_θ], [next_ub_x, next_ub_θ]
end

# ╔═╡ 2b7c09d6-76fb-11eb-127e-311805fea358
md"""
# Visualize Overapproximation
"""

# ╔═╡ 956b135e-76fe-11eb-2008-3b04ccaf68ce
md"""
Get sample next states:
"""

# ╔═╡ ac3e5b30-76fd-11eb-235e-5750873353cd
md"""
Get overapproximated region:
"""

# ╔═╡ 51c1aaa2-76ff-11eb-1aed-930eae90a797
to_plot(lbs, ubs) = Shape([lbs[1], ubs[1], ubs[1], lbs[1], lbs[1]], [lbs[2], lbs[2], ubs[2], ubs[2], lbs[2]])

# ╔═╡ 49d7e35e-76fc-11eb-30b8-dfb092bbd104
md"""
Number of samples: $(@bind nsamps NumberField(1:50:500, default=100))
"""

# ╔═╡ 361a6f2c-76fb-11eb-053a-c3f346261cdb
md"""
xmin: $(@bind xmin NumberField(-11:0.05:11, default=0))
xmax: $(@bind xmax NumberField(-11:0.05:11, default=0.5))
"""

# ╔═╡ 46932d02-76fc-11eb-2bb6-55303e0010fb
xsamples = rand(Uniform(xmin, xmax), nsamps);

# ╔═╡ 164b77ee-76fc-11eb-269a-e7d8e82f9a3f
md"""
θmin: $(@bind θmin NumberField(-30:0.05:30, default=0))
θmax: $(@bind θmax NumberField(-30:0.05:30, default=0.5))
"""

# ╔═╡ 2397fecc-76fc-11eb-2490-f55db9f43b1e
θsamples = rand(Uniform(θmin, θmax), nsamps);

# ╔═╡ 97a8b374-76fc-11eb-378f-0752200854f1
begin
	next_xs = zeros(nsamps)
	next_θs = zeros(nsamps)
	
	for i = 1:nsamps
		x = xsamples[i]
		θ = θsamples[i]
		ϕ = -0.74x - 0.44θ
		next_x, next_θ = next_step(x, θ, ϕ)
		next_xs[i] = next_x
		next_θs[i] = next_θ
	end
end

# ╔═╡ a8d82164-76fe-11eb-0cfa-c955b47a7ef0
begin
	ϕ_min = -0.74xmax - 0.44θmax
	ϕ_max = -0.74xmin + 0.74θmin
	next_lbs, next_ubs = next_region([xmin, θmin], [xmax, θmax], ϕ_min, ϕ_max)
end

# ╔═╡ f53f44e0-76fe-11eb-2608-45cf7ece5954
begin
	p = plot(to_plot([xmin, θmin], [xmax, θmax]), alpha = 0.3, 
		label = "Original Cell", xlabel = "x", ylabel = "θ", legend = :topleft)
	scatter!(p, xsamples, θsamples, label = "Original Samples")
	scatter!(p, next_xs, next_θs, label = "Next Samples")
	plot!(p, to_plot(next_lbs, next_ubs), alpha = 0.3, label = "Next Cell")
end

# ╔═╡ 94f4e1bc-7700-11eb-3f0a-b948ce227cbc
minimum(next_θs)

# ╔═╡ a0319e08-7700-11eb-3b83-0db5a632dd84
argmin(next_θs)

# ╔═╡ c0e3c810-7700-11eb-37f6-bfeb0a368a51
xsamples[435], θsamples[435]

# ╔═╡ 4ba710a6-7701-11eb-2da9-e1b47d2c2cc8
next_lbs

# ╔═╡ 4e98d8da-7701-11eb-3836-579a533db4ea
next_ubs

# ╔═╡ Cell order:
# ╠═070361c0-76fc-11eb-357e-cf19d1d3b5b4
# ╠═3951770e-76fc-11eb-1c87-b78dd730f76a
# ╠═ca1681d2-76fd-11eb-2909-9dd6ab8229b5
# ╟─71698ab4-76fa-11eb-38c9-7fa42f02f56d
# ╠═bf956b40-76fa-11eb-212f-3f3112b8bc6d
# ╠═d07ee170-76fa-11eb-162e-f72ea40b9986
# ╠═d82afe4a-76fa-11eb-28c8-97eb770bc4db
# ╠═b2d41f5c-76fd-11eb-2dbd-9368d2701655
# ╟─2b7c09d6-76fb-11eb-127e-311805fea358
# ╠═46932d02-76fc-11eb-2bb6-55303e0010fb
# ╠═2397fecc-76fc-11eb-2490-f55db9f43b1e
# ╟─956b135e-76fe-11eb-2008-3b04ccaf68ce
# ╠═97a8b374-76fc-11eb-378f-0752200854f1
# ╟─ac3e5b30-76fd-11eb-235e-5750873353cd
# ╠═a8d82164-76fe-11eb-0cfa-c955b47a7ef0
# ╠═51c1aaa2-76ff-11eb-1aed-930eae90a797
# ╟─49d7e35e-76fc-11eb-30b8-dfb092bbd104
# ╟─361a6f2c-76fb-11eb-053a-c3f346261cdb
# ╟─164b77ee-76fc-11eb-269a-e7d8e82f9a3f
# ╠═f53f44e0-76fe-11eb-2608-45cf7ece5954
# ╠═94f4e1bc-7700-11eb-3f0a-b948ce227cbc
# ╠═a0319e08-7700-11eb-3b83-0db5a632dd84
# ╠═c0e3c810-7700-11eb-37f6-bfeb0a368a51
# ╠═4ba710a6-7701-11eb-2da9-e1b47d2c2cc8
# ╠═4e98d8da-7701-11eb-3836-579a533db4ea
