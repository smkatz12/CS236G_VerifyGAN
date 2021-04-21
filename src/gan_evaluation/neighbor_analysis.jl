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

# ╔═╡ a533a520-8b57-11eb-2d0d-81c3fb7c08d1
using PlutoUI

# ╔═╡ bfe7cef0-8b57-11eb-257f-d7b746a04fe0
using Plots

# ╔═╡ cdda3700-8b57-11eb-2ac9-6766eb8ce24d
using BSON

# ╔═╡ 4059192c-8b58-11eb-01e5-f37aaf5fd8dc
using HDF5

# ╔═╡ 714aeeb6-8b58-11eb-2f6c-df3ae7f6222c
using NeuralVerification

# ╔═╡ d1e40128-8b57-11eb-378a-d57f07e0f3c4
md"""
## Load in the data
"""

# ╔═╡ dab764fc-8b57-11eb-3cf5-45a6792eb0fe
begin
	res1 = BSON.load("radius_data/adversarial_radii.bson")
	xsadv = res1[:xs]
	undersadv = res1[:unders]
	oversadv = res1[:overs]
	nothing
end

# ╔═╡ ec61fc6c-8b57-11eb-36ff-2544bdb43ffe
begin
	res2 = BSON.load("radius_data/msle256x4_radii.bson")
	xsmsle = res2[:xs]
	undersmsle = res2[:unders]
	oversmsle = res2[:overs]
	nothing
end

# ╔═╡ 27f5f6b8-8b58-11eb-3fe7-87564afb7f0a
begin
	fn = "../../data/SK_DownsampledGANFocusAreaData.h5"
	images = h5read(fn, "y_train")
	real_images = reshape(images, 16*8, :)
	y = h5read(fn, "X_train")[1:2, :]
	nothing
end

# ╔═╡ 83295136-8b58-11eb-216f-5f6e4a480433
adv_network = read_nnet("../../models/big_normal_gen.nnet");

# ╔═╡ 5f290030-8b58-11eb-12a5-458449347726
msle_network = read_nnet("../../models/mlp256x4_msle.nnet");

# ╔═╡ 0642bc7a-8b58-11eb-16e7-59303e194937
md"""
## Plot nearest neighbors
"""

# ╔═╡ 24bb4eb0-8b58-11eb-21ca-5d25b11efbf7
md"""
image index: $(@bind image_ind NumberField(1:10000, default = 1))
"""

# ╔═╡ 92749cfa-8b5c-11eb-1237-a12952226e62
oversmsle[image_ind]

# ╔═╡ 92fe4694-8b5c-11eb-03d2-df5e5c6eafa9
oversadv[image_ind]

# ╔═╡ a854990c-8b58-11eb-05af-bd511fc1f856
begin
	gen_im_adv = (NeuralVerification.compute_output(adv_network, xsadv[:, image_ind]) .+ 1) ./ 2
	gen_im_adv = reshape(gen_im_adv, 16, 8)'
	
	gen_im_msle = (NeuralVerification.compute_output(msle_network, xsmsle[:, image_ind]) .+ 1) ./ 2
	gen_im_msle = reshape(gen_im_msle, 16, 8)'
	
	real_im = real_images[:, image_ind]
	real_im = reshape(real_im, 16, 8)'
	
	plot_im = cat(real_im, gen_im_msle, gen_im_adv, dims = 1)
	plot(Gray.(plot_im), axis = [])
end

# ╔═╡ 5a2ca4aa-8b5c-11eb-2aca-45b54a8f0b5a
argmax(oversmsle)

# ╔═╡ 6a24c0fe-8b5c-11eb-1e27-e53425b34d50
argmax(oversadv)

# ╔═╡ Cell order:
# ╠═a533a520-8b57-11eb-2d0d-81c3fb7c08d1
# ╠═bfe7cef0-8b57-11eb-257f-d7b746a04fe0
# ╠═cdda3700-8b57-11eb-2ac9-6766eb8ce24d
# ╠═4059192c-8b58-11eb-01e5-f37aaf5fd8dc
# ╠═714aeeb6-8b58-11eb-2f6c-df3ae7f6222c
# ╟─d1e40128-8b57-11eb-378a-d57f07e0f3c4
# ╠═dab764fc-8b57-11eb-3cf5-45a6792eb0fe
# ╠═ec61fc6c-8b57-11eb-36ff-2544bdb43ffe
# ╠═27f5f6b8-8b58-11eb-3fe7-87564afb7f0a
# ╠═83295136-8b58-11eb-216f-5f6e4a480433
# ╠═5f290030-8b58-11eb-12a5-458449347726
# ╟─0642bc7a-8b58-11eb-16e7-59303e194937
# ╠═92749cfa-8b5c-11eb-1237-a12952226e62
# ╠═92fe4694-8b5c-11eb-03d2-df5e5c6eafa9
# ╟─24bb4eb0-8b58-11eb-21ca-5d25b11efbf7
# ╠═a854990c-8b58-11eb-05af-bd511fc1f856
# ╠═5a2ca4aa-8b5c-11eb-2aca-45b54a8f0b5a
# ╠═6a24c0fe-8b5c-11eb-1e27-e53425b34d50
