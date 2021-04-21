using Flux
using HDF5
using Distributions
using LinearAlgebra
using NearestNeighbors

function recall(fake_images, real_images; k = 30)    
    # Get kdtree for fake images to be used for nearest neighbor interpolation
    manifold_tree = KDTree(fake_images)

    # Get distances to kth nearest neighbor for each point
    _, dists = knn(manifold_tree, fake_images, k + 1, true)
    manifold_dists = [dists[i][end] for i = 1:length(dists)]
    max_dist = maximum(manifold_dists)

    # Check if real images are in the manifold
    reals_in = falses(size(real_images, 2))
    for i = 1:size(real_images, 2)
        idxs = inrange(manifold_tree, real_images[:, i], max_dist, true)
        dists = [norm(fake_images[:, j] .- real_images[:, i]) for j in idxs]
        reals_in[i] = any(dists .≤ manifold_dists[idxs])
    end

    return sum(reals_in) / size(real_images, 2)
end

function recall(fake_images, manifold_tree, real_images; k = 30)    
    # Get distances to kth nearest neighbor for each point
    _, dists = knn(manifold_tree, fake_images, k + 1, true)
    manifold_dists = [dists[i][end] for i = 1:length(dists)]
    max_dist = maximum(manifold_dists)

    # Check if real images are in the manifold
    reals_in = falses(size(real_images, 2))
    for i = 1:size(real_images, 2)
        idxs = inrange(manifold_tree, real_images[:, i], max_dist, true)
        dists = [norm(fake_images[:, j] .- real_images[:, i]) for j in idxs]
        reals_in[i] = any(dists .≤ manifold_dists[idxs])
    end

    return sum(reals_in) / size(real_images, 2)
end

function get_inputs(n; latent_dist = Uniform(-1, 1))
    ctes = rand(Uniform(-10 / 6.366468343804353, 10 / 6.366468343804353), 1, n)
    hes = rand(Uniform(-30 / 17.248858791583547, 30 / 17.248858791583547), 1, n)
    l1s = rand(latent_dist, 1, n)
    l2s = rand(latent_dist, 1, n)

    return [l1s; l2s; ctes; hes]
end

""" Tuning """

function vary_sample_size(generator, sizes, real_images; k = 10)
    # Get the fake images to cover all sizes
    inputs = get_inputs(maximum(sizes))
    fake_images = generator(inputs)
    fake_images = (reshape(fake_images, 128, :) .+ 1f0) ./ 2f0

    # Vary the size
    recalls = zeros(length(sizes))
    for i = 1:length(sizes)
        println(i)
        recalls[i] = recall(fake_images[:, 1:sizes[i]], real_images, k = k)
    end

    return recalls
end

function vary_k(generator, real_images, n_samples, ks)
    # Get the fake images to cover all sizes
    inputs = get_inputs(n_samples)
    fake_images = generator(inputs)
    fake_images = (reshape(fake_images, 128, :) .+ 1f0) ./ 2f0

    manifold_tree = KDTree(fake_images)

    # Vary the size
    recalls = zeros(length(ks))
    for i = 1:length(ks)
        #println(i)
        recalls[i] = recall(fake_images, manifold_tree, real_images, k = ks[i])
    end

    return recalls
end