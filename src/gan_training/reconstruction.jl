using Flux
using Flux.Data: DataLoader
using Flux: throttle, @epochs, mse
using BSON
using BSON: @save, @load
using Parameters: @with_kw
using HDF5
using Statistics

@with_kw mutable struct Args
    η::Float64 = 0.001  # 3e-4 was here    # learning rate
    batchsize::Int = 1024  # batch size
    epochs::Int = 100   # number of epochs
end

function build_model(layer_sizes, act)
    # ReLU except last layer identity
    layers = Any[Dense(layer_sizes[i], layer_sizes[i+1], act) for i = 1:length(layer_sizes) - 2]
    push!(layers, Dense(layer_sizes[end-1], layer_sizes[end]))
    println("Model created with layers: ", layers)
    return Chain(layers...)
end

function loss_all(dataloader, model)
    l = 0f0
    for (x,y) in dataloader
        l += mse(model(x), y)
    end
    return l/length(dataloader)
end

function eval_fcn!(m, train_data, start_time)
    # Compute the losses and accuracy
    new_train_loss = loss_all(train_data, m)

    # Print updated losses
    println("Train loss: ", new_train_loss)
end

function train(fn; kws...)
    args = Args(; kws...)
    
    # Load Data
    images = h5read(fn, "y_train") # yes I know the labels seem backwards
    images = reshape(images, 128, :)
    y = h5read(fn, "X_train")#[1:2, :]
    println("std1: ", std(y[1, 1:1200]), " std2: ", std(y[2, 1:1200]))
    y[1,:] ./= 10.0 #std(y[1,:])
    y[2,:] ./= 30.0 #std(y[2,:])
    y[3,:] = (y[3, :] .- 322f0) ./ 30f0

    train_data = DataLoader(y, images, batchsize=args.batchsize, shuffle = true)

    # Construct model and loss
    layer_sizes = (3, 256, 256, 256, 256, 128)
    m = build_model(layer_sizes, relu)
    loss(x,y) = mse(m(x), y)

    ## Training
    # lists to store progress
    start_time = time()

    # Setup the evaluation function
    evalcb = () -> @time eval_fcn!(m, train_data, start_time)

    # Choose your optimizer and train
    opt = ADAM(args.η)
    @epochs args.epochs Flux.train!(loss, Flux.params(m), train_data, opt, cb = throttle(evalcb, 20))

    # Show the final loss on train and validation data
    @show loss_all(train_data, m)

    return m
end

fn = "../../data/SK_DownsampledGANFocusAreaData.h5"
G = train(fn)