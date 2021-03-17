# CS236G_VerifyGAN
Code for CS236G course project on using GANs to verify image-based controllers.

**Contributors**: Sydney Katz, Anthony Corso, Chris Strong

## data
The data folder contains two datasets from images of a runway taken from a camera angle of the right wing of a Cessna 208B Grand Caravan taxiing down runway 04 of Grant County International Airport in the [X-Plane 11 flight simulator](https://www.x-plane.com/). The images are downsampled to 8 x 16 images and converted to grayscale.

`KJ_DownsampledTrainingData.h5` - original data used to train the image-based neural controller for taxiing down the runway; created by running snaking trajectories down the entire runway. The X data contains the images and the y values are crosstrack error (meters) and heading error (degrees) respectively.

To load in Julia:
```julia
using HDF5
X_train = h5read("KJ_DownsampledTrainingData.h5", "X_train") # 16 x 8 x 51463 
y_train = h5read("KJ_DownsampledTrainingData.h5", "y_train") # 2 x 51463
X_val = h5read("KJ_DownsampledTrainingData.h5", "X_val") # 16 x 8 x 7386
y_val = h5read("KJ_DownsampledTrainingData.h5", "y_val") # 2 x 7386
```

`SK_DownsampledGANFocusAreaData.h5` - data from a clean 200 meter stretch of the runway generated specifically for the GAN; created by sampling a random locations in the 200 meter stretch. Since this dataset was created specifically to train the GAN, there is no validation set and X values are the labels of crosstrack error (meters), heading error (degrees), and downtrack position (meters) and the y values are the images.

To load in Julia:
```julia
using HDF5
X_train = h5read("SK_DownsampledGANFocusAreaData.h5", "X_train") # 3 x 10000
y_train = h5read("SK_DownsampledGANFocusAreaData.h5", "y_train") # 16 x 8 x 10000
```

## models
The control network that takes in a downsampled runway image and predicts crosstrack error and heading error is saved in both the [NNet](https://github.com/sisl/NNet) (`KJ_TaxiNet.nnet`) and [Flux](https://fluxml.ai/) (`taxinet.bson`) format.

There are currently two full models consisting on the big MLP and small MLP generator concatenated with the taxinet control network (they go from two latent variables and the crosstrack and heading error to a prediction of the crosstrack and heading error) in both formats as well.

## gan training
The code for training the GANs as well as some of the saved generators can be found in `src/gan_training`. The file `reconstruction.jl` contains the code for training a reconstruction loss baseline. The file `cGAN_common.jl` contains functions for training a conditional GAN, the file `spectral_norm.jl` implements spectral normalization layers in Flux to be used by the discriminator, and the file `taxi_models_and_data` implements functions specific to the taxinet problem and is the main file to call for training the GAN. The `train_smaller_generator` file contains the code to train a smaller generator in a superivised learning fashion.

To run the training code, ensure that all necessary julia packages are installed and then run:
```julia
include("taxi_models_and_data.jl")
```
This code was developed and tested using Julia1.5.

The settings data structures allows for easy specification on training settings:

```julia
@with_kw struct Settings
	G # Generator to train
	D # Discriminator to train
	loss # Loss function
	img_fun # Function to load in the image data
	rand_input # Function to generate a random input for the generator
	batch_size::Int = 128
	latent_dim::Int = 100 # Number of latent variables
	nclasses::Int = 2 # Number of input variables (crosstrack error and heading error)
	epochs::Int = 120
	verbose_freq::Int = 2000 # How often to print and save training info
	output_x::Int = 6 # Size of image output examples
	output_y::Int = 6 # Size of image output examples
	optD = ADAM(0.0002, (0.5, 0.99))
	optG = ADAM(0.0002, (0.5, 0.99))
	output_dir = "output" # Folder to save outputs to
	n_disc = 1 # Number of discriminator training steps between generator training step
end
```

## gan evaluation
The file `gan_evaluation.jl` contains code for calculating recall for various k values and sample sizes. To calculate the recall for a generator network, run:

```julia
g256x4msle = BSON.load("../gan_training/generators/mlp256x4_msle_generator_uniform.bson")[:g];

fn = "../../data/SK_DownsampledGANFocusAreaData.h5"
images = h5read(fn, "y_train")
real_images = reshape(images, 16*8, :)
y = h5read(fn, "X_train")[1:2, :];

recalls256x4msle = vary_sample_size(g256x4msle, sizes, real_images; k = 30);
```

## xplane interface
The python files to interface with the XPlane simulator are contained in the `src/xplane_interface` folder. More information about the `xpc3.py` file created by NASA X-Plane Connect can be found [here](https://github.com/nasa/XPlaneConnect). Other notable files include `genGANData.py` and `SK_genTrainingData.py`, which are used to generate and downsample the GAN training data respectively. The `sim_network.py` file allows us to simulate our simple dynamics model using X-Plane 11 images to drive the controller.

## verification
The verification code relies on the [MIPVerifyWrapper](https://github.com/castrong/MIPVerifyWrapper) repository. The `verify.jl` file located in `src/verification` loads in the necessary files from MIPVerifyWrapper and contains a functions for computing the minimum and maximum control output over a given region in the generator's input space and for doing so for each cell in an input space.

To divide up the input space and run verification, run the following lines of code:

```julia
include("verify.jl")

network = read_nnet("../../models/full_big_normal_v2.nnet")

num_inp = size(network.layers[1].weights, 2)
strategy = MIPVerify.mip
timeout_per_node = 0.5
main_timeout = 10.0
mipverify_network = network_to_mipverify_network(network, "test", strategy)

max_widths = [0.2, 0.5] # Maximum cell widths
lbs = [-11.0, -30.0]
ubs = [11.0, 30.0]

tree = create_tree(max_widths, lbs, ubs)
verify_tree!(tree, mipverify_network, num_inp, main_solver, tightening_solver)
```

To run probabilistic model checking on the tree, run:

```julia
label_tree_failures!(tree)
model_check!(tree)
```

To run forward reachability analysis, run:

```julia
lbs = [-30.0, -11.0] # Lower bounds of region of start states
ubs = [30.0, 11.0] # Upper bounds of region of start states
label_start_states!(tree, lbs, ubs)
trees = forward_reach(tree)
```

Larger networks may require approximate verification techniques to run in finite time. To use an approximate approach instead of including the `verify.jl` file include the `approximate.jl` file.

The `viz/` folder contains code for plotting the results.
