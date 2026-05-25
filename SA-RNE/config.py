# SA-RNE residual estimator hyperparameters
depth = 7
n_channels = 32
image_channels = 1
kernel_size = 3

# LCP hyperparameters
lcp_kernel_sizes = [3, 5, 7]
lcp_use_attention = True
lcp_window_size   = 4

# Loss weights
lambda_tv   = 0.6
lambda_grad = 1.0
epsilon     = 1e-3

# Training hyperparameters
learning_rate = 3e-4
batch_size    = 4
patch_size    = 256
epochs        = 100

# Visualization controls
vis_save_every = 10
vis_num_images = 2
ckpt_save_every = 10

# Paths
dataset_dir = './datasets/train'
outputs_dir = './outputs'
