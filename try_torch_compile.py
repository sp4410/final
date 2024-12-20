import time
import torch, os
import numpy as np
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from torch.profiler import profile, ProfilerActivity
import pickle
import matplotlib.pyplot as plt

# Define a function to run training and record the runtime
def run_training(device, batch_size, amp, num_workers, use_compile, device_info):
    device = torch.device(device)

    # Initialize the U-Net model
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4)
    ).to(device)

    # Configure the Gaussian Diffusion model
    diffusion = GaussianDiffusion(
        model,
        image_size=32,
        timesteps=1000,
        sampling_timesteps=250,
        loss_type='l1'  # L1 loss is used here
    ).to(device)

    # Enable torch.compile if specified
    if use_compile:
        diffusion = torch.compile(diffusion)

    # Generate dummy data for warming up the model
    dummy_data = torch.randn(batch_size, 3, 32, 32).to(device)
    diffusion(dummy_data)

    # Calculate total data size for training
    total_data = (16 * 180) * 16

    # Set up the Trainer for the diffusion model
    trainer = Trainer(
        diffusion,
        './cifar-10/train',
        train_batch_size=batch_size,
        train_lr=2e-4,
        train_num_steps=(total_data // batch_size) // 60,  # Number of training steps
        gradient_accumulate_every=2,  # Gradient accumulation steps
        ema_decay=0.995,  # Exponential moving average decay
        amp=amp,  # Automatic Mixed Precision (AMP) setting
        use_cpu=False,
        num_workers=num_workers  # Number of data loading workers
    )

    # Print configuration details
    print(f"Running on {device_info} with batch size {batch_size}, AMP {'enabled' if amp else 'disabled'}, num_workers {num_workers}, torch.compile {'enabled' if use_compile else 'disabled'}")

    # Measure training runtime
    start = time.time()
    trainer.train()
    end = time.time()
    total_runtime = end - start
    print(f"Total runtime on {device_info} with batch size {batch_size}, AMP {'enabled' if amp else 'disabled'}, num_workers {num_workers}, torch.compile {'enabled' if use_compile else 'disabled'}: {total_runtime} seconds")
    return total_runtime




# Parameters for training
batch_size = 128
devices = ['cuda:0', 'cuda:0,1']  # Single and multi-GPU configurations
amp = True  # Automatic Mixed Precision enabled
use_compile_settings = [True, False]  # Test with and without torch.compile
num_workers = 4

# Initialize results dictionary
results = {f"{device}_compile_{use_compile}": [] for device in devices for use_compile in use_compile_settings}

# File to save results
results_file = 'try_torch_compile.pk'

# Check if results file already exists
if os.path.exists(results_file):
    # Load results if they are already computed
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
else:
    # Run training for each configuration
    for device in devices:
        if device == 'cuda:0,1':
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Set visible GPUs for multi-GPU
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = device.split(':')[1]  # Set GPU for single-GPU

        for use_compile in use_compile_settings:
            # Run training and record runtime
            runtime = run_training(devices[0], batch_size, amp, num_workers, use_compile, device_info=device)
            results[f"{device}_compile_{use_compile}"].append(runtime * 5 * 60)

    # Save results to a file for future use
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)

# Print the results
print(results)

# Plot the results as a bar chart
labels = list(results.keys())
times = [results[label][0] for label in labels]

plt.figure(figsize=(12, 6))
plt.bar(labels, times, color=['skyblue', 'orange', 'skyblue', 'orange'])
plt.xlabel('Configuration')
plt.ylabel('Total Runtime (seconds)')
plt.title('Training Time with and without torch.compile on Different GPU Configurations')
plt.xticks(rotation=0)
plt.grid(True)

# Save the chart as a PNG file
plt.savefig('image_output/training_time_via_torch_compile.png')
