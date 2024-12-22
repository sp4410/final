import time
import torch
import os
import numpy as np
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import pickle
import matplotlib.pyplot as plt

# Define a function to run training and record runtime
def run_training(device, batch_size, amp, num_workers, use_compile, device_info):
    device = torch.device(device)
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4)
    ).to(device)

    diffusion = GaussianDiffusion(
        model,
        image_size=32,
        timesteps=1000,
        sampling_timesteps=250,
        loss_type='l1'
    ).to(device)

    # Use torch.compile with a fallback mechanism
    if use_compile:
        try:
            diffusion = torch.compile(diffusion)
            print("torch.compile successfully enabled")
        except RuntimeError as e:
            print(f"torch.compile failed: {e}")
            print("Falling back to eager execution mode")
            use_compile = False

    # Generate random input data for warming up
    dummy_data = torch.randn(batch_size, 3, 32, 32).to(device)
    diffusion(dummy_data)

    total_data = (16 * 180) * 16

    trainer = Trainer(
        diffusion,
        '/mnt/c/Users/ThinkPad/Desktop/cifar-10',
        train_batch_size=batch_size,
        train_lr=2e-4,
        train_num_steps=(total_data // batch_size) // 60,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        amp=amp,
        use_cpu=False,
        num_workers=num_workers
    )

    print(f"Running on {device_info} with batch size {batch_size}, AMP {'enabled' if amp else 'disabled'}, "
          f"num_workers {num_workers}, torch.compile {'enabled' if use_compile else 'disabled'}")

    start = time.time()
    trainer.train()
    end = time.time()
    total_runtime = end - start
    print(f"Total runtime: {total_runtime:.2f} seconds")
    return total_runtime

# Main program
batch_size = 128
devices = ['cuda:0', 'cuda:0,1']
amp = True
use_compile_settings = [True, False]
num_workers = 4
results = {f"{device}_compile_{use_compile}": [] for device in devices for use_compile in use_compile_settings}

results_file = 'try_torch_compile.pk'

# Check if saved results exist
if os.path.exists(results_file):
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
else:
    for device in devices:
        if device == 'cuda:0,1':
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = device.split(':')[1]

        for use_compile in use_compile_settings:
            runtime = run_training(devices[0], batch_size, amp, num_workers, use_compile, device_info=device)
            results[f"{device}_compile_{use_compile}"].append(runtime * 5 * 60)

    # Save results to file
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)

# Print results
print(results)

# Plot bar chart
labels = list(results.keys())
times = [results[label][0] for label in labels]

plt.figure(figsize=(12, 6))
plt.bar(labels, times, color=['skyblue', 'orange'] * (len(labels) // 2))
plt.xlabel('Configuration')
plt.ylabel('Total Runtime (seconds)')
plt.title('Training Time with and without torch.compile on Different GPU Configurations')
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.savefig('image_output/training_time_via_torch_compile.png')
plt.show()
