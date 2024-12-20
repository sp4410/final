import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from torch.profiler import profile, record_function, ProfilerActivity
import time

# Initialize the U-Net model and move it to GPU
model = Unet(
    dim=64,                      # The dimension of the input/output
    dim_mults=(1, 2, 4)          # Multipliers for feature dimensions across U-Net layers
).cuda()

# Define the diffusion process and move it to GPU
diffusion = GaussianDiffusion(
    model,
    image_size=32,               # The size of the input images
    timesteps=1000,              # Total diffusion steps
    sampling_timesteps=250,      # Number of timesteps during sampling
    loss_type='l1'               # Type of loss function
).cuda()

# Create a Trainer object to train the diffusion model
trainer = Trainer(
    diffusion,
    '/scratch/vb2184/HPML_Project/Performance-Preserving-Optimization-of-Diffusion-Networks/cifar10/train',  # Path to training dataset
    train_batch_size=128,        # Batch size for training
    train_lr=2e-4,               # Learning rate
    train_num_steps=100,         # Number of training steps
    gradient_accumulate_every=2, # Number of steps to accumulate gradients
    ema_decay=0.995,             # Exponential Moving Average (EMA) decay rate
    amp=True                     # Enable Automatic Mixed Precision (AMP)
)

print("2 GPU")
print("Has AMP")

# The following block is a commented-out example for adding a tracing handler
"""
# Add this line to trace the model
def trace_handler(p):
     print("I got here!!!")  # Debugging message to confirm trace handler is triggered
     output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)  # Summarize key statistics
     print(output)
     # Export profiling results as a Chrome Trace format JSON file
     p.export_chrome_trace('./scratch/vb2184/HPML_Project/' + str(p.step_num) + '.json')

# Use PyTorch profiler with a specific schedule
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,  # Profile CPU activity
    torch.profiler.ProfilerActivity.CUDA],           # Profile CUDA (GPU) activity
     schedule=torch.profiler.schedule(               # Profiling schedule
         wait=2,                                     # Initial steps to skip
         warmup=1,                                   # Warmup steps
         active=3,                                   # Active profiling steps
         repeat=2),                                  # Repeat the schedule twice
     on_trace_ready=trace_handler                    # Use the trace handler function
 ) as profiler:
     for i in range(10):                             # Train the model for 10 iterations
        trainer.train()
        profiler.step()                              # Mark a step in the profiler
     
# Print profiling summary
print(profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
# Export profiling results as a Chrome Trace format JSON file
p.export_chrome_trace('./scratch/vb2184/HPML_Project/' + str(p.step_num) + '.json')
"""

# Record the runtime of the training process
start = time.time()
# Use PyTorch profiler for detailed profiling of the training process
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],  # Profile both CPU and CUDA activities
    record_shapes=True,                                       # Record tensor shapes
    profile_memory=True,                                      # Record memory usage
    with_stack=True,                                          # Include stack traces in the profiling results
    use_cuda=True                                             # Enable CUDA profiling
) as prof:
    trainer.train()                                           # Run the training process

# Print profiling summary sorted by CPU time
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

end = time.time()
# Calculate total runtime of the training process
total_runtime = end - start
print("Total runtime was: ", total_runtime)
