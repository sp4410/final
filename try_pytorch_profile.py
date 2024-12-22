import torch
from denoising_diffusion_pytorch_with_profile import Unet, GaussianDiffusion, Trainer
from torch.profiler import profile, record_function, ProfilerActivity
import time
import json
import matplotlib.pyplot as plt

# Define the Unet model
model = Unet(
    dim=64,
    dim_mults=(1, 2, 4)
).cuda()

# Define the Gaussian Diffusion model
diffusion = GaussianDiffusion(
    model,
    image_size=32,
    timesteps=1000,
    sampling_timesteps=250,
    loss_type='l1'
).cuda()

# Define the Trainer
trainer = Trainer(
    diffusion,
    './cifar-10/train',
    train_batch_size=128,
    train_lr=2e-4,
    train_num_steps=10,
    gradient_accumulate_every=2,
    ema_decay=0.995,
    amp=False,
    num_workers=1
)

print("Discover bottleneck without any optimization")

# Add a handler for profiling output
def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=30)
    print(output)
    p.export_chrome_trace('./trace_output/' + str(p.step_num) + '.json')

# Ensure proper multiprocessing initialization
if __name__ == "__main__":
    # Profile the model
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=trace_handler
    ) as p:
        trainer.train(p)

    # Load the JSON trace file
    with open('./trace_output/10.json', 'r') as f:
        trace_data = json.load(f)

    # Extract events
    events = trace_data['traceEvents']

    # Filter out events with 'dur' (duration) field
    durations = [(event['name'], event['dur']) for event in events if 'dur' in event]

    # Sort by duration and take the top 10
    top_durations = sorted(durations, key=lambda x: x[1], reverse=True)[:10]

    # Separate names and durations for plotting
    names, times = zip(*top_durations)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.barh(names, times, color='skyblue')
    plt.xlabel('Duration (us)')
    plt.title('Top 10 Longest Events')
    plt.gca().invert_yaxis()  # Invert y-axis to have the longest event at the top
    plt.show()
