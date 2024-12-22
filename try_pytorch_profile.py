import torch 
from denoising_diffusion_pytorch_with_profile import Unet, GaussianDiffusion, Trainer
from torch.profiler import profile, record_function, ProfilerActivity
import time
import json
import os
import matplotlib.pyplot as plt

# Main program entry point
if __name__ == '__main__':
    # Initialize the model
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4)
    ).cuda()

    # Initialize the diffusion process
    diffusion = GaussianDiffusion(
        model,
        image_size=32,
        timesteps=1000,
        sampling_timesteps=250,
        loss_type='l1'
    ).cuda()

    # Initialize the trainer
    trainer = Trainer(
        diffusion,
        'C:/Users/ThinkPad/Desktop/cifar-10/train',
        train_batch_size=128,
        train_lr=2e-4,
        train_num_steps=10,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        amp=False,
        num_workers=1
    )

    print("Discover bottleneck without any optimization")

    # Create the directory to ensure the trace file save path exists
    output_dir = './trace_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define a trace handler for profiling
    def trace_handler(p):
        output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=30)
        print(output)
        p.export_chrome_trace(os.path.join(output_dir, f'{p.step_num}.json'))

    # Use the profiler for performance tracing
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=trace_handler
    ) as p:
        trainer.train(p)

    # Parse the generated JSON trace file and visualize
    trace_file = os.path.join(output_dir, '10.json')
    if os.path.exists(trace_file):
        # Load the JSON trace file
        with open(trace_file, 'r') as f:
            trace_data = json.load(f)

        # Extract events
        events = trace_data['traceEvents']

        # Filter events with the 'dur' (duration) field
        durations = [(event['name'], event['dur']) for event in events if 'dur' in event]

        # Sort and select the top 10 longest events
        top_durations = sorted(durations, key=lambda x: x[1], reverse=True)[:10]

        # Separate names and durations
        names, times = zip(*top_durations)

        # Plot a horizontal bar chart
        plt.figure(figsize=(10, 6))
        plt.barh(names, times, color='skyblue')
        plt.xlabel('Duration (us)')
        plt.title('Top 10 Longest Events')
        plt.gca().invert_yaxis()  # Display the longest event at the top
        plt.show()
    else:
        print(f"Trace file {trace_file} not found. Ensure profiling completed correctly.")
