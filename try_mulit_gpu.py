import time 
import torch, os
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from torch.profiler import profile, ProfilerActivity
import matplotlib.pyplot as plt

# Define a function to run training and record the runtime
def run_training(device, batch_size, use_cpu=False):
    device = torch.device(device)
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4)
    ).to(device)

    total_data = (16 * 180) * 16

    diffusion = GaussianDiffusion(
        model,
        image_size=32,
        timesteps=1000,
        sampling_timesteps=250,
        loss_type='l1'
    ).to(device)
    trainer = Trainer(
        diffusion,
        'C:/Users/ThinkPad/Desktop/cifar-10/train',
        train_batch_size=batch_size,
        train_lr=2e-4,
        train_num_steps=(total_data // batch_size) // 60,  # 10000
        gradient_accumulate_every=2,
        ema_decay=0.995,
        amp=False,
        use_cpu=use_cpu,
    )
    print(f"Running on {device} with batch size {batch_size}")

    start = time.time()
    # Use PyTorch profiler to capture runtime statistics
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        trainer.train()
    end = time.time()
    total_runtime = end - start
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
    print(f"Total runtime on {device} with batch size {batch_size}: {total_runtime} seconds")
    return total_runtime

# Main program entry point
if __name__ == '__main__':
    batch_sizes = [32, 64, 128, 256, 512]
    devices = ['cpu', 'cuda:0']
    # Uncomment below to test multi-GPU setups
    # devices = ['cpu', 'cuda:0', 'cuda:0,1','cuda:0,1,2','cuda:0,1,2,3']
    results = {device: [] for device in devices}

    for device in devices:
        if device == 'cuda:0,1':
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        elif device == 'cuda:0':
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        for batch_size in batch_sizes:
            runtime = run_training(device, batch_size, True if device == 'cpu' else False)
            results[device].append(runtime)

    # Plot line chart to visualize training time vs batch size
    plt.figure(figsize=(10, 6))
    for device in devices:
        plt.plot(batch_sizes, [t * 500 for t in results[device]], label=device)

    plt.xlabel('Batch Size', fontsize=14)
    plt.ylabel('Total Runtime (seconds)', fontsize=14)
    plt.title('Training Time vs Batch Size on Different Devices', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Ensure output directory exists
    output_dir = 'image_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the plot as an image
    plt.savefig(os.path.join(output_dir, 'training_time_vs_batch_size.png'))
    plt.close()
