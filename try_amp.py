import time
import torch,os
import numpy as np
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from torch.profiler import profile, ProfilerActivity
import matplotlib.pyplot as plt

# 定义一个函数来运行训练并记录时间
def run_training(device, batch_size, amp):
    device = torch.device(device)
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4)
    ).to(device)

    total_data = (16*180) * 16

    diffusion = GaussianDiffusion(
        model,
        image_size = 32,
        timesteps = 1000,
        sampling_timesteps = 250,
        loss_type = 'l1'
    ).to(device)
    trainer = Trainer(
        diffusion,
        './cifar-10/train',
        train_batch_size=batch_size,
        train_lr=2e-4,
        train_num_steps=(total_data//batch_size)//60,  # 10000
        gradient_accumulate_every=2,
        ema_decay=0.995,
        amp=amp,
        use_cpu=False
    )
    print(f"Running on {device} with batch size {batch_size} and AMP {'enabled' if amp else 'disabled'}")

    start = time.time()
    trainer.train()
    end = time.time()
    total_runtime = end - start
    print(f"Total runtime on {device} with batch size {batch_size} and AMP {'enabled' if amp else 'disabled'}: {total_runtime} seconds")
    return total_runtime
# 运行训练并记录时间
batch_size = 128
devices = ['cuda:0', 'cuda:0,1']
amp_settings = [False, True]
results = {f"{device}_amp_{amp}": [] for device in devices for amp in amp_settings}

for device in devices:
    if device == 'cuda:0,1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = device.split(':')[1]

    for amp in amp_settings:
        runtime = run_training(devices[0], batch_size, amp)
        results[f"{device}_amp_{amp}"].append(runtime* 5 * 60)

# 绘制柱状图
labels = list(results.keys())
times = [results[label][0]  for label in labels]
print(results)

plt.figure(figsize=(12, 6))
colors = ['skyblue' if 'cuda:0,1' in label else 'orange' for label in labels]
plt.bar(labels, times, color=colors)
plt.xlabel('Configuration')

# Add patterns and annotations
for i, label in enumerate(labels):
    if i in [2,3]:
        plt.bar(label, times[i], color=colors[i], hatch='//')
        plt.text(i, times[i], 'multi-GPU', ha='center', va='bottom')
    else:
        plt.bar(label, times[i], color=colors[i])
        plt.text(i, times[i], 'single-GPU', ha='center', va='bottom')
plt.ylabel('Total Runtime (seconds)')
plt.title('Training Time with and without AMP on Different GPU Configurations')
plt.xticks(rotation=45)
plt.grid(True)
plt.xticks(rotation=0)
plt.show()
plt.savefig('image_output/training_time_with_amp.png')