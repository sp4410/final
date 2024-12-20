import time
import torch,os
import numpy as np
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from torch.profiler import profile, ProfilerActivity
import matplotlib.pyplot as plt

# 定义一个函数来运行训练并记录时间
def run_training(device, batch_size, amp, num_workers,device_info):
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
        use_cpu=False,
        num_workers=num_workers
    )
    print(f"Running on {device_info} with batch size {batch_size}, AMP {'enabled' if amp else 'disabled'}, num_workers {num_workers}")

    start = time.time()
    trainer.train()
    end = time.time()
    total_runtime = end - start
    print(f"Total runtime on {device} with batch size {batch_size}, AMP {'enabled' if amp else 'disabled'}, num_workers {num_workers}: {total_runtime} seconds")
    return total_runtime

# 运行训练并记录时间
batch_size = 128
devices = ['cuda:0', 'cuda:0,1']
amp_settings = [False, True]
num_workers_list = [0, 4, 8, 12, 16, 20]
results = {f"{device}_amp_{amp}": [] for device in devices for amp in amp_settings}

for device in devices:
    if device == 'cuda:0,1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = device.split(':')[1]

    for amp in amp_settings:
        for num_workers in num_workers_list:
            runtime = run_training(devices[0], batch_size, amp, num_workers,device)
            results[f"{device}_amp_{amp}"].append(runtime * 5 * 60)

print(results)
# 绘制折线图
plt.figure(figsize=(12, 6))
for key in results:
    plt.plot(num_workers_list, results[key], label=key)

plt.xlabel('Number of DataLoader Workers')
plt.ylabel('Total Runtime (seconds)')
plt.title('Training Time with Different Number of DataLoader Workers')
plt.legend()
plt.grid(True)
plt.savefig('image_output/training_time_vs_num_workers.png')