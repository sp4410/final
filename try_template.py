import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from torch.profiler import profile, record_function, ProfilerActivity
import time
model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4)
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = 32,
    timesteps = 1000,
    sampling_timesteps = 250,
    loss_type = 'l1'
).cuda()

trainer = Trainer(
    diffusion,
    '/scratch/vb2184/HPML_Project/Performance-Preserving-Optimization-of-Diffusion-Networks/cifar10/train',
    train_batch_size = 128,
    train_lr = 2e-4,
    train_num_steps = 100,
    gradient_accumulate_every = 2,
    ema_decay = 0.995,
    amp = True
)
print("2 GPU")
print("Has AMP")




"""
# Add this line to trace the model
def trace_handler(p):
     print("I got here!!!")
     output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
     print(output)
     p.export_chrome_trace('./scratch/vb2184/HPML_Project/' + str(p.step_num) + '.json')



with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
    torch.profiler.ProfilerActivity.CUDA,
    ],
     schedule=torch.profiler.schedule(
         wait=2,
         warmup=1,
         active=3,
         repeat=2),
     on_trace_ready=trace_handler
 ) as profiler:
     for i in range(10):
        trainer.train()
        profiler.step()
     
print(profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
p.export_chrome_trace('./scratch/vb2184/HPML_Project/' + str(p.step_num) + '.json')
"""
start=time.time()
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    use_cuda=True
) as prof:
    trainer.train()
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
end=time.time()
total_runtime = end- start
print("Total runtime was: ",total_runtime)
