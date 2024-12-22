# Performance-Preserving-Optimization-of-Diffusion-Networks

## Project Description
The main goal of the project is to explore and implement various methods to optimize the training of diffusion networks while preserving performance, using PyTorch Profiling tools. We used the CIFAR-10 dataset for our experiments, which was downloaded from Kaggle and converted to a folder set using the `cifar-10` script.

## Repository Structure

├── image_output/ # Folder to store results 

├── trace_output/ # Folder to store Chrome trace files 

|── cifar-10/ # CIFAR-10 datasetPerformance-Preserving-Optimization-of-Diffusion-Networks

├── README.md / # Project description file 

├── denoising_diffusion_pytorch.py / # Diffusion model implementation 

├── denoising_diffusion_pytorch_with_profile.py / # Diffusion model implementation with torch profile

├── try_pytorch_profile.py / # Script for performance analysis using PyTorch Profiler 

├── try_mulit_gpu.py / # Multi-GPU training script 

├── try_amp.py / # AMP (Automatic Mixed Precision) training script

├── try_torch_compile.py / # Script for training with torch.compile 

## Data Preparation


## Execute 

### Performance Analysis

##### cifar-10

（the cifar-10 dataset is too large, it can only run on the desktop, sorry

Please refer to the attach file on the homework page: Final Project Github Repository>cifar-10.zip）

Download [cifar-10](https://www.kaggle.com/c/cifar-10/) and decompress train.7z. The direcotry structure is as below:

cifar-10/
|── train
    |──1.png
    |──2.png
    |──....





Save GitHub files to a folder on the desktop called final
After placing CIFAR-10 on the desktop, modify the code

change every code“c/Users/ThinkPad/Desktop/cifar-10”
into“……your_file……/Desktop/cifar-10”

cd “……your_file……/Desktop/cifar-10"

 run `pip install -r requirements.txt`.



then

（1）To perform performance analysis using PyTorch Profiler:
```
python try_pytorch_profile.py
```
（2）Multi-GPU Training
To train on single GPU and multiple GPUs:

```
python try_mulit_gpu.py
```

（3）AMP Training
To train with and without AMP:

```
python try_amp.py
```



##### About：

Training with torch.compile
To train with and without torch.compile:

install PyTorch和triton

```
python try_torch_compile.py
```



（Running on Ubuntu）cd"/mnt/……your_file……/Desktop/final">python try_torch_compile.py

（ install pytorch）

pip install -U triton
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv -y
pip install torchvision
pip install matplotlib

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


（install triton）

source /root/triton_env/bin/activate

which python

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113



## Results
### Performance Analysis Results
This figure shows the eight most time consuming operations obtained from the analysis of the pytorch profile.
![](./image_output/top_10_longest_events.png)


### Multi-GPU Training Results
The comparison of training times on single GPU and multiple GPUs is shown in the figure below:
![](./image_output/training_time_vs_batch_size.png)


### AMP vs no-AMP Training Results
Based on the previous GPU experi-mental results, we chose a batch size of 128 and conducted experiments using one or two GPUs. We measured the time consumption with enabling and disabling Automatic Mixed Precision (AMP),
![](./image_output/training_time_with_amp.png)

### Training on different dataloader workers
We are studying how the training time varies with the number of dataloader workers starting from zero and increment the number of workers by 4.
![](./image_output/training_time_vs_num_workers.png)


### Torch.compile vs non-Torch.compile
The impact of torch.compile on the performance improvement of model training under the conditions of batch size 128, one or two GPUs, with AMP enabled, and
num workers set to 4 is tested.
![](./image_output/training_time_via_torch_compile.png)


## Observations and Summary

1) Framework Effectiveness: Our framework successfully identified the optimal configurations for training diffusion models. For instance, AMP proved to significantly accelerate training, and setting the num workers to 4 balanced data loading efficiency with minimal overhead. 

2) Scalability and Limitations: While multi-GPU setups provided significant time reductions for larger batch sizes, their scaling efficiency diminished beyond three GPUs. Additionally, torch.compile, despite its general potential, was found unsuitable for this specific diffusion model due to excessive overhead.

3) General Applicability: Beyond diffusion models, this framework can be applied to other deep learning or machine learning domains to systematically analyze and optimize training processes. It provides a clear and reusable methodology for benchmarking and improving training efficiency across a wide range of tasks.
