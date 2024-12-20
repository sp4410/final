import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 数据集
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 模型
model = models.resnet18(pretrained=False).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 使用 torch.profiler
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA  # 如果使用 GPU
    ],
    schedule=torch.profiler.schedule(
        wait=2, warmup=2, active=6  # 跳过前2步，预热2步，记录6步
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),  # 保存到 TensorBoard 日志
    record_shapes=True,
    with_stack=True
) as profiler:
    for step, (inputs, labels) in enumerate(train_loader):
        if step >= 10:  # 限制记录步数
            break
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 后向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 优化
        optimizer.step()
        profiler.step()  # 每步记录

print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))
profiler.export_chrome_trace('./trace_output/' + 'test' + '.json')