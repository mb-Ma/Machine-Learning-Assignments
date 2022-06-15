import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from tensorboardX import SummaryWriter
import torchvision.utils as vutils


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=1)  # 卷积核个数10 卷积核尺寸 5 步长1 无边缘填充
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2 maxpool
        self.conv2 = nn.Conv2d(10, 10, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2 maxpool
        self.fc1 = nn.Linear(4 * 4 * 10, 100)  # 全连接层
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):  # 28*28*1
        x = F.relu(self.conv1(x))  # 24x24x10
        x = self.pool(x)  # 12x12x10
        x = F.relu(self.conv2(x))  # 8x8x10
        x = self.pool(x)  # 4x4x10
        x = x.view(-1, 4 * 4 * 10)  # flattening
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = torch.load('Model.pth')  # 加载模型
batch_size = 100
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True,
                   transform=transforms.Compose([transforms.ToTensor()])), batch_size=batch_size, shuffle=True)
total_acc = 0
for images, labels in test_loader:
    images = images.cuda()
    labels = labels.cuda()
    output = model(images)
    total_acc += torch.sum(torch.max(output, dim=1)[1] == labels).item() * 1.0

print("Test accuracy :", total_acc / len(test_loader.dataset))

# 保存Cnn模型的卷积和可视化 在当前路径的Terminal输入       tensorboard --logdir runs
writer = SummaryWriter('runs/cnn_mnist', comment='feature map1')
for i, data in enumerate(test_loader, 0):
    # 获取训练数据
    inputs, labels = data
    x = inputs[2].unsqueeze(0)
    y_label = labels[2].unsqueeze(0)
    x = x.cuda()
    break
img_grid = vutils.make_grid(x, normalize=True, scale_each=True, nrow=2)
writer.add_image(format(y_label.item()), img_grid, global_step=0)
model.eval()
for name, layer in model._modules.items():
    if 'fc' in name:
        break
    print(x.size())
    x = layer(x)
    print(format(name))

    # 查看卷积层的特征图 # tensorboard --logdir runs
    if 'conv' in name:
        x1 = x.transpose(0, 1)  # C，B, H, W  ---> B，C, H, W
        img_grid = vutils.make_grid(x1, normalize=True, scale_each=True, nrow=4)  # normalize进行归一化处理
        writer.add_image(format(name), img_grid, global_step=1)


