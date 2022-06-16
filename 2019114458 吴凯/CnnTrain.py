import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import joblib

torch.manual_seed(2)
torch.cuda.manual_seed_all(2)


def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    return image
# 四层的CNN


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


train_ds = datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
batch_size = 100
validation_split = .1
shuffle_dataset = True
random_seed = 2

# 为训练和验证拆分创建数据索引：
dataset_size = len(train_ds)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
# Creating PT data samplers and loaders:
train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)  # 随机地从原始的数据集中抽样数据
valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                                                sampler=valid_sampler)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True,
                   transform=transforms.Compose([transforms.ToTensor()])), batch_size=batch_size, shuffle=True)

fig, ax = plt.subplots(nrows=2,ncols=3)

# 查看数据 可视化
i = 0
for row in ax:
  for col in row:
    col.imshow(im_convert(train_loader.dataset[i][0]))
    col.set_title("digit "+str(train_loader.dataset[i][1]))
    col.axis("off")
    i+=1

model = Net().cuda()  # 切换到GPU运算
print(model)
# optimizer = optim.Adam(model.parameters(),lr=0.01,betas=(0.9, 0.999),eps=1e-08,weight_decay=0,amsgrad=False)
# optimizer = optim.SGD(model.parameters(), lr=0.01)  # 定义优化器 lr = 0.01
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
# optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.5)

criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数

train_errors = []
train_acc = []
val_errors = []
val_acc = []
n_train = len(train_loader) * batch_size
n_val = len(validation_loader) * batch_size

for i in range(10):
    total_loss = 0
    total_acc = 0
    c = 0
    for images, labels in train_loader:
        images = images.cuda()  # 切换为GPU数据
        labels = labels.cuda()

        optimizer.zero_grad()   # 梯度设置为0
        output = model(images)
        loss = criterion(output, labels)    # 计算交叉熵loss
        loss.backward()  # 反向传播
        optimizer.step()    # 更新参数

        total_loss += loss.item()
        total_acc += torch.sum(torch.max(output, dim=1)[1] == labels).item() * 1.0
        c += 1

    # validation

    total_loss_val = 0
    total_acc_val = 0
    c = 0
    for images, labels in validation_loader:
        images = images.cuda()
        labels = labels.cuda()
        output = model(images)
        loss = criterion(output, labels)

        total_loss_val += loss.item()
        total_acc_val += torch.sum(torch.max(output, dim=1)[1] == labels).item() * 1.0
        c += 1

    train_errors.append(total_loss / n_train)
    train_acc.append(total_acc / n_train)
    val_errors.append(total_loss_val / n_val)
    val_acc.append(total_acc_val / n_val)

print("Trainig complete")
print("Validation accuracy :", val_acc[-1])

# 保存模型
# torch.save(model, 'runs/Model.pth')

# 可视化保存模型
writer = SummaryWriter('runs/cnn_mnist')
dummy_input = torch.rand(100, 1, 28, 28)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 转为 torch.cuda.FloatTensor
dummy_input = dummy_input.to(device)
writer.add_graph(model, dummy_input)
writer.close()

# 保存x用于画图
# joblib.dump(val_errors, 'loss005.pkl')

fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].plot(train_errors, 'r', label="Train")
ax[0].plot(val_errors, 'g', label="Validation")
ax[0].grid()
ax[0].set_title("Loss picture")
ax[0].set_ylabel("Loss (cross-entropy)")
ax[0].set_xlabel("Epoch")
ax[0].legend()
ax[1].plot(train_acc, 'r', label="Train")
ax[1].plot(val_acc, 'g', label="Validation")
ax[1].grid()
ax[1].set_title("Accuracy picture")
ax[1].set_ylabel("Accuracy")
ax[1].set_xlabel("Epoch")
ax[1].legend()
plt.show()

total_acc = 0
for images, labels in test_loader:
    images = images.cuda()
    labels = labels.cuda()
    output = model(images)
    total_acc += torch.sum(torch.max(output, dim=1)[1] == labels).item() * 1.0

print("Test accuracy :", total_acc / len(test_loader.dataset))



