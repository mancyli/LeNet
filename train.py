import torch
from torch import nn
from LeNet.net import MyLeNet
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import os

# 将数据转化为tensor格式
data_transform = transforms.Compose([
    transforms.ToTensor()
])

# 加载训练数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, download=True)
# 给训练集创建一个数据加载器, shuffle=True用于打乱数据集，每次都会以不同的顺序返回。
# 每一次读取的是一个batch_size大小的数据
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=False)
# 加载测试数据集
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
# 给测试集创建一个数据加载器, shuffle=True用于打乱数据集，每次都会以不同的顺序返回。
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

# 如果显卡可用，则用显卡进行训练
device = "cuda" if torch.cuda.is_available() else 'cpu'

# 调用net里面定义的模型，如果GPU可用则将模型转到GPU
model = MyLeNet().to(device)

# 定义损失函数（交叉熵损失）
loss_fn = nn.CrossEntropyLoss()

# 定义优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 学习率每隔10epoch变为原来的0.1
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# 定义训练函数
def train(dataloader, model):
    loss, current, n = 0.0, 0.0, 0
    # enumerate返回为数据和标签还有批次
    for batch, (X, y) in enumerate(dataloader):
        # 前向传播
        X, y = X.to(device), y.to(device)
        output = model(X)
        # 计算损失函数
        cur_loss = loss_fn(output, y)
        # torch.max返回每行最大的概率和最大概率的索引,由于批次是16，所以返回16个概率和索引
        _, pred = torch.max(output, dim=1)

        # 计算每批次的准确率， output.shape[0]为该批次的多少
        cur_acc = torch.sum(y == pred) / output.shape[0]

        # 反向传播
        optimizer.zero_grad()  # 将梯度归零
        cur_loss.backward()  # 反向传播
        optimizer.step()  # 通过梯度做一步参数更新
        # 取出loss值和精度值
        loss += cur_loss.item()
        current += cur_acc.item()
        n = n + 1

    print('train_loss：' + str(loss / n))
    print('train_acc：' + str(current / n))


# 定义验证函数
def val(dataloader, model):
    # 将模型转为验证模式
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    # 非训练，推理期用到（测试时模型参数不用更新， 所以no_grad）

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            output = model(X)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, dim=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1
        print('val_loss：' + str(loss / n))
        print('val_acc：' + str(current / n))

        return current / n


def tes(dataloader, model, loss_fn):
    model.eval()
    correct, test_loss, total = 0.0, 0.0, 0
    with torch.no_grad():
        for data, label in test_dataloader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            test_loss += loss_fn(output, label).item()
            predict = output.argmax(dim=1)
            total += label.size(0)
            correct += (predict == label).sum().item()
        print('test_loss：' + str(test_loss / total))
        print('test_acc：' + str(correct / total))


# 开始训练
epoch = 20
min_acc = 0
for t in range(epoch):

    print(f"epoch{t + 1}\n-----------------------------")
    train(train_dataloader, model)
    # tes(test_dataloader, model, loss_fn)
    a = val(test_dataloader, model)
    # 保存最好的模型权重文件
    if a > min_acc:
        folder = 'sava_model'
        if not os.path.exists(folder):
            os.mkdir('../sava_model')
        min_acc = a
        print('save best model\n', )
        torch.save(model.state_dict(), "../sava_model/best_model.pth")


    lr_scheduler.step()  # 根据迭代epoch更新学习率

    # 保存最后的权重文件
    # if t == epoch - 1:
    # torch.save(model.state_dict(), "sava_model/last_model.pth")

print('Done！')
