import torch
from LeNet.net import MyLeNet
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage

# 数据转化为tensor格式
data_transform = transforms.Compose([
    transforms.ToTensor()
])

# 加载训练数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, download=True)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

# 加载测试数据集
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

# 如果有显卡，可以转到GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 调用net里面定义的模型，将模型类型数据转到GPU
model = MyLeNet().to(device)

# 加载 train.py 里训练好的模型
model.load_state_dict(torch.load("sava_model/best_model.pth"))

# 获取结果
classes = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
]

# 把tensor转成Image， 方便可视化
show = ToPILImage()

# 进入验证阶段
model.eval()
# 对test_dataset里10000张手写数字图片进行推理

for i in range(len(test_dataloader)):
    x, y = test_dataset[i][0], test_dataset[i][1]
    # tensor格式数据可视化
    # show(x).show()
    # 扩展张量维度为4维
    x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False).to(device)
    with torch.no_grad():
        pred = model(x)
        # 得到预测类别中最高的那一类，再把最高的这一类对应classes中的哪一类标签
        predicted, actual = classes[torch.argmax(pred[0])], classes[y]
        # 最终输出的预测值与真实值
        print(f'predicted: "{predicted}", actual:"{actual}"')
