
import torch
pthfile = r'sava_model/last_model.pth'
net = torch.load(pthfile)
print(net)