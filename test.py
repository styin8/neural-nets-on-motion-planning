import torch

# 定义两个输入张量
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# 使用torch.stack进行堆叠
c = torch.cat((a, b), dim=0)

print((c > 2).int())
