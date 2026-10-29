# 二分类模型的结构
#
# 输入 10 个特征，输出 1 个数，过了 Sigmoid 之后是 0~1 的生存概率，大于 0.5 就算幸存

import torch
import torch.nn as nn


class TitanicModel(nn.Module):
    def __init__(self, input_dim):
        super(TitanicModel, self).__init__()

        # 输入(10) → 隐藏层1(32) → 隐藏层2(16) → 输出(1)
        # 一开始只写了一层 16 个神经元的，训练准确率一直上不去，后来加宽成两层
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)      # 每次随机丢 20%，压一压过拟合
        self.sigmoid = nn.Sigmoid()         # 最后一层压到 0~1

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


def save_model(model, path):
    # 只存网络的参数，结构在代码里已经写死了
    torch.save(model.state_dict(), path)


def load_model(path, input_dim):
    # 先按和训练时一样的结构建个空模型，再把存好的参数读进去
    model = TitanicModel(input_dim)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    return model
