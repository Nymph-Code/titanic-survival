import torch.nn as nn

class TitanicModel(nn.Module):
    def __init__(self, input_dim):
        super(TitanicModel, self).__init__()
        # 定义网络层：输入层 -> 隐藏层(16个神经元) -> ReLU激活 -> 输出层(1个神经元) -> Sigmoid
        self.fc1 = nn.Linear(input_dim, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
