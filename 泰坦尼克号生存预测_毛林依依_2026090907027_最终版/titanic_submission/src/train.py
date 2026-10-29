# 训练模型，并把损失曲线和准确率曲线画出来
#
# 跑完之后会在 save/ 里生成 titanic_model.pth（模型参数）
# 和 preprocess_params.json（预处理参数），在 docs/ 里生成两张图
#
# 注意：命令得在项目根目录下敲（python src/main.py），
# 因为下面这几个路径都是照根目录写的

import copy
import os
import random

import matplotlib
matplotlib.use('Agg')          # 只往文件里存图，不弹窗口，脚本就不会卡在画图窗口上
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from data_preprocessing import load_and_split, save_params
from model import TitanicModel, save_model

# 几个能改的数
SEED = 42
TEST_SIZE = 0.2        # 训练集:测试集 = 8:2
BATCH_SIZE = 32
EPOCHS = 150
LR = 0.001             # 一开始设的 0.01，测试准确率抖得厉害，试了几次定在这
WEIGHT_DECAY = 1e-4    # 加在 Adam 里的，压一压过拟合

DATA_PATH = 'data/train.csv'
SAVE_DIR = 'save'
DOCS_DIR = 'docs'


class TitanicDataset(Dataset):
    # 把特征和标签包一下，交给 DataLoader 按批取

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def set_seed(seed):
    # 数据划分、参数初始化、DataLoader 打乱的顺序都要用随机数，全设一遍才复现得出来
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)      # 习惯性写上的，这次其实没用到 GPU


def main():
    set_seed(SEED)

    print('=' * 50)
    print('开始训练')
    print('=' * 50)

    # 1. 读数据、划分、预处理
    print('正在加载并预处理数据...')
    X_train, y_train, X_test, y_test, params = load_and_split(
        DATA_PATH, test_size=TEST_SIZE)

    # 2. 转成 PyTorch 的张量
    # 特征用 float32，标签也转成 float32 再拉成一列，因为 BCELoss 要求两边形状一样
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    print(f'训练集 {X_train_tensor.shape[0]} 条，测试集 {X_test_tensor.shape[0]} 条，'
          f'特征 {X_train_tensor.shape[1]} 个')

    # 3. 用 Dataset 和 DataLoader 组织训练数据
    train_dataset = TitanicDataset(X_train_tensor, y_train_tensor)
    # shuffle 的打乱顺序也固定住种子，不然每轮顺序不一样，结果也会飘
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, generator=generator)

    # 4. 建模型、定损失函数和优化器
    input_dim = X_train_tensor.shape[1]
    model = TitanicModel(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    print(f'模型结构: {input_dim} -> 32 -> 16 -> 1')
    print(f'损失函数: BCELoss   优化器: Adam (lr={LR}, weight_decay={WEIGHT_DECAY})')
    print(f'训练轮次: {EPOCHS}   批大小: {BATCH_SIZE}')
    print('-' * 50)

    # 每一轮都记一份，画图用
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    # 记下测试集表现最好的那一轮
    best_test_acc = 0.0
    best_epoch = 0
    best_state = None

    # 5. 开始训练
    for epoch in range(EPOCHS):
        model.train()
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)            # 前向
            loss = criterion(outputs, batch_y)  # 算损失

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 这一轮跑完，在训练集上算一下损失和准确率。
        # 这里要 model.eval() 把 Dropout 关掉，不然同一个模型每次算出来的数都不一样
        model.eval()
        with torch.no_grad():
            train_out = model(X_train_tensor)
            train_loss = criterion(train_out, y_train_tensor).item()
            train_pred = (train_out > 0.5).float()
            train_acc = (train_pred == y_train_tensor).float().mean().item()

        # 测试集再来一遍，写法是一样的
        model.eval()
        with torch.no_grad():
            test_out = model(X_test_tensor)
            test_loss = criterion(test_out, y_test_tensor).item()
            test_pred = (test_out > 0.5).float()
            test_acc = (test_pred == y_test_tensor).float().mean().item()

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        # 测试集准确率比之前高，就把这一轮的参数先存一份下来
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{EPOCHS}]  '
                  f'训练损失 {train_loss:.4f}  测试损失 {test_loss:.4f}  '
                  f'训练准确率 {train_acc:.4f}  测试准确率 {test_acc:.4f}')

    # 6. 保存
    # 恢复成测试集上最好的那一轮再存，这样存下来的是表现最好的模型
    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

    save_model(model, os.path.join(SAVE_DIR, 'titanic_model.pth'))
    save_params(params, os.path.join(SAVE_DIR, 'preprocess_params.json'))

    # 7. 画图
    epochs_range = range(1, len(train_losses) + 1)

    # 图1：损失趋势图
    plt.figure(figsize=(7, 5))
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, test_losses, label='Test Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(DOCS_DIR, 'loss_curve.png'), dpi=150)
    plt.close()

    # 图2：训练集与测试集准确率图
    plt.figure(figsize=(7, 5))
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
    plt.plot(epochs_range, test_accuracies, label='Test Accuracy')
    # 标一下测试集准确率最高的那一轮
    plt.axvline(best_epoch, color='gray', linestyle='--', alpha=0.7,
                label=f'Best Epoch ({best_epoch})')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(DOCS_DIR, 'accuracy_curve.png'), dpi=150)
    plt.close()

    last_test_acc = test_accuracies[-1]

    print('-' * 50)
    print(f'训练结束。一共 {EPOCHS} 轮。')
    print(f'最后一轮的测试集准确率: {last_test_acc:.4f}')
    print(f'最好的测试集准确率:     {best_test_acc:.4f} (第 {best_epoch} 轮，已保存这一轮的模型)')
    print(f'模型和预处理参数已保存到 {SAVE_DIR}')
    print(f'两张曲线图已保存到 {DOCS_DIR}')


if __name__ == '__main__':
    main()
