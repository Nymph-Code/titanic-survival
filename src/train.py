import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 导入前面的
from data_preprocessing import load_and_preprocess_data
from model import TitanicModel

# 设置固定随机种子（保证实验可复现）
SEED = 42
torch.manual_seed(SEED)

# 自定义 Dataset
class TitanicDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def main():
    # 1. 加载数据
    print("正在加载并预处理数据...")
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = load_and_preprocess_data()

    # 2. 组织 DataLoader
    train_dataset = TitanicDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 3. 实例化模型
    input_dim = X_train_tensor.shape[1]
    model = TitanicModel(input_dim)
    print(f"模型已初始化，输入特征维度为: {input_dim}")

    # 4. 设置损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # 5. 开始
    epochs = 100
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    print("开始训练...")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        # 每轮结束后计算训练集准确率
        model.eval()
        with torch.no_grad():
            train_outputs = model(X_train_tensor)
            train_preds = (train_outputs > 0.5).float()
            train_acc = accuracy_score(y_train_tensor.numpy(), train_preds.numpy())
            train_accuracies.append(train_acc)

        # 每10轮评估一次测试集准确率
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                test_preds = (test_outputs > 0.5).float()
                test_acc = accuracy_score(y_test_tensor.numpy(), test_preds.numpy())
                test_accuracies.append(test_acc)
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Test Acc: {test_acc:.4f}")

    # 6. 保存模型与画图
    os.makedirs('../docs', exist_ok=True)
    torch.save(model.state_dict(), '../docs/titanic_model.pth')
    
    plt.figure(figsize=(12, 5))
    
    # 左图：损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 右图：训练和测试准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), train_accuracies, label='Train Accuracy')
    plt.plot(range(10, epochs+1, 10), test_accuracies, marker='o', label='Test Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('../docs/training_curves.png')
    plt.show()
    print("训练结束！模型和图表已保存至 docs 文件夹。")

if __name__ == '__main__':
    main()