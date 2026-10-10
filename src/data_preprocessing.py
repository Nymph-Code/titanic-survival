import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

# 随机种子
SEED = 42

def load_and_preprocess_data():
    # 1. 读取
    train_df = pd.read_csv('../data/raw/train.csv')

    # 2. 处理缺失值
    train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
    train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
    train_df = train_df.drop(columns=['Cabin'])

    # 3. 特征工程
    train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
    train_df = pd.get_dummies(train_df, columns=['Embarked'], drop_first=True)

    # 提取称呼
    train_df['Title'] = train_df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    train_df['Title'] = train_df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    train_df['Title'] = train_df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    train_df['Title'] = train_df['Title'].replace('Mme', 'Mrs')
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    train_df['Title'] = train_df['Title'].map(title_mapping).fillna(0)

    # 计算家庭人数
    train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1

    # 删除无用列
    train_df = train_df.drop(columns=['PassengerId', 'Name', 'Ticket'])

    # 4. 分离特征 X 和标签 y
    X = train_df.drop(columns=['Survived'])
    y = train_df['Survived']

    # 5. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    # 6. 转换为 PyTorch Tensor 并强制转为 float32
    X_train_tensor = torch.tensor(X_train.astype(np.float32).values)
    y_train_tensor = torch.tensor(y_train.astype(np.float32).values).view(-1, 1)
    X_test_tensor = torch.tensor(X_test.astype(np.float32).values)
    y_test_tensor = torch.tensor(y_test.astype(np.float32).values).view(-1, 1)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor