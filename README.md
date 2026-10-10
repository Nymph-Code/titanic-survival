# 泰坦尼克号生存预测 (PyTorch)

## 项目简介
基于 Kaggle 泰坦尼克号数据集，使用 PyTorch 自定义全连接神经网络，完成数据清洗、特征工程、模型训练与预测。

## 目录结构
- `src/`: 源代码
  - `
data_preprocessing.py
`: 数据清洗与特征工程
  - `
model.py
`: 自定义神经网络模型结构
  - `
train.py
`: 模型训练与可视化入口
  - `
predict.py
`: 新样本预测
- `docs/`: 学习笔记.pdf 及可视化图表 (training_curves.png)

## 运行方法
1. 安装依赖：`pip install -r requirements.txt`
2. 激活环境：`conda activate pytorch`
3. 进入源码目录：`cd src`
4. 训练模型：`python 
train.py
`
5. 运行预测：`python 
predict.py`