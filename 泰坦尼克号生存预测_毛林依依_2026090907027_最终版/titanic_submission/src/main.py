# 整个流程的入口，在项目根目录下敲 python src/main.py
#
# 训练那部分在 train.py，预测那部分在 predict.py，这里只是把它们串起来

import sys

from predict import run_demo
from train import main as train_main


def main():
    # 1 + 2. 训练并在测试集上评估
    train_main()

    # 3. 用训好的模型预测一名示例乘客
    print('\n' + '=' * 50)
    print('用训练好的模型做新样本预测')
    print('=' * 50)
    run_demo()

    print('\n' + '=' * 50)
    print('全部完成')
    print('=' * 50)


if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as e:
        print(f'找不到文件: {e}')
        print('请确认 data/train.csv 存在。')
        sys.exit(1)
