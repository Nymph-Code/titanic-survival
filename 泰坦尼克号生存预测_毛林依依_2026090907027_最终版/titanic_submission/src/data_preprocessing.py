# 数据读取、清洗、特征工程

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 随机种子，固定住数据划分的结果
SEED = 42

# 最后真正喂给模型的 10 个特征。顺序写死在这儿，
# 训练和预测都从它取，列就不会对不上
FEATURE_COLUMNS = [
    'Pclass', 'Sex', 'Age', 'SibSp', 'Parch',
    'Fare', 'Embarked_Q', 'Embarked_S', 'Title', 'FamilySize',
]

# 称谓转数字，少见的都归到 Rare
TITLE_MAPPING = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}

RARE_TITLES = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
               'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']


def extract_title(name_series):
    # 从姓名里把称谓抠出来，'Braund, Mr. Owen Harris' → 'Mr'
    # 正则的意思是：一个空格 + 一串英文字母 + 一个点
    return name_series.str.extract(r' ([A-Za-z]+)\.', expand=False)


def fit_preprocessor(train_df):
    # 只在训练集上算预处理要用的参数，算完返回一个字典。
    # 传进来的只能是训练集，不能是整个数据集
    params = {}

    # 缺失值填什么：年龄填中位数（不容易被极端值带偏），登船港口填众数
    params['age_median'] = float(train_df['Age'].median())
    params['embarked_mode'] = str(train_df['Embarked'].mode()[0])

    # 标准化要用的均值和标准差
    params['age_mean'] = float(train_df['Age'].mean())
    params['age_std'] = float(train_df['Age'].std())
    params['fare_mean'] = float(train_df['Fare'].mean())
    params['fare_std'] = float(train_df['Fare'].std())

    params['title_mapping'] = TITLE_MAPPING
    params['feature_columns'] = FEATURE_COLUMNS

    return params


def transform(df, params):
    # 拿着 fit_preprocessor 算好的参数处理数据，返回 10 列特征的数组。
    # 训练集、测试集、新样本全都走这个函数，保证处理方式一模一样
    df = df.copy()

    # 1. 缺失值
    df['Age'] = df['Age'].fillna(params['age_median'])
    df['Embarked'] = df['Embarked'].fillna(params['embarked_mode'])
    # 客舱号缺了 77%，剩下那点也拼不出规律，这列直接不要了
    if 'Cabin' in df.columns:
        df = df.drop(columns=['Cabin'])

    # 2. 性别转成数字，模型只认数字
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # 3. 登船港口独热编码
    # C/Q/S 三个港口之间没有大小关系，不能直接编成 0/1/2。
    # 这里以 C 为基准，只留 Q 和 S 两列，三列不会加起来恒等于 1（共线性）。
    #
    # 不用 pd.get_dummies(drop_first=True)：它是按"这一批数据里出现过哪些值"
    # 决定生成哪几列的。只喂一行、且这一行是 Q 的时候，Q 就是唯一观测到的类别，
    # 被 drop_first 丢掉，Q 和 S 两列都不生成，补 0 之后整个样本变成 C。
    # 直接按值比较，结果和这一批数据里有几个港口无关。
    df['Embarked_Q'] = (df['Embarked'] == 'Q').astype(int)
    df['Embarked_S'] = (df['Embarked'] == 'S').astype(int)
    df = df.drop(columns=['Embarked'])

    # 4. 从姓名里提称谓
    # 姓名本身没什么用，但称谓能间接看出性别、年龄和身份，是个挺好的特征
    df['Title'] = extract_title(df['Name'])
    df['Title'] = df['Title'].replace(RARE_TITLES, 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    df['Title'] = df['Title'].map(params['title_mapping'])
    # 碰到没见过的称谓，map 完会变成空，统一填 0
    df['Title'] = df['Title'].fillna(0).astype(int)

    # 5. 家庭人数 = 兄弟姐妹配偶数 + 父母子女数 + 自己
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # 6. 标准化
    # 年龄是 0~80，票价是 0~512，量级差太多，直接喂进网络票价会独占风头。
    # 用训练集算出来的均值和标准差，测试集和新样本套同一个
    df['Age'] = (df['Age'] - params['age_mean']) / params['age_std']
    df['Fare'] = (df['Fare'] - params['fare_mean']) / params['fare_std']

    # 7. 只留这 10 列，顺序按 FEATURE_COLUMNS 来
    df = df[params['feature_columns']]

    return df.astype(np.float32).values


def load_and_split(csv_path, test_size=0.2):
    # 读数据 → 先划分 → 只在训练集上算参数 → 两边套同一套参数
    # 顺序不能反。先算参数再划分的话，填年龄用的中位数里就包含了测试集的年龄，等于泄露了
    df = pd.read_csv(csv_path)

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=SEED)

    params = fit_preprocessor(train_df)

    X_train = transform(train_df, params)
    y_train = train_df['Survived'].values
    X_test = transform(test_df, params)
    y_test = test_df['Survived'].values

    return X_train, y_train, X_test, y_test, params


def save_params(params, path):
    # 把预处理参数存成 json，预测的时候再读回来
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False, indent=2)


def load_params(path):
    # 读回预处理参数
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
