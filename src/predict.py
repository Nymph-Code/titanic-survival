import torch
import pandas as pd
from model import TitanicModel

def predict_new_passenger(passenger_info, model_path='../docs/titanic_model.pth', input_dim=10):
    # 1. 将输入字典转为 DataFrame
    df = pd.DataFrame([passenger_info])

    # 2. 复现特征工程
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # 3. 转为 Tensor
    input_tensor = torch.tensor(df.astype('float32').values)

    # 4. 加载模型并预测
    model = TitanicModel(input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)
        prediction = (output > 0.5).float().item()

    return int(prediction)

if __name__ == '__main__':
    print("="*40)
    print("泰坦尼克号生存预测系统")
    print("="*40)
    
    
    sample = {
        'Pclass': 3, 'Sex': 'male', 'Age': 22, 
        'SibSp': 0, 'Parch': 0, 'Fare': 7.25, 
        'Embarked_Q': 0, 'Embarked_S': 1, 
        'Title': 1, 'FamilySize': 1
    }
    
    result = predict_new_passenger(sample)
    print(f"\n📌 自动测试 (22岁三等舱男性): 预测结果 = {result} (1=幸存, 0=遇难)")

    # 手动输入
    print("\n" + "-"*40)
    print("现在你可以尝试输入你自己的乘客数据来预测：")
    try:
        pclass = int(input("请输入船舱等级 (1/2/3): "))
        sex = input("请输入性别 (male/female): ")
        age = float(input("请输入年龄: "))
        sibsp = int(input("请输入同辈亲属数量 (SibSp): "))
        parch = int(input("请输入长辈/子女数量 (Parch): "))
        
    
        embark_q = int(input("是否从 Q 港口登船? (是=1, 否=0): "))
        embark_s = int(input("是否从 S 港口登船? (是=1, 否=0): "))
        title = int(input("称谓编号 (1=Mr, 2=Miss, 3=Mrs, 4=Master, 5=Rare): "))
        
        # 计算 FamilySize
        family_size = sibsp + parch + 1
        
        user_sample = {
            'Pclass': pclass, 'Sex': sex, 'Age': age,
            'SibSp': sibsp, 'Parch': parch, 'Fare': 7.25, # 票价这里随便填个默认值
            'Embarked_Q': embark_q, 'Embarked_S': embark_s,
            'Title': title, 'FamilySize': family_size
        }
        
        user_result = predict_new_passenger(user_sample)
        print(f"\n📌 你的预测结果: {user_result} (1=幸存, 0=遇难)")
        
    except ValueError:
        print("输入格式错误，请重新运行并输入正确的数字/字母。")