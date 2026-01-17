from src.data_loader import HousingDataLoader
from src.model import BJHousingModel
from sklearn.model_selection import train_test_split
import pandas as pd

def main():
    # 数据
    loader = HousingDataLoader("./data/bj_resales_tianchi.csv")
    df = loader.load_clear()

    # X y
    X = df.drop(columns=['price/w'])
    y = df['price/w']

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练模型
    predictor = BJHousingModel()
    predictor.train(X_train, y_train)

    # 测试
    metrics = predictor.evaluate(X_test, y_test)
    print(f"模型 R2 分数：{metrics['R2']:.4f}\n模型 MSE：{metrics['MSE']:.4f}")

    # 重要性分析
    print(predictor.get_feature_importance())


    # 房价预估
    # 85平，两室一厅，10楼，有电梯，精装，2005
    new_house = pd.DataFrame([[85, 2, 1, 10, 1, 3, 2005]], 
                            columns=['area/m^2', 'bedrooms', 'living_rooms', 'floor', 'has_elevator', 'decoration_score', 'year'])
    predicted = predictor.model.predict(new_house)
    print(f"\n房产信息 模拟预测：85平/两室一厅/10楼带电梯/精装/2005年\n估值约为: {predicted[0]:.2f} 万元")


if __name__ == "__main__":
    main()