import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

class BJHousingModel:
    def __init__(self, model_path="models/beijing_v1.pkl"):
        self.model_path = model_path
        # 初始化随机森林，开启全核加速
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.feature_names = None

    def train(self, X, y):
        self.feature_names = X.columns.tolist()
        self.model.fit(X, y)

        # 模型持久化
        joblib.dump({"model": self.model, "features": self.feature_names}, self.model_path)
    
    def evaluate(self, X_test, y_test):
        # 评估性能
        y_pred = self.model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        return {"R2": r2, "MSE": mse}
    
    def get_feature_importance(self):
        # 分析特征重要性
        importance = self.model.feature_importances_
        return pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)