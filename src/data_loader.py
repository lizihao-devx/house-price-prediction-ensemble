import pandas as pd
import numpy as np
import re
class HousingDataLoader:
    def __init__(self, filepath):
        self.filepath = filepath

    def load_clear(self):
        # 加载数据
        df = pd.read_csv(self.filepath)

        # 过滤数据
        df['面积(m²)'] = pd.to_numeric(df['面积(m²)'], errors='coerce') # errors='coerce' 会把无法转换的错位数据变成 NaN
        df['价格(万元)'] = pd.to_numeric(df['价格(万元)'], errors='coerce')
        df = df.dropna(subset=['面积(m²)', '价格(万元)'])

        # 提取特征，正则解析户型