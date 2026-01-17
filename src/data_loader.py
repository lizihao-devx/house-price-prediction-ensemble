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
        df['area/m^2'] = pd.to_numeric(df['面积(㎡)'], errors='coerce')
        df['price/w'] = pd.to_numeric(df['价格(万元)'], errors='coerce')
        df['year'] = pd.to_numeric(df['年份'], errors='coerce')
        df['floor'] = pd.to_numeric(df['楼层'], errors='coerce')

        df = df.dropna(subset=['area/m^2', 'price/w'])

        # 提取特征，正则解析户型
        def parse_layout(layout_str):
            if not isinstance(layout_str, str): return 0, 0
            bedrooms = re.findall(r'(\d+)室', layout_str)
            living_rooms = re.findall(r'(\d+)厅', layout_str)
            r = int(bedrooms[0]) if bedrooms else 0
            h = int(living_rooms[0]) if living_rooms else 0
            return r, h
        
        # 将返回的元组拆成两列
        df['bedrooms'], df['living_rooms'] = zip(*df['户型'].apply(parse_layout))

        # 处理电梯列: 楼层>6且缺失值的补1，其余补0
        def fix_elevator(row):
            if pd.isna(row['电梯']):
                return 1 if row['floor'] > 6 else 0
            return 1 if row['电梯'] == '有电梯' else 0
        
        df['has_elevator'] = df.apply(fix_elevator, axis=1)

        # 量化装修情况：精装3 简装2 其他1 毛坯0
        decoration_map = {'精装': 3, '简装': 2, '其他': 1, '毛坯': 0}
        df['decoration_score'] = df['装修情况'].map(decoration_map).fillna(1)

        # 最终特征保留：楼层 面积 年份 室 厅 电梯 装修
        # 目标值：价格
        features = ['area/m^2', 'bedrooms', 'living_rooms', 'floor', 'has_elevator', 'decoration_score', 'year']
        target = 'price/w'

        df['year'] = df['year'].fillna(df['year'].median())
        df['floor'] = df['floor'].fillna(df['floor'].median())

        final_df = df[features + [target]].copy()

        return final_df
    
if __name__ == "__main__":
    loader = HousingDataLoader("./data/bj_resales_tianchi.csv")
    clean_data = loader.load_clear()
    print("清洗后的数据预览：")
    print(clean_data.head())
    print(clean_data.info())