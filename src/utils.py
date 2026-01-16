import pandas as pd
import re

def clean_housing_data(df):
    """
    专门针对天池二手房数据的清洗函数
    """
    # 处理户型：提取室和厅
    df['rooms'] = df['户型'].apply(lambda x: int(re.findall(r'(\d+)室', x)[0]) if '室' in x else 0)
    df['halls'] = df['户型'].apply(lambda x: int(re.findall(r'(\d+)厅', x)[0]) if '厅' in x else 0)
    
    # 处理电梯：映射为 0 和 1
    df['has_elevator'] = df['电梯'].map({'有电梯': 1, '无电梯': 0})
    
    # 3. 处理装修情况：简单映射 (实际可更精细)
    df['decoration'] = df['装修情况'].map({'精装': 2, '简装': 1, '毛坯': 0}).fillna(1)
    
    # 4. 删除不再需要的原始中文列
    cols_to_drop = ['市区', '小区', '户型', '朝向', '装修情况', '电梯']
    df_cleaned = df.drop(columns=cols_to_drop)
    
    # 5. 重命名列名（避免中文路径或编码报错，也是工程好习惯）
    df_cleaned.columns = ['floor', 'area', 'price', 'year', 'rooms', 'halls', 'has_elevator', 'decoration']
    
    return df_cleaned