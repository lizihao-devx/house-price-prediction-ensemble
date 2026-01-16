import pandas as pd
import numpy as np
import re
class HousingDataLoader:
    def __init__(self, filepath):
        self.filepath = filepath

    def load_clear(self)
        # 加载数据
        df = pd.read_csv(self.filepath)

        # 过滤数据