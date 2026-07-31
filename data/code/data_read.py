import pandas as pd
import numpy as np
import json
import ast

class DataRead:
    def __init__(self, UVPath=None, TestPath=None, combineDataPath=None):
        self.UVPath = UVPath
        self.TestPath = TestPath
        self.combineDataPath = combineDataPath


    def read_uv_data(self, range=None):
        UVWindData  = pd.read_csv(self.UVPath)
        if range is not None:
            UVWindData = UVWindData[range[0]:range[1]]
        # 将每一行的u100和v100数据转换为numpy数组
        UVWindData['u100'] = UVWindData['u100'].apply(lambda x: np.array(json.loads(x)))
        UVWindData['v100'] = UVWindData['v100'].apply(lambda x: np.array(json.loads(x)))
        
        return UVWindData
    
    def read_uv_test_data(self):
        UVWindData = pd.read_csv(self.TestPath)
        # 将每一行的u100和v100数据转换为numpy数组
        UVWindData['u100'] = UVWindData['u100'].apply(lambda x: np.array(json.loads(x)))
        UVWindData['v100'] = UVWindData['v100'].apply(lambda x: np.array(json.loads(x)))
        
        return UVWindData
    
    def read_combine_data(self):
        combineData = pd.read_csv(self.combineDataPath)
        # 将每一行的数据转换为numpy数组
        cols = ['combined', 'u100','v100', 't', 'z']
        combineData[cols] = combineData[cols].applymap(lambda x: np.array(ast.literal_eval(x)))
        
        return combineData

