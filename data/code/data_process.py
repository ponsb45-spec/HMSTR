import numpy as np
import pandas as pd
import xarray as xr
import time
import json
import ast
from data.code.data_read import DataRead
from data.code.VMD import VMD_Decompose
import os
from dotenv import load_dotenv


load_dotenv()

class DataInit:
    def __init__(self, GribPath1, OutputPath, uv_csv_test_path,temp_csv_test_path, type='wind', GribPath2=None,):
        self.UVGribPath = GribPath1
        self.UVGribPath2 = GribPath2
        self.OutputPath = OutputPath
        self.type = type
        self.uv_csv_test_path = uv_csv_test_path
        self.temp_csv_test_path = temp_csv_test_path

    def read_grib(self, path):
        # 读取 GRIB 文件
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}-读取 GRIB 文件:-{path}")
        ds = xr.open_dataset(path, engine="cfgrib")
        print("纬度范围:", ds.latitude.values.min(), "~", ds.latitude.values.max())
        print("经度范围:", ds.longitude.values.min(), "~", ds.longitude.values.max())
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}-读取 GRIB 文件成功")
        return ds

    def process_wind_data(self, ds):
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}-处理数据")
        # 处理数据
        u = ds['u100'].values
        v = ds['v100'].values
        # 获取坐标轴
        times = ds['time'].values  # shape: (52584,)

        # 展开为二维表格
        records = []
        for t_idx, t in enumerate(times):
            # 每10000个时间步打印一次
            if t_idx % 10000 == 0:
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}-处理数据: {t_idx}/{len(times)}")
            u_t = json.dumps(u[t_idx].tolist())
            v_t = json.dumps(v[t_idx].tolist())
            # 将u_t和v_t直接以列表字符串的形式添加到records中
            records.append([t, u_t, v_t])

        df = pd.DataFrame(records)
        df.columns = ['time', 'u100', 'v100']
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}-处理数据成功")
        return df



    def process_temp_data(self, ds):
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}-处理温度、位势数据")
        # 处理数据
        temp = ds['t'].values
        geopotential = ds['z'].values
        # 获取坐标轴
        times = ds['time'].values

        # 展开为二维表格
        records_temp = []
        records_geo = []
        for t_idx, t in enumerate(times):
            # 每10000个时间步打印一次
            if t_idx % 10000 == 0:
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}-处理温度、位势数据: {t_idx}/{len(times)}")
            temp_t = json.dumps(temp[t_idx].tolist())
            geo_t = json.dumps(geopotential[t_idx].tolist())
            # 将temp_t和geo_t直接以列表字符串的形式添加到records中
            records_temp.append([t, temp_t])
            records_geo.append([t, geo_t])

        df_temp = pd.DataFrame(records_temp)
        df_geo = pd.DataFrame(records_geo)
        df_temp.columns = ['time', 't']
        df_geo.columns = ['time', 'z']
        df_temp['z'] = df_geo['z']
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}-处理温度、位势数据成功")
        return df_temp

    def save_data(self, df):
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}-保存数据中")
        df.to_csv(self.OutputPath, index=False)
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}-保存数据成功")
        
    def generate_test_data(self, df, OutputPath):
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}-生成测试数据")
        # 生成测试数据
        test_data = df.iloc[:1000]
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}-生成测试数据成功")
        test_data.to_csv(OutputPath, index=False)
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}-保存测试数据成功")
    
    def run(self):
        ds = self.read_grib(self.UVGribPath)

        if self.UVGribPath2 is not None:
            ds2 = self.read_grib(self.UVGribPath2)
        if self.type == 'wind':
            df = self.process_wind_data(ds)
            self.generate_test_data(df, OutputPath=self.uv_csv_test_path)
        elif self.type == 'temp':
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}-正在处理grib1文件")
            df1 = self.process_temp_data(ds)
            self.generate_test_data(df1, OutputPath=self.temp_csv_test_path)
            if self.UVGribPath2 is not None:
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}-正在处理grib2文件")
                df2 = self.process_temp_data(ds2)
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}-正在合并数据")
                df = pd.concat([df1, df2], axis=0, ignore_index=True)
            else:
                df = df1
        else:
            raise ValueError("Unsupported data type")
        self.save_data(df)

class DataCombine:
    '''
    读取wind_U, wind_V, temp, geo数据，将其合并成一个三维图json数据进行存储，用于后续输入模型训练
    '''
    def __init__(self, UVPath, TempGeoPath, OutputPath, test:bool=False, UVPathTest=None, TempGeoPathTest=None, OutputPathTest=None,ElePath=None):
        self.UVPath = UVPath if not test else UVPathTest
        self.TempGeoPath = TempGeoPath if not test else TempGeoPathTest
        self.OutputPath = OutputPath if not test else OutputPathTest
        self.ElePath = ElePath


    def read_data(self):
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}-读取wind_U_V, temp_geo数据-{self.UVPath}, {self.TempGeoPath}")
        # 读取wind_U, wind_V, temp, geo数据
        wind_U_V = pd.read_csv(self.UVPath)
        temp_geo = pd.read_csv(self.TempGeoPath)
        return wind_U_V, temp_geo

    
    def calc_wind_speed(self, df):
    #     '''
    #     计算风速，其中u100和v100均为一个json数组字符串，且均为11x11的数组
    #     '''
    #     print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}-计算风速")
    #     # 将字符串转为数组
        df['u100'] = df['u100'].apply(lambda x: np.array(ast.literal_eval(x)))
        df['v100'] = df['v100'].apply(lambda x: np.array(ast.literal_eval(x)))
    #
    #     df['wind_speed'] = df.apply(lambda row: np.sqrt(row['u100']**2 + row['v100']**2), axis=1)
        return df['u100'],df['v100']

    def prepare_training_data(self, combined_data, temp_geo):
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}-合并四通道训练数据")
        # 将字符串转为数组

        temp_geo['t'] = temp_geo['t'].apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else x)
        temp_geo['z'] = temp_geo['z'].apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else x)

        # 按时间对齐
        merged = pd.merge(temp_geo,combined_data[['time','u100','v100']], left_on='time', right_on='time')

        # 合并为 (11, 11, 4)
        merged['combined'] = merged.apply(
            lambda r: np.stack([r['t'], r['z'], r['u100'],r['v100']], axis=-1),
            axis=1
        )
        return merged
    
    def save_training_data(self, combined_data):
        # 保存训练数据
        combined_data.to_csv(self.OutputPath, index=False)
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}-保存训练数据成功")
    
    def run(self):
        '''
        将temp与geo数据合并到合并速度中，构建一个三通道图数据，用于后续输入模型训练
        '''
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}-开始合并数据")
        # 构建一个新dataframe，用于存储合并后的数据
        combined_data = pd.DataFrame()
        # 读取wind_U_V, temp_geo数据
        wind_U_V, temp_geo = self.read_data()
        # 计算风速
        combined_data['time'] = temp_geo['time']
        combined_data['u100'],combined_data['v100'] = self.calc_wind_speed(wind_U_V)
        # 组合三通道数据
        combined_data = self.prepare_training_data(combined_data, temp_geo)
        # 将t，z，wind_speed转换为json字符串存储
        cols = ['combined', 'u100', 'v100', 't', 'z']
        combined_data[cols] = combined_data[cols].applymap(lambda x: json.dumps(x.tolist()))
        
        self.save_training_data(combined_data)
        
class DataVmd:
    '''
    生成分解后的数据集(合并三通道数据)
    '''
    def __init__(self, test:bool=False, combineDataPath=None, OutputPath=None,ElePath=None):
        self.test = test
        self.combineDataPath = combineDataPath if combineDataPath is not None else (os.getenv('CombinedUVDataTestPathSmall') if test else os.getenv('CombinedUVDataPathSmall'))
        self.outputPath = OutputPath if OutputPath is not None else (os.getenv('VmdDataUVTestPathSmall') if test else os.getenv('VmdDataUVPathSmall'))

    def combine_three_channels(self, u_vmd, v_vmd, temp_geo):
        # 确保每个t, z是np.array类型
        temp_geo['t'] = temp_geo['t'].apply(lambda x: np.array(x))
        temp_geo['z'] = temp_geo['z'].apply(lambda x: np.array(x))
        
        combined_list = []

        for i in range(len(temp_geo)):
            t = temp_geo.iloc[i]['t']        # (11, 11)
            z = temp_geo.iloc[i]['z']        # (11, 11)
            u = u_vmd[i]                     # (4, 11, 11)
            v = v_vmd[i]

            # 扩展 t、z 到 wind_speed 相同形状
            t_expanded = np.repeat(t[np.newaxis, :, :], u.shape[0], axis=0)  # (4,11,11)
            z_expanded = np.repeat(z[np.newaxis, :, :], u.shape[0], axis=0)  # (4,11,11)


            # 拼接为 (4,11,11,4)
            combined = np.stack([t_expanded, z_expanded, u, v], axis=-1)
            combined_list.append(combined)

        # 转换为最终形状 (1000,4,11,11,4)
        return np.stack(combined_list, axis=0)

    def vmd_decompose(self):
        data = DataRead(combineDataPath=self.combineDataPath)
        combinedData = data.read_combine_data()
        wind_u100 = np.stack(combinedData['u100'].values)
        wind_v100 = np.stack(combinedData['v100'].values)
        # 执行分解过程
        u_vmd, u_imfs_hat, u_omega = VMD_Decompose(wind_u100).vmd_parallel()
        v_vmd, u_imfs_hat, u_omega = VMD_Decompose(wind_v100).vmd_parallel()

        # 合并数据
        temp_geo = combinedData[['time', 't', 'z']]
        wind_vmd = self.combine_three_channels(u_vmd,v_vmd,temp_geo)

        return wind_vmd

    def save_vmd_data(self, wind_vmd):
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}-保存VMD分解数据中")
        df = pd.DataFrame({
            'combined': list(wind_vmd)  # 每行是 (4, 11, 11, 4) 的 ndarray
        })
        df.to_pickle(self.outputPath)
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}-保存VMD分解数据成功")
    
    def read_vmd_data(self, custom_path=None):
        if custom_path:
            self.outputPath = custom_path
        else:
            self.outputPath = self.outputPath
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}-读取VMD分解数据中")
        df = pd.read_pickle(self.outputPath)
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}-读取VMD分解数据成功")
        return df
    
    def run(self):
        wind_vmd = self.vmd_decompose()
        self.save_vmd_data(wind_vmd)
        
