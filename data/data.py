import os
from dotenv import load_dotenv
from data.code.data_process import DataInit
from data.code.data_read import DataRead

load_dotenv()

class Data:
    def __init__(self):
        self.UVGribPath = os.getenv('UVGribPath')
        self.UVCsvProcessedPath = os.getenv('UVCsvProcessedPath')
        self.UVCsvTestPath = os.getenv('UVCsvTestPath')
        self.TempCsvTestPath = os.getenv('TempCsvTestPath')
        self.UVData = None

    def process_data(self):
        # 初始化：转换GRIB文件为CSV文件
        DataInit(self.UVGribPath, self.UVCsvProcessedPath,self.UVCsvTestPath,self.TempCsvTestPath).run()
    
    def read_data(self):
        print("读取数据")
        self.UVData = DataRead(UVPath=self.UVCsvProcessedPath).read_uv_data()
    
    def read_test_data(self):
        print("读取测试数据")
        self.UVData = DataRead(TestPath=self.UVCsvTestPath).read_uv_test_data()
    
    def run(self, read_test_data=True, small=False):
        if small:
            self.UVCsvProcessedPath = os.getenv('UVCsvProcessedPathSmall')
            self.UVCsvTestPath = os.getenv('UVCsvTestPathSmall')
        # 判断是否存在CSV文件
        if not os.path.exists(self.UVCsvProcessedPath):
            self.process_data()
        if read_test_data:
            self.read_test_data()
        else:
            self.read_data()
            
class TempData:
    def __init__(self):
        self.TempGribPath1 = os.getenv('TempGribPath1')
        self.TempGribPath2 = os.getenv('TempGribPath2')
        self.TempCsvProcessedPath = os.getenv('TempCsvProcessedPath')
        self.TempCsvTestPath = os.getenv('TempCsvTestPath')
        self.UVCsvTestPath = os.getenv('UVCsvTestPath')
        self.TempCsvTestPath = os.getenv('TempCsvTestPath')
        self.TempData1 = None

    def process_data(self):
        # 初始化：转换GRIB文件为CSV文件 注意：grib2文件为可选参数，温度数据由于数据源限制必须进行分段下载
        DataInit(GribPath1=self.TempGribPath1, GribPath2=self.TempGribPath2, OutputPath=self.TempCsvProcessedPath, uv_csv_test_path=self.UVCsvTestPath,temp_csv_test_path=self.TempCsvTestPath, type='temp').run()
    
    def read_data(self):
        print("读取温度数据")
        self.TempData = DataRead(TempPath=self.TempCsvProcessedPath, type='temp').read_temp_data()
    
    def read_test_data(self):
        print("读取温度测试数据")
        self.TempData = DataRead(TempTestPath=self.TempCsvTestPath, type='temp').read_temp_test_data()
    
    def run(self, read_test_data=True, small=False):
        if small:
            self.TempCsvProcessedPath = os.getenv('TempCsvProcessedPathSmall')
            self.TempCsvTestPath = os.getenv('TempCsvTestPathSmall')
        # 判断是否存在CSV文件
        if not os.path.exists(self.TempCsvProcessedPath):
            self.process_data()
        if read_test_data:
            self.read_test_data()
        else:
            self.read_data()
            
class Combine:
    def __init__(self):
        self.combineDataPathSmall = os.getenv('CombinedDataPathSmall')
        self.combineDataTestPathSmall = os.getenv('CombinedDataTestPathSmall')
        self.combineData = None
    
    def read_data(self, test=False, custom_path=None):
        print(f"读取组合数据 test={test} 数据路径: {self.combineDataPathSmall if not test else self.combineDataPathSmall}")
        if custom_path:
            self.combineData = DataRead(combineDataPath=custom_path).read_combine_data()
        elif test:
            self.combineData = DataRead(combineDataPath=self.combineDataTestPathSmall).read_combine_data()
        else:
            self.combineData = DataRead(combineDataPath=self.combineDataPathSmall).read_combine_data()