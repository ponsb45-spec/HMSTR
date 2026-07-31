import os
from data.data import Data, TempData
from data.code.data_process import DataCombine, DataVmd
from dotenv import load_dotenv
from data.code.data_small import process_wind_data, process_temp_geopotential_data

load_dotenv()


if __name__ == "__main__":
    # 1. 处理风数据（将grib文件转换为csv文件）
    wind_data = Data()
    wind_data.process_data()
    
    # 2. 处理温度数据（将grib文件转换为csv文件）
    temp_data = TempData()
    temp_data.process_data()
    
    # 3. 风速、温度数据缩小化（将csv文件转换为11x11的csv文件）
    process_wind_data(
        input_filename=os.getenv("UVCsvProcessedPath"),
        output_filename=os.getenv("UVCsvPathSmall"),
        target_shape=(11, 11),
        downsample_factor=1,
        center=None,
    )

    process_temp_geopotential_data(
        input_filename=os.getenv("TempCsvProcessedPath"),
        output_filename=os.getenv("TempCsvPathSmall"),
        target_shape=(11, 11),
        downsample_factor=1,
        center=None,
    )

    # 4. 合并wind和temp_geo数据,构建3通道数据
    DataCombine(
        UVPath=os.getenv("UVCsvPathSmall"),
        TempGeoPath=os.getenv("TempCsvPathSmall"),
        OutputPath=os.getenv("CombinedUVDataPathSmall"),
        test=False,
        UVPathTest=os.getenv("UVCsvTestPathSmall"),
        TempGeoPathTest=os.getenv("TempCsvTestPathSmall"),
        OutputPathTest=os.getenv("CombinedUVDataTestPathSmall"),
    ).run()

    # 5. VMD分解三通道数据，构建分解后数据集
    DataVmd(test=False).run()
