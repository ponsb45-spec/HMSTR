# HMSTR

面向网格化风场的多步时空预测代码。
HMSTR 使用 TensorFlow 实现，当前流程包括气象数据预处理、变分模态分解（Variational Mode Decomposition, VMD）、多分支时空特征建模，以及未来U/V 风速分量的多步预测与评估。



## 仓库结构

```text
HMSTR/
├── analysis/
│   └── nature_paper_plot_demo.ipynb  # 结果分析与论文绘图
├── data/
│   └── code/
│       ├── data_process.py           # 数据合并及 VMD 数据构建
│       ├── data_read.py              # CSV 数据读取
│       ├── data_small.py             # 裁剪与降采样
│       ├── ele.py                    # 地形高程处理
│       └── VMD.py                    # VMD 分解
├── evaluation/
│   ├── evaluate.py                   # 网格预测评估
│   ├── loss.py                       # 组合损失函数
│   ├── utils.py                      # 纬度加权指标
│   ├── uv_evaluate.py                # U/V 风场评估
│   └── wdfa.py                       # 风向预测准确率
├── experiments/                      # 实验配置与脚本
├── figure/                           # 论文图表
├── models/
│   └── VST_convlstm.py               # 数据加载、模型、训练和保存逻辑
├── outputs/                          # 实验输出
├── data_init.py                      # 数据预处理入口
└── main.py                           # 训练与测试入口
```

## 环境配置

### 1. 使用 Conda 创建环境

```bash
conda create --name hmstr python=3.10 pip -y
conda activate hmstr
```

### 2. 安装依赖

```bash
pip install -r requrements.txt
```


## 数据准备

模型所需原始数据为grib格式，需要处理成模型所需数据张量；为此本项目提供了完善的数据处理与分解工具，在.env文件内配置好各路径后(建议保持默认值，仅修改原始数据名称即可)，运行根目录下的data_init.py文件即可一件完成模型训练所需所有数据处理(含分解数据生成)

数据处理相关逻辑与数据文件存放结构：

```
data
｜
｜- code // 数据处理相关代码存放文件夹
    ｜- data_preview.ipynb // Jupyter notebook测试脚本，用于开发过程中零散代码调试与数据预览检查
    ｜- data_process.py // 原始数据读取、输出处理、分解、合并核心模块
    ｜- data_read.py // 初始化后的数据读取工具（含json转换）
    ｜- data_small.py // 数据集图缩放模块
    ｜- VMD.py // 分解模块封装
｜- source // 原始grib数据存放
｜- processed // 处理后的风速数据存放处
｜- processed_small // 处理后的风速数据存放处（缩小后）
｜- processed_temp // 处理后的温度、位势数据存放处
｜- processed_temp_small // 处理后的温度、位势数据存放处（缩小后）
｜- processed_combined_small // 存放多维度合并后的数据（缩小后）
｜- vmd_data_small // 存放风速分解并合并多维度的数据（缩小后）
｜- data.py
```
### 训练入口要求的文件

`main.py` 当前直接读取：


## 数据预处理

在补齐 `data/data.py`、配置 `.env` 并准备原始数据后运行：

```bash
python data_init.py
```

预处理流程依次执行：

1. 从 GRIB 文件读取风场及其他气象变量；
2. 将数据裁剪或降采样至目标区域；
3. 合并不同气象变量；
4. 对时序数据执行 VMD 分解；
5. 保存训练需要的 CSV 和 Pickle 文件。

GRIB 文件通过 `xarray` 的 `cfgrib` 引擎读取，因此系统中还需要可用的 ecCodes。

## 训练

确认数据目录正确后，从仓库根目录运行：

```bash
python main.py
```


