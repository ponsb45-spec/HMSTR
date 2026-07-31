"""
该脚本用于评估网格预测结果的准确性
ERA5 的官方误差评估方式
1. 读取预测结果和真实结果
2. 使用纬度加权评估函数，对网格预测结果进行误差分析
3. 计算每个时间步的RMSE
4. 计算整体的RMSE
5. 计算每个时间步的MAE
6. 计算整体的MAE
"""


import numpy as np
import tensorflow as tf

def lat_weighted_rmse(pred, true, lat):
    """
    pred, true shape: (n,24,lat,lon,1)
    lat shape: (lat,)
    return: RMSE per forecast step, shape = (24,)
    """
    # 经纬度差
    err = pred - true
    H,W = pred.shape[2],pred.shape[3]

    # 误差平方
    err2 = err ** 2

    # 纬度权重
    w = np.cos(np.deg2rad(lat))
    w = w / w.mean()     # shape (11,)

    # 广播到 (n,24,11,11,1)
    w = w.reshape(1,1,-1,1,1)

    # 加权
    weighted_err2 = err2 * w

    # 对 lat, lon 求平均 → 空间均值
    spatial_mean = weighted_err2.mean(axis=(2,3))  # shape (n, 24, 1)

    # 对 batch n 求平均
    batch_mean = spatial_mean.mean(axis=0)          # shape (24, 1)

    # 开平方得到 RMSE
    rmse = np.sqrt(batch_mean).squeeze()            # shape (24,)

    return rmse


def weighted_mae(pred, true, lat):
    """
    pred, true shape: (lat, lon, n, 24, 1)
    lat shape: (lat,)
    return: MAE per forecast step, shape = (24,)
    """
    # 计算绝对误差
    err = np.abs(pred - true)       # shape (10,10,n,24,1)


    # 纬度权重
    w = np.cos(np.deg2rad(lat))
    w /= w.mean()     # shape (10,)

    # 广播到 (10,10,n,24,1)
    w = w.reshape(1, 1, -1, 1, 1)

    # 加权
    weighted_err = err * w

    # 对 lat, lon 求平均 → 空间均值
    spatial_mean = weighted_err.mean(axis=(2, 3))  # shape (n, 24, 1)

    # 对 batch n 求平均
    batch_mean = spatial_mean.mean(axis=0)          # shape (24, 1)

    # squeeze 得到 MAE
    mae = batch_mean.squeeze()            # shape (24,)

    return mae


def weighted_acc(pred, true, lat):
    """
    计算加权异常相关系数 (Anomaly Correlation Coefficient)
    pred, true shape: (lat, lon, n, 24, 1)
    lat shape: (lat,)
    return: ACC per forecast step, shape = (24,)
    """
    # 纬度权重
    w = np.cos(np.deg2rad(lat))
    w /= w.mean()     # shape (11,)
    # 广播到 (lat, lon) 用于后续计算
    w = w.reshape(-1,1)
    
    # 初始化 ACC 数组
    ACC = np.empty(24)
    
    # 对每个预测时间步计算 ACC
    for i in range(24):
        # 提取第 i 个时间步，shape: (lat, lon, n, 1) -> (lat, lon, n)
        pred_i = pred[:, i, :, :, 0]
        true_i = true[:, i, :, :, 0]

        
        # 计算气候态（对批次求平均）
        clim = true_i.mean(axis=0)  # shape (lat, lon)
        
        # 计算真实值的异常
        a = true_i - clim  # shape (n, lat, lon)
        a_prime = a - a.mean()  # 减去全局均值，shape (n, lat, lon)
        a_prime = tf.transpose(a_prime, (0,2,1))
        
        # 计算预测值的异常
        fa = pred_i - clim  # shape (n, lat, lon)
        fa_prime = fa - fa.mean()  # 减去全局均值，shape (n, lat, lon)
        fa_prime = tf.transpose(fa_prime, (0, 2, 1))
        
        # 应用纬度权重：w shape (lat, 1)，需要广播到 (n, lat, lon)
        # 将 w 转置为适合广播的形状
        w_broadcast = w.T.reshape(1, w.shape[0], 1)  # shape (1, lat, 1)
        
        # 计算加权相关系数
        numerator = np.sum(w * fa_prime * a_prime)
        denominator = np.sqrt(
            np.sum(w * fa_prime ** 2) * np.sum(w * a_prime ** 2)
        )
        
        ACC[i] = numerator / denominator
    
    return ACC
