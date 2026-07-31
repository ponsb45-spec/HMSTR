"""
该脚本用于评估网格预测结果的准确性
1. 读取预测结果和真实结果
2. 使用纬度加权评估函数，对网格预测结果进行误差分析
3. 计算每个时间步的RMSE
4. 计算整体的RMSE
5. 计算每个时间步的MAE
6. 计算整体的MAE
7. 计算每个时间步的ACC
8. 计算整体的ACC
"""
import numpy as np
from evaluation.utils import lat_weighted_rmse, weighted_mae, weighted_acc
import pickle

class GridEvaluate:
    """
    网格预测结果评估类
    """
    def __init__(self,save_path,lat_dim, lon_dim, start_lat, end_lat):
        self.save_path = save_path
        self.lat_dim = lat_dim
        self.lon_dim = lon_dim
        self.lat = np.linspace(start_lat, end_lat, lat_dim)
        self.result = None

    def compute_wind_speed(self, u, v):
        return np.sqrt(u ** 2 + v ** 2)

    def read_result(self):
        """
        读取预测结果和真实结果
        """
        result=np.load(self.save_path)

        pred = result['pred']
        true = result['true']
        y_pred_u = pred[:, :, 0, :, :]
        y_pred_v = pred[:, :, 1, :, :]

        y_true_u = true[:, :, 0, :, :]
        y_true_v = true[:, :, 1, :, :]
        y_pred = self.compute_wind_speed(y_pred_u, y_pred_v)
        y_true = self.compute_wind_speed(y_true_u, y_true_v)

        return y_pred,y_true


    def evaluate(self):
        """
        评估网格预测结果的准确性
        """
        pred,true = self.read_result()
        print(f"数据形状 - pred: {pred.shape}, true: {true.shape}")
        
        # 如果数据是4维，添加最后一维
        if pred.ndim == 4:
            pred = np.expand_dims(pred, axis=-1)
            true = np.expand_dims(true, axis=-1)
            print(f"扩展后形状 - pred: {pred.shape}, true: {true.shape}")
        
        rmse = lat_weighted_rmse(pred, true, self.lat)
        mae = weighted_mae(pred, true, self.lat)
        acc = weighted_acc(pred, true, self.lat)
        self.result = {
            "rmse": rmse.tolist(),
            "mae": mae.tolist(),
            "acc": acc.tolist()
        }
        return rmse, mae, acc

if __name__ == "__main__":

    result_path = "HMSTR/outputs/test.npz"


    lat_dim = 11
    lon_dim = 11
    start_lat = 44.75
    end_lat = 47.5
    evaluation = GridEvaluate(result_path, lat_dim, lon_dim, start_lat, end_lat)
    rmse, mae, acc = evaluation.evaluate()
    global_rmse = rmse.mean()
    global_mae = mae.mean()
    global_acc = acc.mean()

    print("=" * 50)
    print("全局评估指标:")
    print("=" * 50)
    print(f"Global RMSE: {global_rmse:.4f}")
    print(f"Global MAE:  {global_mae:.4f}")
    print(f"Global ACC:  {global_acc:.4f}")

    print("\n" + "=" * 50)
    print("各预测时次的详细指标:")
    print("=" * 50)
    print("\nRMSE per step:")
    print(", ".join(f"{value:.4f}" for value in rmse))
    print("\nMAE per step:")
    print(", ".join(f"{value:.4f}" for value in mae))
    print("\nACC per step:")
    print(", ".join(f"{value:.4f}" for value in acc))
    result = {
        "global_rmse": float(global_rmse),
        "global_mae": float(global_mae),
        "global_acc": float(global_acc),
        "rmse": rmse.tolist(),
        "mae": mae.tolist(),
        "acc": acc.tolist(),
    }
    print(result)



