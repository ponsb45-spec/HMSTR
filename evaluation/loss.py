"""
该脚本用于定义损失函数
"""
import numpy as np
import tensorflow as tf


class WindSpeedErrorLoss:

    def wind_speed_loss(self, y_pred, y_true):
        diff = y_pred - y_true
        loss = tf.reduce_mean(tf.square(diff))
        return loss

    def compute_wind_speed(self, y_pred, y_true):
        """
        分别计算u,v损失
        [batch,time,H,W,2]
        """
        pred_u = y_pred[..., 0]
        pred_v = y_pred[..., 1]
        true_u = y_true[..., 0]
        true_v = y_true[..., 1]

        # 计算损失
        loss_u = self.wind_speed_loss(pred_u, true_u)
        loss_v = self.wind_speed_loss(pred_v, true_v)

        return loss_u, loss_v


class AngErrorLoss:
    def __init__(self,epsilon=1e-8):
        self.epsilon = epsilon

    def calculate_wind_direction(self, y_pred, y_true):
        pred_u = y_pred[..., 0]
        pred_v = y_pred[..., 1]

        true_u = y_true[..., 0]
        true_v = y_true[..., 1]

        # 计算风向
        pred_dir = tf.atan2(pred_v, pred_u)
        true_dir = tf.atan2(true_v, true_u)
        pred_dir_deg = pred_dir * 180.0 / np.pi
        true_dir_deg = true_dir * 180.0 / np.pi

        # 角度误差计算
        wind_dir_diff = tf.abs(pred_dir_deg - true_dir_deg)
        wind_dir_diff = tf.where(wind_dir_diff > 180.0, 360.0 - wind_dir_diff, wind_dir_diff)
        diff_normalized = wind_dir_diff / 180.0

        mse = tf.reduce_mean(diff_normalized ** 2)
        rmse = tf.sqrt(mse+self.epsilon)
        return rmse


class SSIMErrorLoss:

    def ssim_loss_stable(self,y_true, y_pred):
        # y_true/pred shape: [Batch, Time, H, W, C]
        # 展平 batch 和 time 维度，将其视为多通道图像
        b, t, h, w, c = tf.shape(y_true)[0], tf.shape(y_true)[1], tf.shape(y_true)[2], tf.shape(y_true)[3], \
        tf.shape(y_true)[4]

        y_true_flat = tf.reshape(y_true, [b * t, h, w, c])
        y_pred_flat = tf.reshape(y_pred, [b * t, h, w, c])

        # max_val 是标准化后的取值范围，通常 StandardScaler 处理后在 [-3, 3] 左右，取 6.0 比较稳妥
        return 1.0 - tf.reduce_mean(tf.image.ssim(y_true_flat, y_pred_flat, max_val=6.0))


class TotalWindLoss(tf.keras.losses.Loss):
    def __init__(self, name="total_wind_loss", verbose=True):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        wind_loss = WindSpeedErrorLoss()
        angle_loss = AngErrorLoss()
        ssim_loss = SSIMErrorLoss()

        loss_u, loss_v = wind_loss.compute_wind_speed(y_pred, y_true)
        loss_angle = angle_loss.calculate_wind_direction(y_pred, y_true)
        loss_ssim = ssim_loss.ssim_loss_stable(y_pred, y_true)

        total_loss = loss_u + loss_v + loss_angle + loss_ssim
        return total_loss




if __name__ == "__main__":
    result_path = r"D:/风速预测/output/2026_03_test.npz"
    result = np.load(result_path)

    pred = result["pred"]
    true = result["true"]
    loss_obj = TotalWindLoss()
    tf_total_loss = loss_obj.call(tf.constant(true), tf.constant(pred)).numpy()
    print(tf_total_loss)
