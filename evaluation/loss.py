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
    def __init__(self,y_mean,y_scale,calm_threshold=0.5):
        self.y_mean = tf.constant(y_mean, dtype=tf.float32)
        self.y_scale = tf.constant(y_scale, dtype=tf.float32)
        self.calm_threshold = tf.constant(calm_threshold, dtype=tf.float32)

    def inverse_transform(self, y):
        return y * self.y_scale + self.y_mean
    
    def calculate_wind_direction(self, y_pred, y_true):
       
        pred_physical = self.inverse_transform(y_pred)
        true_physical = self.inverse_transform(y_true)

        pred_u = pred_physical[..., 0]
        pred_v = pred_physical[..., 1]

        true_u = true_physical[..., 0]
        true_v = true_physical[..., 1]

        # 计算风速
        pred_speed = tf.sqrt(pred_u ** 2 + pred_v ** 2)
        true_speed = tf.sqrt(true_u ** 2 + true_v ** 2)

        # 仅按真实风速定义有效格点，确保不同模型使用相同评价样本。
        valid_mask = tf.cast(
            true_speed >= self.calm_threshold,
            tf.float32,
        )

        # Wind direction in radians, in [-pi, pi].
        pred_dir = tf.atan2(pred_v, pred_u)
        true_dir = tf.atan2(true_v, true_u)

        # Signed shortest-arc circular difference, also in [-pi, pi].
        raw_diff = pred_dir - true_dir
        circular_diff = tf.atan2(tf.sin(raw_diff), tf.cos(raw_diff))
        direction_error = (1.0 - tf.cos(circular_diff))

        # Exclude calm true-wind locations from both numerator and denominator.
        masked_loss = direction_error * valid_mask
        valid_count = tf.reduce_sum(valid_mask)
        loss = tf.math.divide_no_nan(
            tf.reduce_sum(masked_loss),
            valid_count,
        )
        return loss


class SSIMErrorLoss:
    def __init__(self, window_size, sigma, channel, size_average=True):
        self.window_size = window_size
        self.sigma = sigma
        self.channel = channel
        self.size_average = size_average

    def create_gaussian(self):

        x = tf.range(self.window_size, dtype=tf.float32)
        gauss = tf.exp(-(x - (self.window_size // 2)) ** 2 / (2.0 * self.sigma ** 2))
        return gauss / tf.reduce_sum(gauss)

    def create_window(self,channel):
        _1D_window = tf.reshape(self.create_gaussian(), (-1, 1))
        _2D_window = tf.matmul(_1D_window, tf.transpose(_1D_window))
        _2D_window = tf.reshape(_2D_window, (self.window_size, self.window_size, 1, 1))

        window = tf.tile(_2D_window, [1, 1, channel, 1])

        return window

    def _ssim(self, img1, img2, window):
        window = tf.cast(window, img1.dtype)
        mu1 = tf.nn.depthwise_conv2d(img1, window, strides=[1, 1, 1, 1], padding="SAME")
        mu2 = tf.nn.depthwise_conv2d(img2, window, strides=[1, 1, 1, 1], padding="SAME")

        mu1_sq = tf.square(mu1)
        mu2_sq = tf.square(mu2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = tf.nn.depthwise_conv2d(img1 * img1, window, strides=[1, 1, 1, 1], padding="SAME") - mu1_sq
        sigma2_sq = tf.nn.depthwise_conv2d(img2 * img2, window, strides=[1, 1, 1, 1], padding="SAME") - mu2_sq

        sigma12 = tf.nn.depthwise_conv2d(img1 * img2, window, strides=[1, 1, 1, 1], padding="SAME") - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        if self.size_average:
            return tf.reduce_mean(ssim_map)
        else:
            return tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(ssim_map, axis=1), axis=1), axis=1)

    def SSIM(self, img1, img2):

        min_val = tf.reduce_min(img2)
        max_val = tf.reduce_max(img2)

        img1 = (img1 - min_val) / (max_val - min_val)
        img2 = (img2 - min_val) / (max_val - min_val)

        img1 = tf.clip_by_value(img1, 0.0, 1.0)
        img2 = tf.clip_by_value(img2,0.0,1.0)

        if len(img1.shape) == 3:
            img1 = tf.expand_dims(img1, 0)
        if len(img2.shape) == 3:
            img2 = tf.expand_dims(img2, 0)

        channel = img1.shape[-1] if img1.shape[-1] is not None else tf.shape(img1)[1]

        window = self.create_window(channel)
        window = tf.cast(window, tf.float32)

        return self._ssim(img1, img2, window)

    def run(self, y_pred, y_true):
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        pred_np = tf.transpose(y_pred, [1, 0, 2, 3, 4])
        true_np = tf.transpose(y_true, [1, 0, 2, 3, 4])

        total_loss = tf.constant(0.0, dtype=tf.float32)

        for i in range(pred_np.shape[0]):
            loss = 1 - self.SSIM(pred_np[i], true_np[i])
            total_loss += loss

        average_loss = total_loss / tf.cast(tf.shape(pred_np)[0], tf.float32)

        return average_loss


class TotalWindLoss(tf.keras.losses.Loss):
    def __init__(self,
                 y_mean,
                 y_scale,
                 calm_threshold=1.0,
                 name="total_wind_loss"):
        super().__init__(name=name)
        self.y_mean = list(y_mean)
        self.scale = list(y_scale)
        self.calm_threshold = calm_threshold

        self.log_sigma_uv = tf.Variable(
            initial_value=0.0,
            trainable=True,
            dtype=tf.float32,
            name="log_sigma_uv"
        )

        self.log_sigma_angle = tf.Variable(
            initial_value=2.3,
            trainable=True,
            dtype=tf.float32,
            name="log_sigma_angle"
        )

        self.log_sigma_ssim = tf.Variable(
            initial_value=0.69,
            trainable=True,
            dtype=tf.float32,
            name="log_sigma_ssim"
        )

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        wind_loss = WindSpeedErrorLoss()
        angle_loss = AngErrorLoss(y_mean=self.y_mean,
                                  y_scale=self.scale,
                                  calm_threshold=self.calm_threshold
            )
        ssim_loss = SSIMErrorLoss(window_size=3, sigma=1.5, channel=2, size_average=True)

        loss_u, loss_v = wind_loss.compute_wind_speed(y_pred, y_true)
        loss_uv = loss_u + loss_v
        loss_angle = angle_loss.calculate_wind_direction(y_pred, y_true)
        loss_ssim = ssim_loss.run(y_pred, y_true)

        total_loss = tf.exp(-self.log_sigma_uv)*loss_uv + self.log_sigma_uv +\
                    tf.exp(-self.log_sigma_angle)*loss_angle + self.log_sigma_angle +\
                    tf.exp(-self.log_sigma_ssim)*loss_ssim + self.log_sigma_ssim
        return total_loss
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "y_mean": self.y_mean,
                "y_scale": self.y_scale,
                "calm_threshold": self.calm_threshold,
            }
        )
        return config




if __name__ == "__main__":
    result_path = r"D:/风速预测/output/2026_03_test.npz"
    result = np.load(result_path)

    pred = result["pred"]
    true = result["true"]
    loss_obj = TotalWindLoss()
    tf_total_loss = loss_obj.call(tf.constant(true), tf.constant(pred)).numpy()
    print(tf_total_loss)
